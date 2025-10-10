import os
from pathlib import Path
from typing import Tuple, Optional
from functools import partial
import regex as re

import torch
from torch.utils.data import Dataset, DataLoader

import pyarrow as pa
import pyarrow.dataset as ds
import tiktoken
import numpy as np
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader
import torch.distributed as dist


def init_worker(seed):
    np.random.seed(seed)


def parse_directory_name(dir_name: str) -> Tuple[int, bool]:
    """
    Parse directory name to extract numeric value and plus suffix.

    Args:
        dir_name: Directory name (e.g., "1000", "2500+")

    Returns:
        Tuple of (numeric_value, has_plus_suffix)
    """
    match = re.match(r"^(\d+)(\+?)$", dir_name)
    if not match:
        raise ValueError(f"Invalid directory name format: {dir_name}")

    numeric_value = int(match.group(1))
    has_plus = bool(match.group(2))

    return numeric_value, has_plus


def discover_storage_files(root_dir: Path, max_elo_group: int) -> list[Path]:
    """
    Discover all parquet files in the two-level directory structure.

    Args:
        root_dir: Root directory containing first-level folders (1000, 1500, etc.)

    Returns:
        List of paths to data.parquet files
    """
    storage_files = []

    for first_level in sorted(root_dir.iterdir()):
        if not first_level.is_dir():
            continue
        numeric_val, _ = parse_directory_name(first_level.name)
        if numeric_val > max_elo_group:
            continue

        for second_level in sorted(first_level.iterdir()):
            if not second_level.is_dir():
                continue
            numeric_val, _ = parse_directory_name(second_level.name)
            if numeric_val > max_elo_group:
                continue
            parquet_file = second_level / "data.parquet"
            if parquet_file.exists():
                storage_files.append(parquet_file)

    if not storage_files:
        raise FileNotFoundError(f"No data.parquet files found under {root_dir}")

    print(f"Found {len(storage_files)} storage files")
    return storage_files


def count_filtered_rows(file_path: Path, min_timectl: int) -> int:
    """
    Count rows in a parquet file that meet the timeCtl filter requirement.

    Args:
        file_path: Path to the parquet file
        min_timectl: Minimum timeCtl value required

    Returns:
        Number of rows meeting the filter criteria
    """
    try:
        dataset = ds.dataset(str(file_path), format="parquet")
        filter_expr = ds.field("timeCtl") >= pa.scalar(min_timectl, pa.int16())

        try:
            count = dataset.count_rows(filter=filter_expr)
        except Exception:
            # Fallback if driver lacks count_rows filter support
            table = dataset.to_table(columns=["timeCtl"])
            timectl_col = table.column("timeCtl")
            count = sum(
                1 for v in timectl_col.to_pylist() if v is not None and v >= min_timectl
            )

        return count // dist.get_world_size()
    except Exception as e:
        print(f"Error counting rows in {file_path}: {e}")
        return 0


def get_file_counts_with_repeats(
    storage_files: list[Path], min_timectl: int, max_repeats: int
) -> dict[Path, int]:
    """
    Find the maximum number of filtered rows across all storage files.

    Args:
        storage_files: List of storage file paths
        min_timectl: Minimum timeCtl value required

    Returns:
        Maximum number of filtered rows in any single file
    """
    max_rows = 0
    file_counts = {}

    for file_path in storage_files:
        count = count_filtered_rows(file_path, min_timectl)
        file_counts[file_path] = count 
        max_rows = max(max_rows, count)

    for fp in file_counts:
        file_counts[fp] = min(max_rows, max_repeats * file_counts[fp])

    print(f"Maximum filtered rows in any file: {max_rows:.2e}")
    return file_counts


class FilteredRowIterator:
    """Memory-efficient iterator that streams filtered rows from a parquet file, cycling when exhausted."""

    def __init__(
        self,
        file_path: Path,
        min_timectl: int,
        max_repeats: int,
        batch_size: int = 1_000_000,
    ):
        self.file_path = file_path
        self.min_timectl = min_timectl
        self.filter_expr = ds.field("timeCtl") >= pa.scalar(min_timectl, pa.int16())
        self.max_repeats = max_repeats
        self.batch_size = batch_size
        self.dataset = ds.dataset(str(self.file_path), format="parquet")
        # create schema that is subset of dataset schema
        self.columns = [
            "moves",
            #"clk",
            "result",
            "welo",
            "belo",
            "timeCtl",
            "increment",
        ]
        self.schema = pa.schema(
            {
                field.name: field.type
                for field in self.dataset.schema
                if field.name in self.columns
            }
        )
        # Create scanner with filter and limit
        self.scanner = self.dataset.scanner(
            filter=self.filter_expr,
            columns=self.columns,
        )
        self._count_total_rows()
        self.max_rows = self.max_repeats * self.total_filtered_rows
        self.refresh_epoch()

    def refresh_epoch(self):
        self.current_batch = None
        self.current_batch_index = 0
        self.batch_start = self.start_index
        self.total_rows_out = 0

    def _count_total_rows(self):
        """Count total filtered rows for cycling logic."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        total_filtered_rows = self.dataset.count_rows(filter=self.filter_expr)
        self.total_filtered_rows = total_filtered_rows // world_size
        self.start_index = rank * self.total_filtered_rows
        self.end_index = min((rank + 1) * self.total_filtered_rows, total_filtered_rows)


    def _load_next_batch(self):
        """Load the next batch of filtered rows."""
        if self.total_filtered_rows == 0 or self.total_rows_out >= self.max_rows:
            self.current_batch = None
            return

        indices = pa.array(
            range(
                self.batch_start,
                min(self.batch_start + self.batch_size, self.end_index),
            )
        )
        if len(indices) < self.batch_size:
            # concatenate remaining indices from start_index
            rest = self.batch_size - len(indices)
            indices = pa.concat_arrays(
                [indices, pa.array(range(self.start_index, self.start_index + rest))]
            )
            self.batch_start = self.start_index + rest
        else:
            self.batch_start += self.batch_size

        self.current_batch = self.scanner.take(indices)
        self.current_batch_index = 0
        self.total_rows_out += len(indices)

    def get_next_row(self) -> Optional[pa.Table]:
        """Get the next filtered row, cycling back to start if needed."""
        if self.total_filtered_rows == 0:
            return None

        # Check if we need to load a new batch
        if self.current_batch is None or self.current_batch_index >= len(
            self.current_batch
        ):
            self._load_next_batch()

        if self.current_batch is None or len(self.current_batch) == 0:
            return None

        # Get current row from batch
        row = self.current_batch.slice(self.current_batch_index, 1)

        # Advance indices
        self.current_batch_index += 1

        return row


class ChessParquetDataset(Dataset):
    """
    PyTorch Dataset to read chess game data from a two-level Elo-bucket directory
    structure containing Parquet files. Each leaf directory contains exactly one
    file named "data.parquet" with the schema:

        moves (string)
        clk (string, optional)
        eval (string, optional)
        result (int8, optional)
        welo (int16, optional)
        belo (int16, optional)
        white (string, optional)
        black (string, optional)
        timeCtl (int16, optional)
        increment (int16, optional)

    Parameters
    - root_dir: top-level directory containing per-Elo buckets
    - min_timectl: only include rows with timeCtl >= min_timectl
    - max_repeats: cap of rows per parquet file per epoch
    - encoding_name: tiktoken encoding name (e.g., 'cl100k_base')
    - columns: override columns to read; defaults to exactly what's needed
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        max_elo_group: int,
        min_timectl: int,
        max_repeats: int,
        max_seq_len: int,
        encoder_params: dict,
        columns: Optional[list[str]] = None,
        valp: float = 0.05,
        testp: float = 0.05,
        max_rows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.max_elo_group = max_elo_group
        self.min_timectl = int(min_timectl)
        self.max_repeats = int(max_repeats)
        self.max_seq_len = (int(max_seq_len) - 4)
        self.encoder_params = encoder_params
        ranks = {}
        for tok_str, tok_id in encoder_params["ranks"].items():
            # convert tok_str to bytes
            tok_str = tok_str.encode("utf-8")
            ranks[tok_str] = tok_id

        self.encoding = tiktoken.Encoding(
            "pgn",
            pat_str=encoder_params["pat_str"],
            mergeable_ranks=ranks,
            special_tokens=encoder_params["special_tokens"],
        )

        self.columns = (
            columns
            if columns is not None
            else [
                "moves",
                #"clk",
                "welo",
                "belo",
                "increment",
                "result",
                "timeCtl",
            ]
        )

        self.files: list[Path] = discover_storage_files(
            self.root_dir, self.max_elo_group
        )
        self.file_counts = get_file_counts_with_repeats(
            self.files, self.min_timectl, self.max_repeats
        )
        self._prefix: list[Tuple[Path, int, int]] = []  # (file, start, length)
        total = 0
        for p, n in self.file_counts.items():
            self._prefix.append((p, total, n))
            total += n

        self._total = max_rows if max_rows is not None else total
        self._valp = valp
        self._testp = testp
        self._trainp = 1 - valp - testp
        self.iterators = []
        for file_path, _ in self.file_counts.items():
            self.iterators.append(
                FilteredRowIterator(file_path, min_timectl, max_repeats)
            )

    def refresh_epoch(self) -> None:
        for iterator in self.iterators:
            iterator.refresh_epoch()

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int) -> int:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        # binary search over prefix
        lo, hi = 0, len(self._prefix) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            p, start, n = self._prefix[mid]
            if idx < start:
                hi = mid - 1
            elif idx >= start + n:
                lo = mid + 1
            else:
                return mid
                # return p, idx - start
        # Should not happen
        raise RuntimeError("Index mapping failed")

    @staticmethod
    def _interleave_tokens(moves_tokens: list[str], clk_tokens: list[str]) -> list[str]:
        out: list[str] = []
        L = max(len(moves_tokens), len(clk_tokens))
        for i in range(L):
            if i < len(moves_tokens):
                out.append(moves_tokens[i])
            if i < len(clk_tokens):
                out.append(clk_tokens[i])
        return out

    def _outcome(self, val: int) -> int:
        if val == 0:
            return 1
            #return "<|WHITEWINS|>"
        elif val == 1:
            return -1
            #return "<|BLACKWINS|>"
        elif val == 2:
            return 0
            #return "<|DRAW|>"
        else:
            raise Exception("Invalid outcome")
        
    def _elo_token(self, val: int) -> str:
        if val <= self.max_elo_group:
            if val <= 1000:
                return "<|ELO1000|>"
            elif val <= 1200:
                return "<|ELO1200|>"
            elif val <= 1400:
                return "<|ELO1400|>"
            elif val <= 1600:
                return "<|ELO1600|>"
            elif val <= 1800:
                return "<|ELO1800|>"
            elif val <= 2000:
                return "<|ELO2000|>"
            elif val <= 2200:
                return "<|ELO2200|>"
            elif val <= 2400:
                return "<|ELO2400|>"
            elif val <= 2600:
                return "<|ELO2600|>"
            elif val <= 2800:
                return "<|ELO2800|>"
            elif val <= 3000:
                return "<|ELO3000|>"
        return "<|ELOMAX|>"

    def _timeCtl_token(self, val: int) -> str:
        if val <= 60:
            return "<|1MINUTE|>"
        elif val <= 180:
            return "<|3MINUTES|>"
        elif val <= 300:
            return "<|5MINUTES|>"
        elif val <= 600:
            return "<|10MINUTES|>"
        elif val <= 900:
            return "<|15MINUTES|>"
        elif val <= 1800:
            return "<|30MINUTES|>"
        elif val <= 3600:
            return "<|1HOUR|>"
        else:
            return "<|3HOURS|>"

    def _increment_token(self, val: int) -> str:
        if val == 0:
            return "<|0SECONDS|>"
        elif val == 1:
            return "<|1SECOND|>"
        elif val <= 5:
            return "<|5SECONDS|>"
        elif val <= 10:
            return "<|10SECONDS|>"
        elif val <= 15:
            return "<|15SECONDS|>"
        elif val == 30:
            return "<|30SECONDS|>"
        else:
            return "<|MAXSECONDS|>"

    def __getitem__(self, idx: int) -> torch.Tensor:
        local_idx = self._locate(idx)
        start_idx = local_idx
        while True:
            it = self.iterators[local_idx]
            row = it.get_next_row()
            if row is not None:
                break
            local_idx = (local_idx + 1) % len(self.iterators)
            if local_idx == start_idx:
                raise RuntimeError("No rows found")

        def get_col(name: str):
            return row[name].to_pylist()[0] if name in row.schema.names else None

        moves = get_col("moves") 
        #clk = get_col("clk")
        res = self._outcome(get_col("result"))

        header_parts = [
            self._elo_token(get_col("welo")),
            self._elo_token(get_col("belo")),
            self._timeCtl_token(get_col("timeCtl")),
            self._increment_token(get_col("increment")),
        ]
        header = " ".join(header_parts)

        moves_tokens = moves.split()
        #clk_tokens = clk.split()
        #assert len(moves_tokens) == len(clk_tokens)
        #interleaved = self._interleave_tokens(moves_tokens, clk_tokens)
        interleaved = moves_tokens

        body = " ".join(interleaved)
        full = f"{header} {body}"
        ids = self.encoding.encode(full, allowed_special="all")
        ids = torch.tensor(ids, dtype=torch.long)
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        return ids, res


def collate_fn(NOOP, batch):
    maxinp = 0
    for ids, res in batch:
        maxinp = max(maxinp, len(ids))

    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.long)
    results = torch.empty((bs,), dtype=torch.float)

    for i, (ids, res) in enumerate(batch):
        inputs[i, : ids.shape[0]] = ids
        results[i] = res

    return inputs, results


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir,
        max_elo_group,
        min_timectl,
        max_repeats,
        max_seq_len,
        encoder_params,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.max_elo_group = max_elo_group
        self.min_timectl = min_timectl
        self.max_repeats = max_repeats
        self.max_seq_len = max_seq_len
        self.encoder_params = encoder_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.NOOP = encoder_params["special_tokens"]["<|NOOP|>"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = ChessParquetDataset(
                self.root_dir / "train",
                self.max_elo_group,
                self.min_timectl,
                self.max_repeats,
                self.max_seq_len,
                self.encoder_params,
            )
            self.valset = ChessParquetDataset(
                self.root_dir / "val",
                self.max_elo_group,
                self.min_timectl,
                1,
                self.max_seq_len,
                self.encoder_params,
                max_rows=5_000_000 // self.batch_size
            )
        if stage == "validate":
            self.valset = ChessParquetDataset(
                self.root_dir / "val",
                self.max_elo_group,
                self.min_timectl,
                1,
                self.max_seq_len,
                self.encoder_params,
                max_rows=5_000_000 // self.batch_size
            )

    def train_dataloader(self):
        return StatefulDataLoader(
            self.trainset,
            collate_fn=partial(collate_fn, self.NOOP),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_worker,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            collate_fn=partial(collate_fn, self.NOOP),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
