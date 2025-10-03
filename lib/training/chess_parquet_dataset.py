import os
import random
from pathlib import Path
from typing import Tuple, Optional
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

import pyarrow as pa
import pyarrow.dataset as ds
import tiktoken
import numpy as np
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader


def init_worker(seed):
    np.random.seed(seed)


def discover_storage_files(root_dir: Path) -> list[Path]:
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

        for second_level in sorted(first_level.iterdir()):
            if not second_level.is_dir():
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

        return count
    except Exception as e:
        print(f"Error counting rows in {file_path}: {e}")
        return 0


def get_file_counts_with_repeats(
    storage_files: list[Path], min_timectl: int, max_repeats: int
) -> Tuple[int, dict[Path, int]]:
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

    print("Counting filtered rows in each storage file...")

    for file_path in storage_files:
        count = count_filtered_rows(file_path, min_timectl)
        file_counts[file_path] = count
        max_rows = max(max_rows, count)
        print(f"{file_path}: {count} filtered rows")

    for fp in file_counts:
        file_counts[fp] = min(max_rows, max_repeats * file_counts[fp])

    print(f"Maximum filtered rows in any file: {max_rows:.2e}")
    return file_counts


class FilteredRowIterator:
    """Memory-efficient iterator that streams filtered rows from a parquet file, cycling when exhausted."""

    def __init__(
        self, file_path: Path, min_timectl: int, max_rows: int, batch_size: int = 1000
    ):
        self.file_path = file_path
        self.min_timectl = min_timectl
        self.filter_expr = ds.field("timeCtl") >= pa.scalar(min_timectl, pa.int16())
        self.batch_size = batch_size
        self.total_filtered_rows = 0
        self.max_rows = max_rows
        self.dataset = ds.dataset(str(self.file_path), format="parquet")
        # create schema that is subset of dataset schema
        self.columns = [
            "moves",
            "clk",
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
        self.refresh_epoch()
        self._count_total_rows()
        self._load_next_batch()

    def refresh_epoch(self):
        self.current_batch = None
        self.current_batch_index = 0
        self.global_index = 0
        self._total_rows_out = 0

    def _count_total_rows(self):
        """Count total filtered rows for cycling logic."""
        self.total_filtered_rows = self.dataset.count_rows(filter=self.filter_expr)
        print(f"Found {self.total_filtered_rows} filtered rows in {self.file_path}")

    def _load_next_batch(self):
        """Load the next batch of filtered rows."""
        if self.total_filtered_rows == 0 or self._total_rows_out >= self.max_rows:
            self.current_batch = None
            return

        # Calculate which batch to load based on global index
        batch_start = (self.global_index // self.batch_size) * self.batch_size

        # Create scanner with filter and limit
        scanner = self.dataset.scanner(
            filter=self.filter_expr,
            columns=self.columns,
        )
        # create empty batch with number of arrays equal to schema
        arrays = [pa.array([], type=field.type) for field in self.schema]
        self.current_batch = pa.Table.from_arrays(arrays, schema=self.schema)
        while len(self.current_batch) < self.batch_size:
            indices = pa.array(
                range(
                    batch_start,
                    min(batch_start + self.batch_size, self.total_filtered_rows),
                )
            )
            # concatenate batch
            self.current_batch = pa.concat_tables(
                [self.current_batch, scanner.take(indices)]
            )
            batch_start = (batch_start + len(indices)) % self.total_filtered_rows
            self._total_rows_out += len(indices)
            if self._total_rows_out >= self.max_rows:
                break

        self.current_batch_index = 0

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
        self.global_index = (self.global_index + 1) % self.total_filtered_rows

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
        min_timectl: int,
        max_repeats: int,
        encoder_params,
        columns: Optional[list[str]] = None,
        iterator_batch_size: int = 1000,
        mode: str = None,
        valp: float = 0.05,
        testp: float = 0.05,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.min_timectl = int(min_timectl)
        self.max_repeats = int(max_repeats)
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
                "clk",
                "welo",
                "belo",
                "increment",
                "result",
                "timeCtl",
            ]
        )

        self.files: list[Path] = discover_storage_files(self.root_dir)
        self.file_counts = get_file_counts_with_repeats(
            self.files, self.min_timectl, self.max_repeats
        )
        self._prefix: list[Tuple[Path, int, int]] = []  # (file, start, length)
        total = 0
        for p, n in self.file_counts.items():
            self._prefix.append((p, total, n))
            total += n

        if mode == "train":
            self._offset = 0
            self._length = int(total * (1 - valp - testp))
        elif mode == "val":
            self._offset = int(total * (1 - valp - testp))
            self._length = int(total * valp)
        elif mode == "test":
            self._offset = int(total * (1 - testp))
            self._length = int(total * testp)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        self.iterators = []
        for file_path, count in self.file_counts.items():
            self.iterators.append(
                FilteredRowIterator(file_path, min_timectl, count, iterator_batch_size)
            )

        self.refresh_epoch()

    def refresh_epoch(self, seed: Optional[int] = None) -> None:
        for iterator in self.iterators:
            iterator.refresh_epoch()

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> Tuple[Path, int]:
        if idx < 0 or idx >= self._offset + self._length:
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
                #return p, idx - start
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

    def _to_special_token(self, val: int) -> str:
        if val == 0:
            return "<|WHITEWINS|>"
        elif val == 1:
            return "<|BLACKWINS|>"
        elif val == 2:
            return "<|DRAW|>"
        elif val <= 1000:
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
        else:
            return "<|ELOMAX|>"

    def __getitem__(self, idx: int) -> torch.Tensor:
        idx += self._offset
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

        moves: str = get_col("moves") or ""
        clk: Optional[str] = get_col("clk")
        welo = self._to_special_token(get_col("welo"))
        belo = self._to_special_token(get_col("belo"))
        inc = get_col("increment")
        res = self._to_special_token(get_col("result"))

        header_parts = [
            str(welo),
            str(belo),
            str(inc),
            str(res),
        ]
        header = " ".join(header_parts)

        moves_tokens = moves.split()
        clk_tokens = clk.split() if isinstance(clk, str) and len(clk) > 0 else []
        interleaved = self._interleave_tokens(moves_tokens, clk_tokens)

        body = " ".join(interleaved)
        full = f"{header} {body}" if body else header

        ids = self.encoding.encode(full, allowed_special="all")
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(NOOP, batch):
    maxinp = 0
    for ids in batch:
        maxinp = max(maxinp, len(ids))

    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.long)

    for i, ids in enumerate(batch):
        inputs[i, : ids.shape[0]] = ids

    return inputs


class MMCDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir,
        min_timectl,
        max_repeats,
        encoder_params,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.min_timectl = min_timectl
        self.max_repeats = max_repeats
        self.encoder_params = encoder_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.NOOP = encoder_params["special_tokens"]["<|NOOP|>"]

    def setup(self, stage):
        if stage == "fit":
            self.trainset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_repeats,
                self.encoder_params,
                mode='train',
            )
            self.valset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_repeats,
                self.encoder_params,
                mode='val',
            )
        if stage == "validate":
            self.valset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_repeats,
                self.encoder_params,
                mode='val',
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
