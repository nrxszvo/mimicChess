import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
    - max_rows_per_file: cap of rows per parquet file per epoch
    - encoding_name: tiktoken encoding name (e.g., 'cl100k_base')
    - columns: override columns to read; defaults to exactly what's needed
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        min_timectl: int,
        max_rows_per_file: int,
        encoder_params,
        columns: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.min_timectl = int(min_timectl)
        self.max_rows_per_file = int(max_rows_per_file)
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

        # Discover all parquet files at depth 2 named data.parquet
        self.files: List[Path] = []
        for elo_w_dir in sorted(self.root_dir.iterdir()):
            if not elo_w_dir.is_dir():
                continue
            for elo_b_dir in sorted(elo_w_dir.iterdir()):
                if not elo_b_dir.is_dir():
                    continue
                pfile = elo_b_dir / "data.parquet"
                if pfile.exists():
                    self.files.append(pfile)

        if not self.files:
            raise FileNotFoundError(
                f"No parquet files found under {self.root_dir} (expected */*/data.parquet)"
            )

        # Count rows meeting filter in each file
        self._filter = ds.field("timeCtl") >= pa.scalar(self.min_timectl, pa.int16())
        self.file_counts: Dict[Path, int] = {}
        for p in self.files:
            dset = ds.dataset(str(p), format="parquet")
            try:
                n = dset.count_rows(filter=self._filter)
            except Exception:
                # Fallback if driver lacks count_rows filter
                tbl = dset.to_table(columns=["timeCtl"])  # small column
                tc = tbl.column("timeCtl")
                n = sum(
                    1 for v in tc.to_pylist() if v is not None and v >= self.min_timectl
                )
            self.file_counts[p] = n

        # Per-epoch selected, materialized (reduced) tables per file
        self._epoch_tables: Dict[Path, pa.Table] = {}
        self._cumulative: List[Tuple[Path, int]] = []  # (file, length_of_table)

        self.refresh_epoch()

    def refresh_epoch(self, seed: Optional[int] = None) -> None:
        """(Re)build the epoch sampling plan.

        For each file: materialize only the filtered rows, then cap to
        max_rows_per_file via random sampling (without replacement). Stores a
        reduced in-memory table per file for fast __getitem__.
        """
        if seed is not None:
            random.seed(seed)

        self._epoch_tables.clear()
        self._cumulative.clear()

        for p in self.files:
            count = self.file_counts[p]
            if count <= 0:
                continue

            # Load filtered rows and limit the columns to what's needed
            dset = ds.dataset(str(p), format="parquet")
            scanner = dset.scanner(filter=self._filter, columns=self.columns)
            table = scanner.to_table()

            # Cap rows per file by random sampling (stable order after take)
            if len(table) > self.max_rows_per_file:
                idxs = sorted(random.sample(range(len(table)), self.max_rows_per_file))
                table = table.take(pa.array(idxs, type=pa.int64()))

            self._epoch_tables[p] = table
            self._cumulative.append((p, len(table)))

        # Remove files with zero rows this epoch
        self._cumulative = [(p, n) for (p, n) in self._cumulative if n > 0]

        # Prefix sums for fast index mapping
        total = 0
        self._prefix: List[Tuple[Path, int, int]] = []  # (file, start, length)
        for p, n in self._cumulative:
            self._prefix.append((p, total, n))
            total += n
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate(self, idx: int) -> Tuple[Path, int]:
        if idx < 0 or idx >= self._length:
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
                return p, idx - start
        # Should not happen
        raise RuntimeError("Index mapping failed")

    @staticmethod
    def _interleave_tokens(moves_tokens: List[str], clk_tokens: List[str]) -> List[str]:
        out: List[str] = []
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
        p, local_idx = self._locate(idx)
        table = self._epoch_tables[p]

        row = table.slice(local_idx, 1)

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
        max_rows_per_file,
        encoder_params,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.min_timectl = min_timectl
        self.max_rows_per_file = max_rows_per_file
        self.encoder_params = encoder_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.NOOP = encoder_params['special_tokens']['<|NOOP|>']


    def setup(self, stage):
        if stage == "fit":
            self.trainset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_rows_per_file,
                self.encoder_params,
            )
            self.valset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_rows_per_file,
                self.encoder_params,
            )
        if stage == "validate":
            self.valset = ChessParquetDataset(
                self.root_dir,
                self.min_timectl,
                self.max_rows_per_file,
                self.encoder_params,
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
