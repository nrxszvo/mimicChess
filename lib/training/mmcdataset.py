import json
import os
from collections import OrderedDict

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..pgnutils import NOOP, STARTMV


def init_worker(seed):
    np.random.seed(seed)


def collate_fn(batch):
    maxinp = 0
    openmoves = batch[0]["opening"].shape[0]
    for d in batch:
        maxinp = max(maxinp, d["nmoves"])

    maxtgt = maxinp - openmoves
    bs = len(batch)
    inputs = torch.full((bs, maxinp), NOOP, dtype=torch.int32)
    openings = torch.empty((bs, openmoves), dtype=torch.int64)
    elo_targets = torch.full(
        (bs, maxtgt),
        NOOP,
        dtype=torch.float32
        if batch[0]["elo_target"].dtype == "float32"
        else torch.int64,
    )
    move_targets = torch.full((bs, maxtgt), NOOP, dtype=torch.int64)
    tc_groups = torch.empty(bs, dtype=torch.int64)
    elo_groups = torch.empty((bs, 2), dtype=torch.int64)

    for i, d in enumerate(batch):
        inp = d["input"]
        elo_tgt = d["elo_target"]
        mv_tgt = d["move_target"]
        inputs[i, : inp.shape[0]] = torch.from_numpy(inp)
        elo_targets[i, : elo_tgt.shape[0]] = torch.from_numpy(elo_tgt)
        move_targets[i, : mv_tgt.shape[0]] = torch.from_numpy(mv_tgt)
        openings[i] = torch.from_numpy(d["opening"])
        tc_groups[i] = d["tc_id"]
        elo_groups[i] = torch.tensor(d["elo_groups"], dtype=torch.int64)

    return {
        "input": inputs,
        "elo_target": elo_targets,
        "move_target": move_targets,
        "opening": openings,
        "tc_groups": tc_groups,
        "elo_groups": elo_groups,
    }


class MMCDataset(Dataset):
    def __init__(
        self,
        seq_len,
        opening_moves,
        indices,
        blocks,
        elo_edges,
        tc_groups,
        whiten_params,
        max_nsamp=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.opening_moves = opening_moves
        self.indices = indices
        self.nsamp = len(self.indices)
        if max_nsamp is not None:
            self.nsamp = min(max_nsamp, self.nsamp)

        self.blocks = blocks
        self.elo_edges = elo_edges
        self.tc_groups = tc_groups
        self.whiten_params = whiten_params

    def __len__(self):
        return self.nsamp

    def _get_tc_id(self, tc, inc):
        for tcgrp, incgrps in self.tc_groups.items():
            if tc <= tcgrp:
                for incgrp, tcid in incgrps.items():
                    if inc <= incgrp:
                        return tcid

    def _get_group(self, elo):
        for i, edge in enumerate(self.elo_edges):
            if elo <= edge:
                return i

    def __getitem__(self, idx):
        gidx, gs, nmoves, blk = self.indices[idx]

        mvids = self.blocks[blk]["mvids"]
        welo = self.blocks[blk]["welos"][gidx]
        belo = self.blocks[blk]["belos"][gidx]
        timectl = self.blocks[blk]["timectl"][gidx]
        inc = self.blocks[blk]["increment"][gidx]

        nmoves = min(self.seq_len, nmoves)
        inp = np.empty(nmoves, dtype="int32")
        inp[0] = STARTMV
        inp[1:] = mvids[gs: gs + nmoves - 1]

        opening = np.empty(self.opening_moves, dtype="int64")
        opening[:] = mvids[gs: gs + self.opening_moves]

        elo_tgt = np.empty(
            nmoves - self.opening_moves,
            dtype="float32" if self.whiten_params is not None else "int64",
        )
        if self.whiten_params is not None:
            mu, s = self.whiten_params
            elo_tgt[::2] = (welo - mu) / s
            elo_tgt[1::2] = (belo - mu) / s

        welo_grp = self._get_group(welo)
        belo_grp = self._get_group(belo)

        mv_tgt = np.empty(nmoves - self.opening_moves, dtype="int64")
        mv_tgt[:] = mvids[gs + self.opening_moves: gs + nmoves]

        tc_id = self._get_tc_id(timectl, inc)

        return {
            "input": inp,
            "elo_target": elo_tgt,
            "move_target": mv_tgt,
            "opening": opening,
            "nmoves": nmoves,
            "tc_id": tc_id,
            "elo_groups": [welo_grp, belo_grp],
        }


def load_data(dirname):
    with open(f"{dirname}/fmd.json") as f:
        fmd = json.load(f)
    data = {
        "fmd": fmd,
        "train": np.memmap(
            os.path.join(dirname, "train.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["train_shape"]),
        )
        if fmd["train_shape"][0] > 0
        else [],
        "val": np.memmap(
            os.path.join(dirname, "val.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["val_shape"]),
        )
        if fmd["val_shape"][0] > 0
        else [],
        "test": np.memmap(
            os.path.join(dirname, "test.npy"),
            mode="r",
            dtype="int64",
            shape=tuple(fmd["test_shape"]),
        ),
    }
    blocks = []
    for blkdn in fmd["block_dirs"]:
        dn = os.path.join(dirname, blkdn)
        md = np.load(os.path.join(dn, "md.npy"), allow_pickle=True).item()
        blocks.append({
            "md": md,
            "welos": np.memmap(
                os.path.join(dn, "welos.npy"),
                mode="r",
                dtype="int16",
                shape=md["ngames"],
            ),
            "belos": np.memmap(
                os.path.join(dn, "belos.npy"),
                mode="r",
                dtype="int16",
                shape=md["ngames"],
            ),
            "mvids": np.memmap(
                os.path.join(dn, "mvids.npy"),
                mode="r",
                dtype="int16",
                shape=md["nmoves"],
            ),
            "timectl": np.memmap(
                os.path.join(dn, "timeCtl.npy"), mode="r", dtype="int16"
            ),
            "increment": np.memmap(
                os.path.join(dn, "inc.npy"), mode="r", dtype="int16"
            ),
        })
    data["blocks"] = blocks
    return data


class MMCDataModule(L.LightningDataModule):
    def _validate_tc_groups(self):
        tc_histo = {
            int(tc): {int(inc): n for inc, n in incs.items()}
            for tc, incs in self.fmd["tc_histo"].items()
        }
        for tc, incs in tc_histo.items():
            for tcg, incgs in self.tc_groups.items():
                if tc <= tcg:
                    for inc in incs:
                        for incg in incgs:
                            if inc <= incg:
                                break
                        else:
                            raise Exception(
                                f"increment {inc} for tc group {tcg} is greater than largest inc group"
                            )
                    break
            else:
                raise Exception(
                    f"time control {tc} is greater than largest tc group")

    def _init_tc_groups(self, tc_groups):
        self.tc_groups = OrderedDict()
        tcid = 0
        for tcg, incgs in sorted(tc_groups.items()):
            self.tc_groups[tcg] = OrderedDict()
            for inc in sorted(incgs):
                self.tc_groups[tcg][inc] = tcid
                tcid += 1
        self.n_tc_groups = tcid
        self._validate_tc_groups()

    def __init__(
        self,
        datadir,
        elo_edges,
        tc_groups,
        max_seq_len,
        batch_size,
        num_workers,
        whiten_params=None,
        max_testsamp=None,
        opening_moves=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.elo_edges = elo_edges
        self.whiten_params = whiten_params
        self.__dict__.update(load_data(datadir))
        self._init_tc_groups(tc_groups)
        # min_moves is the minimum game length that can be included in the dataset
        # we subtract one here so that it now represents the minimum number of moves that the
        # model must see before making its first prediction
        if opening_moves is None:
            self.opening_moves = self.fmd["min_moves"] - 1
        else:
            self.opening_moves = opening_moves

        self.max_testsamp = max_testsamp

    def setup(self, stage):
        if stage == "fit":
            self.trainset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.train,
                self.blocks,
                self.elo_edges,
                self.tc_groups,
                self.whiten_params,
            )
            self.valset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.val,
                self.blocks,
                self.elo_edges,
                self.tc_groups,
                self.whiten_params,
            )
        if stage == "validate":
            self.valset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.val,
                self.blocks,
                self.elo_edges,
                self.tc_groups,
                self.whiten_params,
            )

        if stage in ["test", "predict"]:
            self.testset = MMCDataset(
                self.max_seq_len,
                self.opening_moves,
                self.test,
                self.blocks,
                self.elo_edges,
                self.tc_groups,
                self.whiten_params,
                self.max_testsamp,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=init_worker,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.testset,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return self.predict_dataloader()
