import os
import json

import torch

from .mmc import MimicChessModule, MMCModuleArgs
from .mmcdataset import MMCDataModule
from .model import ModelArgs


def get_model_args(cfgyml):
    model_args = ModelArgs(cfgyml.model_args.__dict__)
    model_args.vocab_size += len(cfgyml.elo_groups)
    return model_args


def init_modules(
    cfgyml,
    name,
    strategy,
    devices,
    num_nodes=1,
    alt_datadir=None,
    n_samp=None,
    cp=None,
    outdir=None,
    n_workers=None,
):
    if n_workers is None:
        n_workers = os.cpu_count() // devices if torch.cuda.is_available() else 0
    datadir = cfgyml.datadir if alt_datadir is None else alt_datadir
    model_args = get_model_args(cfgyml)

    assert cfgyml.global_batch_size % cfgyml.accumulate_grad_batches == 0
    batch_size = int(cfgyml.global_batch_size /
                     cfgyml.accumulate_grad_batches)

    dm = MMCDataModule(
        datadir=datadir,
        elo_edges=cfgyml.elo_groups,
        mvid_offset=cfgyml.model_args.vocab_size,
        max_seq_len=model_args.max_seq_len,
        batch_size=batch_size,
        num_workers=n_workers,
        max_testsamp=n_samp,
    )

    module_args = MMCModuleArgs(
        name=name,
        model_args=model_args,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
        max_steps=cfgyml.max_gradient_steps,
        val_check_steps=cfgyml.val_check_steps,
        accumulate_grad_batches=cfgyml.accumulate_grad_batches,
        random_seed=cfgyml.random_seed,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        outdir=outdir,
    )
    if cp is not None:
        mmc = MimicChessModule.load_from_checkpoint(cp, params=module_args)
    else:
        mmc = MimicChessModule(module_args)
    return mmc, dm
