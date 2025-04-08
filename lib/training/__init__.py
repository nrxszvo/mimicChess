import os
import json

import torch

from .mmc import MimicChessModule, MMCModuleArgs
from .mmcdataset import MMCDataModule
from .model import ModelArgs


def get_model_args(cfgyml):
    model_args = ModelArgs(cfgyml.model_args.__dict__)
    if cfgyml.elo_params.predict:
        model_args.gaussian_elo = cfgyml.elo_params.loss == "gaussian_nll"
        if cfgyml.elo_params.loss == "cross_entropy":
            model_args.elo_pred_size = len(cfgyml.elo_params.edges) + 1
        elif cfgyml.elo_params.loss == "gaussian_nll":
            model_args.elo_pred_size = 2
        elif cfgyml.elo_params.loss == "mse":
            model_args.elo_pred_size = 1
        else:
            raise Exception("did not recognize loss function name")
    model_args.n_timecontrol_heads = len([
        n for _, grp in cfgyml.tc_groups.items() for n in grp
    ])
    model_args.n_elo_heads = len(cfgyml.elo_params.edges)
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
    constant_var=False,
):
    if n_workers is None:
        n_workers = os.cpu_count() // devices if torch.cuda.is_available() else 0
    datadir = cfgyml.datadir if alt_datadir is None else alt_datadir
    model_args = get_model_args(cfgyml)

    whiten_params = None
    if cfgyml.elo_params.loss in ["gaussian_nll", "mse"]:
        with open(f"{cfgyml.datadir}/fmd.json") as f:
            fmd = json.load(f)
        whiten_params = (fmd["elo_mean"], fmd["elo_std"])
    cfgyml.elo_params.whiten_params = whiten_params

    assert cfgyml.effective_batch_size % cfgyml.accumulate_grad_batches == 0
    batch_size = int(cfgyml.effective_batch_size /
                     cfgyml.accumulate_grad_batches)

    dm = MMCDataModule(
        datadir=datadir,
        elo_edges=cfgyml.elo_params.edges,
        tc_groups=cfgyml.tc_groups,
        max_seq_len=model_args.max_seq_len,
        batch_size=batch_size,
        num_workers=n_workers,
        whiten_params=whiten_params,
        max_testsamp=n_samp,
        opening_moves=cfgyml.opening_moves,
    )
    cfgyml.elo_params.constant_var = constant_var

    max_steps = int(cfgyml.effective_max_steps*cfgyml.accumulate_grad_batches)
    val_check_steps = int(cfgyml.effective_val_check_steps *
                          cfgyml.accumulate_grad_batches)

    module_args = MMCModuleArgs(
        name=name,
        elo_params=cfgyml.elo_params,
        model_args=model_args,
        opening_moves=dm.opening_moves,
        lr_scheduler_params=cfgyml.lr_scheduler_params,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        accumulate_grad_batches=cfgyml.accumulate_grad_batches,
        random_seed=cfgyml.random_seed,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        outdir=outdir,
    )
    if hasattr(cfgyml, 'pretrain_cp'):
        module_args.pretrain_cp = cfgyml.pretrain_cp
    if cp is not None:
        mmc = MimicChessModule.load_from_checkpoint(cp, params=module_args)
    else:
        mmc = MimicChessModule(module_args)
    return mmc, dm
