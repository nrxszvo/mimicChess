import argparse
import os
import json
from datetime import datetime

import torch
import yaml
from lib.training import MimicChessModule, MMCModuleArgs, MMCDataModule, ModelArgs

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", default="cfg.yml", help="yaml config file")
parser.add_argument("--datadir", default="dataset", help="directory containing parquet files")
parser.add_argument("--token_file", default="pgn_tokens.json", help="json token file")
parser.add_argument("--num_workers", default=-1, type=int, help="number of workers")
parser.add_argument(
    "--save_path",
    default="outputs",
    help="folder for saving config and checkpoints",
)
parser.add_argument(
    "--name",
    default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="experiment name for log files and checkpoints",
)
parser.add_argument(
    "--ckpt",
    default=None,
    help="MMC checkpoint from which to resume training",
)
parser.add_argument(
    "--commit",
    default=None,
    help="current commit associated with this version of codebase",
)

def get_vocab_size(encoder_params):
    max_tok = 0
    for tok in encoder_params["ranks"].values():
        max_tok = max(max_tok, tok)
    for tok in encoder_params["special_tokens"].values():
        max_tok = max(max_tok, tok)
    return max_tok + 1

def main(args):
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
    cfg["commit"] = args.commit

    save_path = os.path.join(args.save_path, args.name)
    os.makedirs(save_path, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg["random_seed"])

    mmc = MimicChessModule(
        MMCModuleArgs(
            name=args.name,
            model_args=ModelArgs(cfg["model_args"]),
            lr_scheduler_params=cfg["lr_scheduler_params"],
            max_steps=cfg["max_steps"],
            val_check_steps=cfg["val_check_steps"],
            accumulate_grad_batches=cfg["accumulate_grad_batches"],
            random_seed=cfg["random_seed"],
            strategy=cfg["strategy"],
            devices=1,
            outdir=save_path,
        )
    )
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = os.cpu_count() - 1
    with open(args.token_file) as f:
        encoder_params = json.load(f)

    cfg['model_args']['vocab_size'] = get_vocab_size(encoder_params)
    dm = MMCDataModule(
        root_dir=args.datadir,
        min_timectl=cfg["min_timectl"],
        max_rows_per_file=cfg["max_rows_per_file"],
        encoder_params=encoder_params,
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
    )

    nweights, nflpweights = mmc.num_params()
    est_tflops = (
        6
        * nflpweights
        * cfg["batch_size"]
        * cfg["model_args"]["max_seq_len"]
        / 1e12
    )
    print(f"# model params: {nweights:.2e}")
    print(f"estimated TFLOPs: {est_tflops:.1f}")

    mmc.fit(dm, ckpt=args.ckpt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
