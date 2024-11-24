import argparse
import os

n_workers = os.cpu_count() - 1
os.environ["OMP_NUM_THREADS"] = str(n_workers)
import json
from pgn.py.lib.reconstruct import count_invalid

import torch
from config import get_config
from mmc import MimicChessCoreModule, MMCModuleArgs, MimicChessHeadModule
from mmcdataset import MMCDataModule, NOOP
from model import ModelArgs


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    model_args = ModelArgs(cfgyml.model_args)

    datadirs = {"core": cfgyml.datadir}
    for elo in os.listdir(os.path.join(cfgyml.datadir, "elos")):
        datadirs[elo] = os.path.join(cfgyml.datadir, "elos", elo)

    with open(os.path.join(cfgyml.datadir, "fmd.json")) as f:
        fmd = json.load(f)

    module_args = MMCModuleArgs(
        "prediction",
        os.path.join("outputs", "dummy"),
        model_args,
        fmd["min_moves"] - 1,
        NOOP,
        cfgyml.lr_scheduler_params,
        cfgyml.max_steps,
        cfgyml.val_check_steps,
        cfgyml.random_seed,
        "auto",
        1,
    )
    mmc = MimicChessCoreModule.load_from_checkpoint(args.cp, params=module_args)
    outputs = {}
    for dataname, pth in datadirs.items():
        print(f"Predicting {dataname}")
        dm = MMCDataModule(
            pth,
            cfgyml.elo_edges,
            model_args.max_seq_len,
            cfgyml.batch_size,
            n_workers,
        )
        outputs[dataname] = mmc.predict(dm)

    def evaluate(outputs):
        npass = 0
        ngm = 0
        nvalid = 0
        ntotal = 0
        nbatch = len(outputs)
        top_n_stats = [{"n": 3, "pred": 0, "match": 0}, {"n": 5, "pred": 0, "match": 0}]
        for i, (tokens, probs, heads, opening, tgt) in enumerate(outputs):
            print(f"Evaluation {int(100*i/nbatch)}% done", end="\r")
            for s in top_n_stats:
                head_tokens = torch.index_select(tokens, 0, heads[:, 0])
                btokens = torch.index_select(tokens, 0, heads[:, 1])
                head_tokens[:, :, 1::2] = btokens[:, :, 1::2]
                matches = (head_tokens[:, : s["n"]] == tgt).sum(dim=-1, keepdim=True)
                matches[tgt == NOOP] = 0
                npred = (tgt != NOOP).sum()
                s["pred"] += npred
                s["match"] += matches.sum()

            ngm += tokens.shape[0]
            for game in zip(head_tokens, opening, tgt):
                pred, opn, tgt = game
                nmoves, nfail = count_invalid(pred[:, 0], opn, tgt[:, 0])
                nvalid += nmoves - nfail
                ntotal += nmoves
                if nfail == 0:
                    npass += 1
        print()
        print(f"Legal game frequency: {100*npass/ngm:.1f}%")
        print(f"Legal move frequency: {100*nvalid/ntotal:.1f}%")
        for s in top_n_stats:
            print(f"Top {s['n']} accuracy: {100*s['match']/s['pred']:.2f}%")

    for name, data in outputs.items():
        print(f"Evaluating {name}")
        evaluate(data)


if __name__ == "__main__":
    main()
