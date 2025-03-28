import argparse
import os

import torch
import numpy as np

from lib import (
    get_config,
    AccuracyStats,
    TargetStats,
    LegalGameStats,
    MoveStats,
    init_modules,
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cfg", required=True, help="yaml config file")
parser.add_argument("--cp", required=True, help="checkpoint file")
parser.add_argument(
    "--datadir",
    default=None,
    help="alternative data directory to use instead of value in cfg file",
)
parser.add_argument("--bs", default=None, type=int, help="batch size")
parser.add_argument(
    "--report_fn", default=None, help="text file to write summary results"
)
parser.add_argument(
    "--nsamp",
    default=None,
    help="maximum number of samples (games) to process",
    type=int,
)
parser.add_argument(
    "--constant_var",
    action="store_true",
    default=False,
    help="use constant for Elo variance param",
)


@torch.inference_mode()
def evaluate(outputs, seq_len, elo_edges):
    nbatch = len(outputs)
    acc_stats = AccuracyStats(seq_len, elo_edges, 2 / len(elo_edges))
    tstats = TargetStats()
    game_stats = LegalGameStats()
    move_stats = MoveStats()

    elo_loss = np.array([elo_op["loss"] for _, elo_op in outputs]).mean()
    move_loss = np.array([mv_op["loss"] for mv_op, _ in outputs]).mean()
    loc_err = np.array([
        elo_op["location_error"] if "location_error" in elo_op else 0
        for _, elo_op in outputs
    ]).mean()
    avg_std = np.array([
        elo_op["average_std"] if "average_std" in elo_op else 0
        for _, elo_op in outputs
    ]).mean()
    for i, (move_data, elo_data) in enumerate(outputs):
        print(f"Evaluation {int(100 * i / nbatch)}% done", end="\r")
        game_stats.eval(
            move_data["sorted_tokens"], move_data["openings"], move_data["targets"]
        )
        move_stats.eval(move_data["sorted_tokens"], move_data["targets"])

    report = (
        game_stats.report()
        + move_stats.report()
        + [f"Location error: {loc_err:.1f}", f"Average std: {avg_std:.1f}"]
        + [f"Elo loss: {elo_loss:.2f}", f"Move loss: {move_loss:.2f}"]
    )
    return report


def predict(cfgyml, datadir, cp, n_samp, constant_var):
    name = (os.path.splitext(os.path.basename(cp))[0],)
    cfgyml.elo_params.constant_var = constant_var
    mmc, dm = init_modules(
        cfgyml,
        name,
        "auto",
        1,
        alt_datadir=datadir,
        n_samp=n_samp,
    )
    mmc.load_state_dict(torch.load(cp))
    predictions = mmc.predict(dm)
    seqlen = dm.max_seq_len - dm.opening_moves + 1
    return predictions, seqlen


def main():
    args = parser.parse_args()
    cfgfn = args.cfg
    cfgyml = get_config(cfgfn)

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfgyml.random_seed)

    if args.bs:
        cfgyml.batch_size = args.bs

    datadir = cfgyml.datadir if args.datadir is None else args.datadir
    outputs, seq_len = predict(
        cfgyml, datadir, args.cp, args.nsamp, args.constant_var)
    report = evaluate(outputs, seq_len, cfgyml.elo_params.edges)
    print()
    for line in report:
        print(line)
    if args.report_fn is not None:
        with open(args.report_fn, "w") as f:
            f.write("\n".join(report))


if __name__ == "__main__":
    main()
