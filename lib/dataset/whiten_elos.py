import numpy as np
import argparse
import json
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("fmd", help="fmd.json file with paths to block directories")


def get_whiten_params(fmd):
    sumv = 0
    n = 0
    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            sumv += elos.sum()
            n += elos.shape[0]
    mean = sumv / n

    sumv = 0
    for dn in fmd["block_dirs"]:
        for fn in ["welos.npy", "belos.npy"]:
            elos = np.memmap(
                os.path.join(dn, fn),
                mode="r",
                dtype="int16",
            )
            sumv += ((elos - mean) ** 2).sum()

    std = (sumv / (n - 1)) ** 0.5

    return mean, std


def main():
    args = parser.parse_args()
    with open(os.path.join(args.fmd)) as f:
        fmd = json.load(f)

    mean, std = get_whiten_params(fmd)

    print(f"mean: {mean}")
    print(f"std: {std}")

    fmd["elo_mean"] = mean
    fmd["elo_std"] = std
    with open(os.path.join(args.fmd), "w") as f:
        json.dump(fmd, f)


if __name__ == "__main__":
    main()
