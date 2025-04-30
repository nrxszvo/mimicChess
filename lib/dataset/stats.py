import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--npydir", required=True, help="top-level directory containing either fmd.json or block folders"
)
parser.add_argument(
    "--elo_hist",
    default=False,
    action="store_true",
    help="histogram of Elos for white and black",
)
parser.add_argument(
    "--elo_matrix", default=False, action="store_true", help="generate Elo 2d histogram"
)
parser.add_argument(
    "--time_hist",
    default=False,
    action="store_true",
    help="generate histogram of game time controls",
)
parser.add_argument('--time_hist_edges', nargs='+', type=int,
                    default=[0, 300, 600, 1800, 10800], help='time edges in seconds for time_hist')
parser.add_argument('--plot', action='store_true',
                    help='generate pyplot plots')
parser.add_argument('--title', default=None, help='plot title')

parser.add_argument('--elo_edges', default=None, nargs='+', type=int, help='elo edges')

def elo_matrix(
    welos,
    belos,
    plot=False,
    edges=[0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 4000],
    spacing=12,
    title=None,
):
    if edges[0] > 0:
        edges.insert(0, 0)
    maxelo = int(max(welos.max(), belos.max())) + 1
    if maxelo > edges[-1]:
        edges.append(maxelo)
    else:
        edges[-1] = maxelo

    H, w_edges, b_edges = np.histogram2d(welos, belos, bins=(edges, edges))

    nedge = H.shape[0]
    Hfull = H.copy()
    for i in range(nedge):
        for j in range(nedge):
            if i != j:
                Hfull[i, j] += H[j, i]
    T = np.tril(Hfull)

    if plot:
        fig, ax = plt.subplots()
        ax.tick_params(axis='both', labelsize=7)
        plot = ax.pcolormesh(np.log10(H), cmap="rainbow")

        cbar = fig.colorbar(plot)
        tick_labels = [
            f'{10**exp:.1e}' for exp in cbar.get_ticks()]
        cbar.set_ticklabels(tick_labels, fontsize=7)
        cbar.ax.tick_params(pad=5)
        cbar.ax.set_ylabel('# games', rotation=270, labelpad=20)

        elo_labels = [str(e) for e in edges]
        elo_ticks = range(len(edges))

        ax.set_xticks(elo_ticks)
        ax.set_xticklabels(elo_labels)
        ax.set_yticks(elo_ticks)
        ax.set_yticklabels(elo_labels)

        ax.set_xlabel('White Elo', labelpad=12)
        ax.set_ylabel('Black Elo', labelpad=12)
        if title:
            ax.set_title(f'Games by Elo Matchup - {title}')

        plt.tight_layout()
        plt.savefig("elo_matrix.png", dpi=500)

    for i, row in reversed(list(enumerate(T))):

        print(str(int(w_edges[i + 1])).rjust(spacing), end="")
        for j, c in enumerate(row):
            if j <= i:
                v = f'{c:.1e}'
                print(v.rjust(spacing), end="")
            else:
                print(' '.rjust(spacing), end='')
        print()
    print("".rjust(spacing), end="")
    for e in b_edges[1:]:
        print(str(int(e)).rjust(spacing), end="")
    print()


def elo_hist(
    welos,
    belos,
    plot=False,
    edges=[0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 4000],
):
    max_elo = max(welos.max(), belos.max())
    hw, es = np.histogram(welos, edges)
    hb, es = np.histogram(belos, edges)

    es[-1] = max_elo

    def print_h(hs, es):
        for h in hs:
            print(f"{np.log10(h):.2f}".rjust(8), end="")
        print()
        for e in es[1:]:
            print(str(int(e)).rjust(8), end="")
        print()

    print("white:")
    print_h(hw, es)
    print("black:")
    print_h(hb, es)

    if plot:
        x = np.arange(len(edges) - 1)
        width = 0.33
        mult = 0
        ax = plt.figure().add_subplot()
        for name, data in [("white", hw), ("black", hb)]:
            offset = width * mult
            ax.bar(x + offset, data, width, label=name)
            mult += 1
        ax.set_ylabel("# games")
        ax.set_yscale("log")
        edge_labels = [str(e) for e in edges[1:]]
        edge_labels[-1] = f">{edge_labels[-2]}"
        ax.set_xticks(x + width, edge_labels)
        ax.legend(loc="upper right")

        plt.savefig("elos.png", dpi=500)


def time_hist(blocks, edges):
    for grp in ["0", ">0"]:
        all_hs = None
        if "train" in blocks[0]:
            for cat in ["train", "test", "val"]:
                indices = blocks[0][cat]
                for i, blk in enumerate(blocks):
                    gidx = indices[:, 0][indices[:, 3] == i]
                    start_times = blk["tc"][gidx]
                    if grp == "0":
                        start_times = start_times[blk["inc"][gidx] == 0]
                    else:
                        start_times = start_times[blk["inc"][gidx] > 0]
                    hs, es = np.histogram(start_times, edges)
                    if all_hs is None:
                        all_hs = hs
                    else:
                        all_hs += hs
        else:
            for blk in blocks:
                start_times = blk["tc"]
                if grp == "0":
                    start_times = start_times[blk["inc"] == 0]
                else:
                    start_times = start_times[blk["inc"] > 0]
                hs, es = np.histogram(start_times, edges)
                if all_hs is None:
                    all_hs = hs
                else:
                    all_hs += hs

        print(f"Increment: {grp}")
        for h, e in zip(all_hs, es):
            print(f"\t{int(e)}: {h:.1e}")


def game_lengths(md, gs):
    gamelengths = np.diff(gs)
    mean = np.mean(gamelengths)
    std = np.var(gamelengths) ** 0.5
    print(
        f"gamelength stats:\n\ttotal games: {len(gamelengths)}\n\tmax length: {gamelengths.max()}\n\tmean: {mean:.2f}, std: {std:.2f}"
    )
    ax = plt.figure().add_subplot()
    ax.hist(gamelengths, bins=100)
    plt.savefig("gamelengths.png", dpi=500)


def load_block_data(blockdirs):
    blocks = []
    for dn in blockdirs:
        gs = np.memmap(f"{dn}/gamestarts.npy", mode="r", dtype="int64")
        clk = np.memmap(f"{dn}/clk.npy", mode="r", dtype="int16")
        welos = np.memmap(f"{dn}/welos.npy", mode="r", dtype="int16")
        belos = np.memmap(f"{dn}/belos.npy", mode="r", dtype="int16")
        tc = np.memmap(f"{dn}/timeCtl.npy", mode="r", dtype="int16")
        inc = np.memmap(f"{dn}/inc.npy", mode="r", dtype="int16")
        blocks.append(
            {"gs": gs, "clk": clk, "welos": welos,
                "belos": belos, "tc": tc, "inc": inc}
        )
    return blocks


def load_filtered_data(fmd, npydir):

    data = {}
    for name in ["train", "val", "test"]:
        data[name] = np.memmap(
            f"{npydir}/{name}.npy", mode="r", dtype="int64", shape=fmd[f"{name}_shape"]
        )
    if 'elo_histo' in fmd:
        data['elo_edges'] = sorted(int(e)
                                   for e in list(fmd['elo_histo'].keys()))

    blocks = []
    for dn in fmd["block_dirs"]:
        clk = np.memmap(f"{dn}/clk.npy", mode="r", dtype="int16")
        welos = np.memmap(f"{dn}/welos.npy", mode="r", dtype="int16")
        belos = np.memmap(f"{dn}/belos.npy", mode="r", dtype="int16")
        gs = np.memmap(f"{dn}/gamestarts.npy", mode="r", dtype="int64")
        tc = np.memmap(f"{dn}/timeCtl.npy", mode="r", dtype="int16")
        inc = np.memmap(f"{dn}/inc.npy", mode="r", dtype="int16")
        blocks.append(
            {"welos": welos, "belos": belos, "gs": gs,
                "clk": clk, "tc": tc, "inc": inc}
        )
    blocks[0].update(data)

    return blocks


def print_data_info(fmd):
    for split in ['train', 'test', 'val']:
        print(split)
        for cat in ['games', 'moves']:
            print(f'\t{cat}: {fmd[split + "_" + cat]:.2e}')


def get_elos(blocks, fmd):
    welos = np.array([])
    belos = np.array([])
    for i, blk in enumerate(blocks):
        print(f"collecting block {i} elos...", end="\r")
        if fmd:
            blk_welos = np.array([])
            blk_belos = np.array([])
            for name in ["train", "val", "test"]:
                data = blocks[0][name]
                indices = data[data[:, -1] == i][:, 0]
                blk_welos = np.concatenate([blk_welos, blk["welos"][indices]])
                blk_belos = np.concatenate([blk_belos, blk["belos"][indices]])
        else:
            blk_welos = blk["welos"]
            blk_belos = blk["belos"]
        welos = np.concatenate([welos, blk_welos])
        belos = np.concatenate([belos, blk_belos])
    print()
    return welos, belos


if __name__ == "__main__":
    args = parser.parse_args()
    fmdfn = "fmd.json" in os.listdir(args.npydir)
    if fmdfn:
        with open(f"{args.npydir}/fmd.json") as f:
            fmd = json.load(f)
        blocks = load_filtered_data(fmd, args.npydir)
        print_data_info(fmd)
    else:
        blockdirs = [
            os.path.abspath(f"{args.npydir}/{dn}")
            for dn in os.listdir(args.npydir)
            if "block-" in dn
        ]
        blocks = load_block_data(blockdirs)

    if args.elo_matrix or args.elo_hist:
        welos, belos = get_elos(blocks, fmdfn)
        kwargs = {'plot': args.plot, 'title': args.title}
        if args.elo_edges:
            kwargs['edges'] = args.elo_edges
        elif 'elo_edges' in blocks[0]:
            kwargs['edges'] = blocks[0]['elo_edges']
        if args.elo_hist:
            elo_hist(welos, belos)
        if args.elo_matrix:
            elo_matrix(welos, belos, **kwargs)

    if args.time_hist:
        time_hist(blocks, args.time_hist_edges)
