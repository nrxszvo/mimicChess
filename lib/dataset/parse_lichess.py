from pzp import ParserPool
import argparse
import os
from pathlib import Path
import time
import re
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--list", default='list.txt')

def name_from_zst(zst):
    return re.match(r".+standard_rated_([0-9\-]+)\.pgn\.zst", zst).group(1)

def print_loop(pool, nfiles):
    print("\033[2J", end="")
    while len(pool.get_completed()) < nfiles:
        info = pool.get_info()
        clear = [f"\033[{i}H\033[K" for i in range(len(info))]
        print("".join(clear), end="")
        print("\033[0H", end="")
        print("\n".join(info))
        time.sleep(1)

    info = pool.get_info()
    clear = [f"\033[{i}H\033[K" for i in range(len(info))]
    print("".join(clear), end="")
    print("\033[0H", end="")
    print("\n".join(info))


def main(args):
    pool = ParserPool(
        nSimultaneous=2,
        nReadersPerFile=2,
        nParsersPerFile=30,
        minSec=60,
        maxSec=10800,
        maxInc=600,
        elo_edges=[1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
        chunkSize=1024 * 1024,
        printFreq=1,
        outdir=args.out_dir,
    )
    
    zsts = {}
    with open(args.list) as f:
        for line in f:
            zst = line.rstrip()
            zsts[name_from_zst(zst)] = zst

    print_thread = threading.Thread(
        target=print_loop, args=(pool, len(zsts))
    )
    print_thread.start()

    tld = Path(args.out_dir)
    zstdir = tld / "zst"

    os.makedirs(zstdir, exist_ok=True)

    processed = set() 
    if (tld / "processed.txt").exists():
        with open(tld / "processed.txt") as f:
            for line in f:
                name = line.split(",")[0]
                processed.add(name)

    processing = set()
    while len(processed) < len(zsts):
        downloaded = [zstdir / fn for fn in os.listdir(zstdir) if fn.endswith(".zst")]
        for zst in sorted(downloaded, reverse=True):
            name = name_from_zst(zst.name)
            if name not in processed and name not in processing:
                pool.enqueue(str(zst), name)
                processing.add(name)

        with open(tld / "processed.txt", "a") as f:
            for name, ngames in pool.get_completed():
                if name not in processed:
                    f.write(f"{name},{ngames}\n")
                    processed.add(name)
                    processing.remove(name)
        time.sleep(60)
    
    print_thread.join()
    pool.join()
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
 