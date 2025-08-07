import subprocess
import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--list", default='list.txt')
parser.add_argument("--out_dir", default='zst')

def parse_url(url):
    m = re.match(r".*(lichess_db.*pgn\.zst)", url)
    zst = m.group(1)
    return zst, name_from_zst(zst)


def name_from_zst(zst):
    return re.match(r".+standard_rated_([0-9\-]+)\.pgn\.zst", zst).group(1)


if __name__ == "__main__":
    args = parser.parse_args()

    tld = Path(args.out_dir)
    tmpdir = tld / "tmp_zst"
    zstdir = tld / "zst"
    
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(zstdir, exist_ok=True)
    
    with open(args.list) as f:
        lines = f.readlines()

    urls = [line.rstrip() for line in lines]

    for i, url in enumerate(urls):
        zst, name = parse_url(url)
        print(f'downloading {name} ({i+1} of {len(urls)})')
        subprocess.call(f'wget {url} -P {tmpdir}', shell=True)
        src = tmpdir / zst
        dest = zstdir / zst
        if dest.exists():
            os.remove(dest)
        os.rename(src, dest)
        