import argparse
from pathlib import Path
import os
import re
import time
import subprocess
import threading
from pzp import ParserPool

devnull = open(os.devnull, "w")


def get_ellapsed(start):
    end = time.time()
    nsec = end - start
    hr = int(nsec // 3600)
    minute = int((nsec % 3600) // 60)
    sec = int(nsec % 60)
    return f"{hr}:{minute:02}:{sec:02}"


def collect_existing(out_dir):
    existing = []
    procfn = os.path.join(out_dir, "processed.txt")
    if os.path.exists(procfn):
        with open(procfn) as f:
            for line in f:
                name = line.split(",")[0]
                existing.append(name)
    return existing


def name_from_zst(zst):
    return re.match(r".+standard_rated_([0-9\-]+)\.pgn\.zst", zst).group(1)


def zst_from_name(name):
    return f"lichess_db_standard_rated_{name}.pgn.zst"


def collect_remaining(list_fn, out_dir):
    existing = collect_existing(out_dir)
    to_proc = []
    with open(list_fn) as f:
        for line in f:
            name = name_from_zst(line)
            if name not in existing:
                to_proc.append(line.rstrip())
    return to_proc


def parse_url(url):
    m = re.match(r".*(lichess_db.*pgn\.zst)", url)
    zst = m.group(1)
    return zst, name_from_zst(zst)


class Manager:
    def __init__(self, dl_dir, max_cached, urls, parser_pool):
        self.dl_dir = Path(dl_dir)
        self.lock = threading.Lock()
        os.makedirs(self.dl_dir, exist_ok=True)
        self.dl_log = self.dl_dir.joinpath("downloaded.txt")
        self.proc_log = self.dl_dir.joinpath("processed.txt")
        self.max_cached = max_cached
        self.parser_pool = parser_pool

        self.processed = set()
        if self.proc_log.exists():
            with open(self.proc_log) as f:
                for line in f:
                    name = line.split(",")[0]
                    self.processed.add(name)

        self.to_dl = []
        self.zst_list = [parse_url(url)[0] for url in urls]
        existing = self.get_existing_zsts()
        for url in sorted(urls):
            zst, name = parse_url(url)
            if name in self.processed:
                continue
            if zst in existing:
                self.parser_pool.enqueue(str(self.dl_dir / zst), name)
            else:
                self.to_dl.append(url)

    def get_existing_zsts(self):
        with self.lock:
            existing = []
            if self.dl_log.exists():
                with open(self.dl_log) as f:
                    for line in f:
                        existing.append(line.rstrip())
            zsts = list(
                filter(
                    lambda fn: fn in self.zst_list and fn in existing,
                    os.listdir(self.dl_dir),
                )
            )
            return zsts

    def cache_full(self):
        zsts = self.get_existing_zsts()
        return len(zsts) >= self.max_cached

    def update_downloaded(self, zst):
        with self.lock:
            with open(self.dl_log, "a") as f:
                f.write(f"{zst}\n")

    def update_processed(self, name, ngames):
        if name in self.processed:
            return

        with self.lock:
            self.processed.add(name)

            with open(self.proc_log, "a") as f:
                f.write(f"{name},{ngames}\n")

            zst = zst_from_name(name)
            if (self.dl_dir / zst).exists():
                os.remove(self.dl_dir / zst)

            with open(self.dl_log, "r") as f:
                lines = f.readlines()
            with open(self.dl_log, "w") as f:
                for line in lines:
                    if line.rstrip() != zst:
                        f.write(line)

    def download_loop(self, info):
        active = []
        npid = 1
        curpid = 0
        while not stopping and (self.to_dl or active):
            if not self.cache_full() and len(active) < self.max_cached//2 and self.to_dl:
                url = self.to_dl.pop()
                zst, name = parse_url(url)
                p = subprocess.Popen(
                    ["wget", "-nv", url],
                    cwd=self.dl_dir,
                    stdout=devnull,
                    stderr=devnull,
                )
                npid = max(npid, len(active) + 1)
                curpid %= npid
                start = time.time()
                active.append((p, zst, name, start, curpid))
                if len(info) <= curpid:
                    info.append((False, "", 0))
                info[curpid] = (False, self.dl_dir / zst, start)
                curpid += 1

            for p, zst, name, start, pid in active:
                ret = p.poll()
                if ret is not None:
                    active.remove((p, zst, name, start, pid))
                    if ret == 0:
                        info[pid] = (True, name, get_ellapsed(start))
                        self.update_downloaded(zst)
                        self.parser_pool.enqueue(str(self.dl_dir / zst), name)
                    else:
                        print("wget failed for", name)
                    curpid = pid

            time.sleep(1)

    def remove_completed(self, sleep=5):
        while not stopping:
            completed = self.parser_pool.get_completed()
            for name, ngames in completed:
                self.update_processed(name, ngames)
            time.sleep(sleep)

        completed = self.parser_pool.get_completed()
        for name, ngames in completed:
            self.update_processed(name, ngames)


def get_dl_status(dl_info):
    for finished, name, ts in dl_info:
        if finished:
            yield f"finished downloading {name} in {ts}"
        else:
            if name.exists():
                size = os.path.getsize(name)
                MBps = size / 1024 / 1024 / (time.time() - ts)
                yield f"downloading {name_from_zst(name.name)} ({get_ellapsed(ts)}, {MBps:.2f} MB/s)"
            else:
                yield f"downloading {name_from_zst(name.name)} ({get_ellapsed(ts)})"


def print_loop(pool, nfiles, dl_info):
    print("\033[2J", end="")
    while len(pool.get_completed()) < nfiles and not stopping:
        info = list(get_dl_status(dl_info)) + [""] + pool.get_info()
        clear = [f"\033[{i}H\033[K" for i in range(len(info))]
        print("".join(clear), end="")
        print("\033[0H", end="")
        print("\n".join(info))
        time.sleep(1)

    info = list(get_dl_status(dl_info)) + [""] + pool.get_info()
    clear = [f"\033[{i}H\033[K" for i in range(len(info))]
    print("".join(clear), end="")
    print("\033[0H", end="")
    print("\n".join(info))


stopping = False


def main(
    list_fn,
    out_dir,
    max_dl_cache,
    max_active_procs,
    n_reader_proc,
    n_move_proc,
    minSec,
    elo_edges,
):
    to_proc = collect_remaining(list_fn, Path(out_dir) / "tmp_zst")
    if len(to_proc) == 0:
        print("All files already processed")
        return

    pool = ParserPool(
        nSimultaneous=max_active_procs,
        nReadersPerFile=n_reader_proc,
        nParsersPerFile=n_move_proc,
        minSec=minSec,
        maxSec=10800,
        maxInc=600,
        elo_edges=elo_edges,
        chunkSize=1024 * 1024,
        printFreq=1,
        outdir=out_dir,
    )
    monitor = Manager(Path(out_dir) / "tmp_zst", max_dl_cache, to_proc, pool)

    rc_thread = threading.Thread(target=monitor.remove_completed)
    rc_thread.start()

    dl_info = []
    print_thread = threading.Thread(
        target=print_loop, args=(pool, len(to_proc), dl_info)
    )
    print_thread.start()

    dl_thread = threading.Thread(target=monitor.download_loop, args=(dl_info,))
    dl_thread.start()

    global stopping
    try:
        while not stopping:
            time.sleep(1)
            completed = pool.get_completed()
            if len(completed) == len(to_proc):
                stopping = True
    except Exception as e:
        stopping = True
        print(e)
    finally:
        rc_thread.join()
        dl_thread.join()
        print_thread.join()
        pool.join()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--list",
        default="list.txt",
        help="txt file containing list of pgn zips to download and parse",
    )
    parser.add_argument("--outdir", default=".", help="folder to save parquet files")
    parser.add_argument(
        "--max_dl_cache",
        default=2,
        type=int,
        help="max number of zsts to store on disk",
    )
    parser.add_argument(
        "--n_active_procs",
        default=2,
        type=int,
        help="number of zsts to process in parallel",
    )
    parser.add_argument(
        "--n_reader_procs",
        default=2,
        help="number of decompressor/game parser threads",
        type=int,
    )
    parser.add_argument(
        "--n_move_procs",
        default=os.cpu_count() - 2,
        help="number of move parser threads",
        type=int,
    )
    parser.add_argument(
        "--min_seconds",
        default=60,
        help="minimum time control for games in seconds",
        type=int,
    )
    parser.add_argument(
        "--elo_edges",
        default=[
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
            3000,
        ],
        type=int,
        nargs="+",
        help="Elo rating buckets",
    )

    args = parser.parse_args()

    main(
        list_fn=args.list,
        out_dir=args.outdir,
        max_dl_cache=args.max_dl_cache,
        max_active_procs=args.n_active_procs,
        n_reader_proc=args.n_reader_procs,
        n_move_proc=args.n_move_procs,
        minSec=args.min_seconds,
        elo_edges=args.elo_edges,
    )
