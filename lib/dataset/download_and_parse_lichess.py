import argparse
import os
import re
import time
import subprocess
from multiprocessing import Lock, Process, Queue
import threading
from pzp import ParserPool

devnull = open(os.devnull, "w")


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, *args, **kwargs):
        self.lock.acquire()
        try:
            print(*args, **kwargs)
        finally:
            self.lock.release()


def timeit(fn):
    start = time.time()
    ret = fn()
    end = time.time()
    nsec = end - start
    hr = int(nsec // 3600)
    minute = int((nsec % 3600) // 60)
    sec = int(nsec % 60)
    return ret, f"{hr}:{minute:02}:{sec:02}"


def collect_existing(out_dir):
    existing = []
    procfn = os.path.join(out_dir, "processed.txt")
    if os.path.exists(procfn):
        with open(procfn) as f:
            for line in f:
                name = line.rstrip()
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
                to_proc.append((line.rstrip(), name))
    return to_proc


def parse_url(url):
    m = re.match(r".*(lichess_db.*pgn\.zst)", url)
    zst = m.group(1)
    pgn_fn = zst[:-4]
    return zst, pgn_fn


class Monitor:
    def __init__(self, dl_dir, max_active_procs, zst_list):
        self.dl_dir = dl_dir
        os.makedirs(self.dl_dir, exist_ok=True)
        self.max_active_procs = max_active_procs
        self.zst_list = zst_list

    def get_existing_zsts(self):
        zsts = list(filter(lambda fn: fn in self.zst_list, os.listdir(self.dl_dir)))
        return zsts

    def should_sleep(self):
        zsts = self.get_existing_zsts()
        return len(zsts) >= 2 * self.max_active_procs


def download_proc(pid, dl_start, url_q, zst_q, print_safe, monitor):
    line = dl_start + pid
    while True:
        url, name = url_q.get()
        if url == "DONE":
            zst_q.put(("DONE", None))
            break
        zst, _ = parse_url(url)
        if not os.path.exists(os.path.join(monitor.dl_dir, zst)):
            while monitor.should_sleep():
                print_safe(f"\033[{line}H\033[Kdl proc: sleeping...", end="\r")
                time.sleep(5)
            print_safe(f"\033[{line}H\033[Kdl proc: downloading {name}", end="\r")
            _, time_str = timeit(
                lambda: subprocess.call(
                    ["wget", "-nv", url],
                    cwd=monitor.dl_dir,
                    stdout=devnull,
                    stderr=devnull,
                )
            )
            print_safe(
                f"\033[{line}H\033[Kdl proc: finished downloading {name} in {time_str}",
                end="\r",
            )
        zst_q.put((name, zst))


def start_download_procs(dl_start, url_q, zst_q, print_safe, monitor, nproc):
    procs = []
    for pid in range(nproc):
        p = Process(
            target=download_proc,
            args=(pid, dl_start, url_q, zst_q, print_safe, monitor),
        )
        p.daemon = True
        p.start()
        procs.append(p)
    return procs


def remove_completed(dl_dir, parser_pool, total, sleep=5):
    while True:
        time.sleep(sleep)
        completed = parser_pool.get_completed()
        for name in completed:
            zst = zst_from_name(name.decode("utf-8"))
            if os.path.exists(os.path.join(dl_dir, zst)):
                os.remove(os.path.join(dl_dir, zst))
        if len(completed) == total:
            break


def main(
    list_fn,
    out_dir,
    n_dl_proc,
    max_active_procs,
    n_reader_proc,
    n_move_proc,
    minSec,
    elo_edges,
):
    to_proc = collect_remaining(list_fn, out_dir)
    if len(to_proc) == 0:
        print("All files already processed")
        return

    url_q = Queue()
    zst_q = Queue()

    print_safe = PrintSafe()
    zst_list = [fn.split("/")[-1] for fn, _ in to_proc]

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
        printOffset=2 + n_dl_proc,
        outdir=out_dir,
    )

    monitor = Monitor(os.path.join(out_dir, "tmp_zst"), max_active_procs, zst_list)

    rc_thread = threading.Thread(
        target=remove_completed,
        args=(monitor.dl_dir, pool, len(to_proc)),
    )
    rc_thread.start()

    dl_start = 1

    print("\033[2J", end="")

    dl_ps = start_download_procs(dl_start, url_q, zst_q, print_safe, monitor, n_dl_proc)

    existing_zsts = monitor.get_existing_zsts()

    for url, name in to_proc:
        zst, _ = parse_url(url)
        if zst in existing_zsts:
            zst_q.put((name, zst))
        else:
            url_q.put((url, name))

    for _ in range(n_dl_proc):
        url_q.put(("DONE", None))

    n_dl_done = 0

    try:
        while True:
            name, zst_fn = zst_q.get()
            if name == "DONE":
                n_dl_done += 1
                if n_dl_done == n_dl_proc:
                    break
            else:
                pool.enqueue(os.path.join(monitor.dl_dir, zst_fn), name)

    finally:
        print_safe("cleaning up...")
        pool.join()
        url_q.close()
        zst_q.close()
        for dl_p in dl_ps:
            try:
                dl_p.join(0.25)
            except Exception as e:
                print(e)
                dl_p.kill()
        rc_thread.join()
        for fn in os.listdir(monitor.dl_dir):
            if re.match(r"lichess_db_standard_rated.*\.zst.*\.tmp", fn):
                os.remove(fn)


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
        "--n_dl_procs",
        default=2,
        type=int,
        help="number of zsts to download in parallel",
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
        default="1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000",
        help="Elo rating buckets",
    )

    args = parser.parse_args()

    main(
        list_fn=args.list,
        out_dir=args.outdir,
        n_dl_proc=args.n_dl_procs,
        max_active_procs=args.n_active_procs,
        n_reader_proc=args.n_reader_procs,
        n_move_proc=args.n_move_procs,
        minSec=args.min_seconds,
        elo_edges=args.elo_edges,
    )
