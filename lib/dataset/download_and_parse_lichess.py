import argparse
import os
import re
import subprocess
import time
from multiprocessing import Lock, Process, Queue


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, string, end="\n"):
        self.lock.acquire()
        try:
            print(string, end=end)
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


def collect_existing_npy(out_dir):
    existing = []
    procfn = os.path.join(out_dir, "processed.txt")
    if os.path.exists(procfn):
        with open(procfn) as f:
            for line in f:
                _, name, _, _, _, status = line.rstrip().split(",")
                if status != "failed":
                    existing.append(name)
    return existing


def collect_remaining(list_fn, out_dir):
    existing = collect_existing_npy(out_dir)
    to_proc = []
    with open(list_fn) as f:
        for line in f:
            print(line)
            name = re.match(r".+standard_rated_([0-9\-]+)\.pgn\.zst", line).group(1)
            if name not in existing:
                to_proc.append((line.rstrip(), name))
    return to_proc


def parse_url(url):
    m = re.match(r".*(lichess_db.*pgn\.zst)", url)
    zst = m.group(1)
    pgn_fn = zst[:-4]
    return zst, pgn_fn


def download_proc(pid, url_q, zst_q, print_safe):
    added = 0
    completed = 0
    while True:
        url, name = url_q.get()
        if url == "DONE":
            zst_q.put(("DONE", None))
            break
        zst, _ = parse_url(url)
        if not os.path.exists(zst):
            if not os.path.exists(zst):
                while added - completed > 2:
                    print_safe(f"download proc {pid} is sleeping...")
                    time.sleep(5 * 60)
                print_safe(f"{name}: downloading...")
                _, time_str = timeit(lambda: subprocess.call(["wget", url]))
                print_safe(f"{name}: finished downloading in {time_str}")
                completed += 1
        added += 1
        zst_q.put((name, zst))


def start_download_procs(url_q, zst_q, print_safe, nproc):
    procs = []
    for pid in range(nproc):
        p = Process(target=download_proc, args=((pid, url_q, zst_q, print_safe)))
        p.daemon = True
        p.start()
        procs.append(p)
    return procs


def main(
    list_fn,
    out_dir,
    processor_bin,
    n_dl_proc,
    max_active_procs,
    n_reader_proc,
    n_move_proc,
    minSec,
):
    to_proc = collect_remaining(list_fn, out_dir)
    if len(to_proc) == 0:
        print("All files already processed")
        return

    url_q = Queue()
    zst_q = Queue()

    print_safe = PrintSafe()
    dl_ps = start_download_procs(url_q, zst_q, print_safe, n_dl_proc)
    for url, name in to_proc:
        url_q.put((url, name))
    for _ in range(n_dl_proc):
        url_q.put(("DONE", None))

    n_dl_done = 0
    try:
        active_procs = []
        terminate = False
        while True:
            name, zst_fn = zst_q.get()
            if name == "DONE":
                n_dl_done += 1
                if n_dl_done == n_dl_proc:
                    terminate = True
            else:
                print_safe(f"{name}: processing zst into {out_dir}...")
                cmd = [
                    processor_bin,
                    "--zst",
                    zst_fn,
                    "--name",
                    name,
                    "--outdir",
                    os.path.join(out_dir, name),
                    "--nReaders",
                    str(n_reader_proc),
                    "--nMoveProcessors",
                    str(n_move_proc),
                    "--minSec",
                    str(minSec),
                ]
                p = subprocess.Popen(cmd)
                active_procs.append((p, name, zst_fn))

            def check_cleanup(p, name, zst):
                finished = False
                status = p.poll()
                if status is not None:
                    if status != 0:
                        print_safe(f"{name}: poll returned {status}")
                        _, errs = p.communicate()
                        if errs is not None:
                            print_safe(f"{name}: returned errors:\n{errs}")
                        return True, status
                    os.remove(zst)
                    finished = True

                return finished, status

            while len(active_procs) == max_active_procs or (
                terminate and len(active_procs) > 0
            ):
                time.sleep(5)
                for procdata in reversed(active_procs):
                    finished, status = check_cleanup(*procdata)
                    if finished:
                        if status > 0:
                            terminate = True
                            print_safe(
                                f"Last archive failed with status {status}, 'terminate' signaled"
                            )
                        active_procs.remove(procdata)
                        break

            if terminate and len(active_procs) == 0:
                break

    finally:
        print_safe("cleaning up...")
        for p, _, zst in active_procs:
            p.kill()
            # os.remove(zst)
        url_q.close()
        zst_q.close()
        for dl_p in dl_ps:
            try:
                dl_p.join(0.25)
            except Exception as e:
                print(e)
                dl_p.kill()
        for fn in os.listdir("."):
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
        "--processor",
        default="cpp/build/processZst/processZst",
        help="zst/pgn processor binary",
    )
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

    args = parser.parse_args()

    main(
        args.list,
        args.outdir,
        args.processor,
        args.n_dl_procs,
        args.n_active_procs,
        args.n_reader_procs,
        args.n_move_procs,
        args.min_seconds,
    )
