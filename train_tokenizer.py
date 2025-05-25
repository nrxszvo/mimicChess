import io
import itertools
import os
import random
from multiprocessing import Process, Queue

import json
import regex
import chess.pgn
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path


def chomp_fen(fen):
    return " ".join(fen.split()[:-2])


def pgns_to_all_fens(pgns, max_moves=20):
    games = []
    for pgn in pgns:
        fens = []
        pgn = pgn.as_py()
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        for i, move in enumerate(game.mainline_moves()):
            if i == max_moves:
                break
            fens.append(chomp_fen(board.fen()))
            board.push(move)
        games.append(fens)

    return games


def pgns_to_fens(pgns, fens_per_game=10):
    fens = []
    for pgn in pgns:
        pgn = pgn.as_py()
        nmoves = len(pgn.split())
        skew = 4
        uni = 1 / (nmoves + 1)
        minv = 1 / (skew * (nmoves + 1))
        crem = (uni - minv) * (2 / nmoves)
        cumsum = 0

        game = chess.pgn.read_game(io.StringIO(pgn))
        board = game.board()
        r = random.random()
        n_fen = 0
        for i, move in enumerate(game.mainline_moves()):
            cumsum += minv + i * crem
            if r < cumsum:
                fens.append(chomp_fen(board.fen()))
                n_fen += 1
                if n_fen == fens_per_game:
                    break
            board.push(move)

    return fens


def pgns_to_fens_proc(inq, outq, fens_per_game=10):
    while True:
        pgns = inq.get()
        outq.put(pgns_to_all_fens(pgns))
        if len(pgns) == 0:
            break


def add_pgns(pqfile, inq, batch_size, nwriteprocs):
    for batch in pq.ParquetFile(pqfile).iter_batches(
        batch_size=batch_size, columns=["moves"]
    ):
        inq.put(batch["moves"])

    for _ in range(nwriteprocs):
        inq.put([])


def create_fens_serial(pqfile, batch_size, fen_fn="fens.raw"):
    total = pq.ParquetFile(pqfile).metadata.num_rows
    nfen = 0
    with open(fen_fn, "w") as ff:
        for batch in pq.ParquetFile(pqfile).iter_batches(
            batch_size=batch_size, columns=["moves"]
        ):
            fens = pgns_to_fens(batch["moves"])
            nfen += len(fens)
            ff.write("\n".join(fens) + "\n")
            print(f"{100*nfen/total:.2f}% done", end="\r")


def create_fens(pqfile, nwriteprocs, batch_size, fen_fn="fens.raw", fens_per_game=10):
    inq = Queue()
    outq = Queue()

    add_p = Process(target=add_pgns, args=(pqfile, inq, batch_size, nwriteprocs))
    add_p.daemon = True
    add_p.start()

    writeprocs = []
    for i in range(nwriteprocs):
        p = Process(target=pgns_to_fens_proc, args=(inq, outq))
        p.daemon = True
        p.start()
        writeprocs.append(p)

    total = fens_per_game * pq.ParquetFile(pqfile).metadata.num_rows
    ngames = 0
    ndone = 0
    with open(fen_fn, "w") as ff:
        while True:
            games = outq.get()
            if len(games) == 0:
                ndone += 1
                if ndone == nwriteprocs:
                    if ngames < total:
                        print(f"Warning: only wrote {ngames} games out of {total}")
                    break
            else:
                ngames += len(games)
                for game in games:
                    ff.write("\n".join(game) + "\n")
                print(f"{100*ngames/total:.2f}% done", end="\r")

    add_p.join()
    for p in writeprocs:
        p.join()

    return fen_fn


class PgnIterator:
    def __init__(self, pqfile, batch_size):
        self.iter = pq.ParquetFile(pqfile).iter_batches(
            batch_size=batch_size, columns=["moves", "clk"]
        )
        self.pat_str = "O-O-O|O-O|[RNBQK]|[a-h][0-9]|[a-h]|[RNBQKa-h]|[x=+#]|\s"

    def __iter__(self):
        for batch in self.iter:
            for pgn, clk in zip(batch["moves"], batch["clk"]):
                grps = regex.findall(self.pat_str, pgn.as_py())
                idx = 0
                recs = []
                for mv in pgn.as_py().split():
                    rec = []
                    while idx < len(grps):
                        if grps[idx] == " ":
                            idx += 1
                            break
                        rec.append(grps[idx])
                        idx += 1
                    # print(mv.ljust(6), rec)
                    recs.append(rec)
                vals = []
                for val in clk.as_py().split():
                    e = len(val) - 1
                    end = e
                    while val[end] == "0":
                        end -= 1
                    for i in range(end + 1):
                        v = val[i]
                        vals.append(str(int(v) * 10 ** (e - i)))
                    vals.append(" ")
                pidx, gidx = 0, 0
                game = []
                while pidx < len(recs):
                    while gidx < len(vals):
                        game.append(vals[gidx])
                        gidx += 1
                        if vals[gidx] == " ":
                            gidx += 1
                            break
                    game.extend(recs[pidx])
                    pidx += 1

                yield game


class FenIterator:
    PAT_STR = r"\s[a-h][1-8]|\s[bw\-]|\s[KQkq]+|[/pPrRnNbBqQkK1-8]+"

    def __init__(self, fenfile):
        self.fenfile = fenfile
        self.pat_str_old = r"\s[a-h][1-8]|\s[bw]|\s[qkQK]+|p+|r|n|b|q|k|P+|R|N|B|Q|K|[1-8]|\s[0-9]+|\s-"
        self.pat_str = (
            r"\s[a-h][1-8]|\s[bw\-]|\s[KQkq]+|p+|r|n|b|q|k|P+|R|N|B|Q|K|[1-8]"
        )
        self.pat_str_new = FenIterator.PAT_STR

    def __iter__(self):
        with open(self.fenfile) as f:
            for fen in f:
                # grps = regex.findall(self.pat_str_new, fen)
                # yield grps[0]
                fen = fen.replace("/", "")
                yield fen.split()[0]

class MPFenIterator:
    def __init__(self, pqfile, batch_size, nwriteprocs):
        self.inq = Queue()
        self.outq = Queue()
        self.total = pq.ParquetFile(pqfile).metadata.num_rows
        self.add_p = Process(target=add_pgns, args=(pqfile, self.inq, batch_size, nwriteprocs))
        self.add_p.daemon = True
        self.add_p.start()

        self.writeprocs = []
        for i in range(nwriteprocs):
            p = Process(target=pgns_to_fens_proc, args=(self.inq, self.outq))
            p.daemon = True
            p.start()
            self.writeprocs.append(p)
        self.ngames = 0
        self.ndone = 0

    def __iter__(self):
        while True:
            games = self.outq.get()
            if len(games) == 0:
                self.ndone += 1
                if self.ndone == len(self.writeprocs):
                    if self.ngames < self.total:
                        print(f"Warning: only wrote {self.ngames} games out of {self.total}")
                    break
            else:
                self.ngames += len(games)
                for game in games:
                    for fen in game:
                        fen = fen.replace("/", "")
                        yield fen.split()[0]
                print(f"{100*self.ngames/self.total:.2f}% done", end="\r")

    def __del__(self):
        try:
            self.add_p.join(timeout=0.1)
        except:
            pass
        for p in self.writeprocs:
            try:
                p.join(timeout=0.1)
            except:
                pass


class PgnFenIterator:
    def __init__(self, pqfile, fenfile, batch_size):
        self.pgn_iter = PgnIterator(pqfile, batch_size)
        self.fen_iter = FenIterator(fenfile)

    def __iter__(self):
        return itertools.chain(self.pgn_iter, self.fen_iter)


def train_bpe(pqfile, fen_file, batch_size, tokenizer_fn="tokenizer.json"):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        special_tokens=[
            "[ENDOFFEN]",
            "[NOOP]",
            "[ELO1000]",
            "[ELO1100]",
            "[ELO1200]",
            "[ELO1300]",
            "[ELO1400]",
            "[ELO1500]",
            "[ELO1600]",
            "[ELO1700]",
            "[ELO1800]",
            "[ELO1900]",
            "[ELO2000]",
            "[ELO2100]",
            "[ELO2200]",
            "[ELO2300]",
            "[ELO2400]",
            "[ELO2500]",
            "[ELO2600]",
            "[ELO2700]",
            "[ELO2800]",
            "[ELO2900]",
            "[ELO3000]",
            "[ELO3100]",
            "[ELO3200]",
            "[ELO3300]",
            "[ELO3400]",
            "[ELO3500]",
            "[ELO3600]",
            "[ELO3700]",
            "[ELO3800]",
            "[ELO3900]",
            "[ELO4000]",
        ]
    )
    if fen_file:
        tokenizer.train_from_iterator(
            PgnFenIterator(pqfile, fen_file, batch_size), trainer=trainer
        )
    else:
        tokenizer.train_from_iterator(PgnIterator(pqfile, batch_size), trainer=trainer)
    tokenizer.save(tokenizer_fn)


def train_fen(pqfile, batch_size, nwriteprocs, tokenizer_fn="fen_tokenizer.json"):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=100000, show_progress=False)
    tokenizer.train_from_iterator(MPFenIterator(pqfile, batch_size, nwriteprocs), trainer=trainer)

    p = Path(tokenizer_fn)
    ref = p.parent / ("ref_" + p.name)
    tokenizer.save(ref.expanduser().as_posix())

    with open(ref.expanduser()) as f:
        data = json.loads(f.read())

    newdata = {"ranks": data["model"]["vocab"], "pat_str": FenIterator.PAT_STR}
    with open(p.expanduser(), "w") as f:
        json.dump(newdata, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("pqfile", type=str, help="Path to parquet file")
    parser.add_argument(
        "--nprocs",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument("--fen_file", type=str, default=None, help="Path to fens file")
    parser.add_argument(
        "--fens_per_game", type=int, default=10, help="Number of fens per game"
    )
    parser.add_argument(
        "--tokenizer_fn",
        type=str,
        default="tokenizer.json",
        help="Path to tokenizer file",
    )
    parser.add_argument(
        "--serial", action="store_true", default=False, help="Use serial processing"
    )
    parser.add_argument(
        "--fen_only", action="store_true", default=False, help="only train on fen data"
    )
    args = parser.parse_args()

    print("training tokenizer...")
    if args.fen_only:
        train_fen(args.pqfile, args.batch_size, args.nprocs, args.tokenizer_fn)
    else:
        train_bpe(args.pqfile, args.fen_file, args.batch_size, args.tokenizer_fn)
    print(f"tokenizer saved to {args.tokenizer_fn}")
