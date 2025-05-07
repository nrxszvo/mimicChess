import io
import itertools
import os
import random
from multiprocessing import Process, Queue

import chess.pgn
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def pgns_to_fens(pgns):
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
        for i, move in enumerate(game.mainline_moves()):
            cumsum += minv + i * crem
            if r < cumsum:
                break
            board.push(move)

        fens.append(board.fen())
    return fens


def pgns_to_fens_proc(inq, outq):
    while True:
        pgns = inq.get()
        if len(pgns) == 0:
            outq.put([])
            return

        outq.put(pgns_to_fens(pgns))


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


def create_fens(pqfile, nwriteprocs, batch_size, fen_fn="fens.raw"):
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

    total = pq.ParquetFile(pqfile).metadata.num_rows
    nfen = 0
    ndone = 0
    with open(fen_fn, "w") as ff:
        while True:
            fens = outq.get()
            if len(fens) == 0:
                ndone += 1
                if ndone == nwriteprocs:
                    break
            else:
                nfen += len(fens)
                ff.write("\n".join(fens) + "\n")
                print(f"{100*nfen/total:.2f}% done", end="\r")

    add_p.join()
    for p in writeprocs:
        p.join()

    return fen_fn


class PgnIterator:
    def __init__(self, pqfile, batch_size):
        self.iter = pq.ParquetFile(pqfile).iter_batches(
            batch_size=batch_size, columns=["moves"]
        )

    def __iter__(self):
        for batch in self.iter:
            for pgn in batch["moves"]:
                yield pgn.as_py()


class FenIterator:
    def __init__(self, fenfile):
        self.fenfile = fenfile

    def __iter__(self):
        with open(self.fenfile) as f:
            for fen in f:
                yield fen


class PgnFenIterator:
    def __init__(self, pqfile, fenfile, batch_size):
        self.pgn_iter = PgnIterator(pqfile, batch_size)
        self.fen_iter = FenIterator(fenfile)

    def __iter__(self):
        return itertools.chain(self.pgn_iter, self.fen_iter)


def train_bpe(pqfile, fen_file, batch_size, tokenizer_fn="tokenizer.json"):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=[
            "[ENDOFFEN]",
            "[NOOP]",
            "[WHITEWINS]",
            "[BLACKWINS]",
            "[DRAW]",
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
    tokenizer.train_from_iterator(
        PgnFenIterator(pqfile, fen_file, batch_size), trainer=trainer
    )
    tokenizer.save(tokenizer_fn)


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
    parser.add_argument(
        "--fen_file", type=str, default="fens.raw", help="Path to fens file"
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
    args = parser.parse_args()

    if not os.path.exists(args.fen_file):
        print("generating fens...")
        if args.serial:
            create_fens_serial(args.pqfile, args.batch_size, args.fen_file)
        else:
            create_fens(args.pqfile, args.nprocs, args.batch_size, args.fen_file)
    else:
        print(f"using existing fen file: {args.fen_file}")

    print("training tokenizer...")
    train_bpe(args.pqfile, args.fen_file, args.batch_size, args.tokenizer_fn)
    print(f"tokenizer saved to {args.tokenizer_fn}")
