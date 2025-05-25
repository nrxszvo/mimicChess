import tiktoken
import json
import pyarrow.parquet as pq
from train_tokenizer import pgns_to_fens_proc, add_pgns
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import argparse
import os


def pass_through_pgns(inq, outq, *args):
    while True:
        pgns = inq.get()
        outq.put([pgn.as_py() for pgn in pgns])
        if len(pgns) == 0:
            break


def calc_token_histo(
    pqfile, nwriteprocs, batch_size, tok, vocab, proc, samples_per_game
):
    inq = Queue()
    outq = Queue()

    add_p = Process(target=add_pgns, args=(pqfile, inq, batch_size, nwriteprocs))
    add_p.daemon = True
    add_p.start()

    writeprocs = []
    for i in range(nwriteprocs):
        p = Process(target=proc, args=(inq, outq, samples_per_game))
        p.daemon = True
        p.start()
        writeprocs.append(p)

    total = samples_per_game * pq.ParquetFile(pqfile).metadata.num_rows
    histo = Counter()
    for token in vocab.values():
        histo[token] = 0

    ndone = 0
    nloops = 0
    n_samples = 0
    n_tokens = 0
    while True:
        nloops += 1
        samples = outq.get()
        if len(samples) == 0:
            ndone += 1
            if ndone == nwriteprocs:
                print()
                if n_samples < total:
                    print(f"Warning: only processed {n_samples} samples out of {total}")
                break
        else:
            n_samples += len(samples)
            for sample in samples:
                for fen in sample:
                    fen = fen.replace("/", "")
                    fen = fen.split()[0]
                    tokens = tok.encode(fen)
                    n_tokens += len(tokens)
                    histo.update(tokens)
                # print(histo.most_common(5), end="\r")
            if nloops % 10 == 0:
                print(f"{n_samples:.2e}, {100*n_samples/total:.3f}% done", end="\r")

    try:
        add_p.join(timeout=0.1)
    except TimeoutError:
        pass
    for p in writeprocs:
        try:
            p.join(timeout=0.1)
        except TimeoutError:
            pass

    return histo, n_tokens, n_samples


def test_tokenizer(
    tokenizer_fn, pqfile, batch_size, max_tokens, proc, nwriteprocs, samples_per_game
):
    with open(tokenizer_fn) as f:
        data = json.loads(f.read())
    vocab = {k.encode(): v for k, v in data["ranks"].items() if v < max_tokens}
    print(f"Number of tokens: {len(vocab)}")

    tok = tiktoken.Encoding(
        name="mimictokens",
        pat_str=data["pat_str"],
        mergeable_ranks=vocab,
        special_tokens={},
    )

    histo, n_tokens, n_games = calc_token_histo(
        pqfile, nwriteprocs, batch_size, tok, vocab, proc, samples_per_game
    )

    hs, es = np.histogram(
        list(histo.values()), [0, 0.1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    )

    decoder = {v: k.decode("utf-8") for k, v in vocab.items()}
    with open(f'{tokenizer_fn.split(".")[0]}_stats.txt', "w") as f:
        f.write(f"# tokens: {n_tokens:.2e}\n")
        f.write(f"# games: {n_games:.2e}\n")
        f.write(f"tokens/game: {n_tokens/n_games:.2f}\n")
        f.write("\n")
        for e in es[1:]:
            f.write(f"{int(e):.1e}".rjust(10))
        f.write("\n")
        for h in hs:
            f.write(str(int(h)).rjust(10))
        f.write("\n\n")
        for k, v in histo.most_common():
            f.write(f"{decoder[k].rjust(10)}: {v}\n")

    print(f"# tokens: {n_tokens:.2e}")
    print(f"# games: {n_games:.2e}")
    print(f"tokens/game: {n_tokens/n_games:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_fn", default="lichess_db_2025_04_tokens.json")
    parser.add_argument("--pqfile", default="datasets/lichess_db/2025-04/data.parquet")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=int(1e10))
    parser.add_argument("--proc", default="pgn", choices=["pgn", "fen"])
    parser.add_argument("--nwriteprocs", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--fens_per_game", type=int, default=10)
    args = parser.parse_args()

    if args.proc == "pgn":
        proc = pass_through_pgns
    elif args.proc == "fen":
        proc = pgns_to_fens_proc

    test_tokenizer(
        args.tokenizer_fn,
        args.pqfile,
        args.batch_size,
        args.max_tokens,
        proc,
        args.nwriteprocs,
        args.fens_per_game,
    )
