import tiktoken
import json
import pyarrow.parquet as pq
from train_tokenizer import pgns_to_fens_proc
from collections import Counter
import numpy as np
from multiprocessing import Process, Queue
import argparse
import os

def add_pgns(pqfile, inq, batch_size, nwriteprocs):
    for batch in pq.ParquetFile(pqfile).iter_batches(
        batch_size=batch_size, columns=["moves"]
    ):
        inq.put(batch["moves"])

    for _ in range(nwriteprocs):
        inq.put([])


def calc_fen_token_histo(pqfile, nwriteprocs, batch_size, tok, byte_vocab):
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
    histo = Counter()
    for token in byte_vocab.values():
        histo[token] = 0

    nfen = 0
    ndone = 0
    nloops = 0
    n_tokens = 0
    while True:
        nloops += 1
        fens = outq.get()
        if len(fens) == 0:
            ndone += 1
            if ndone == nwriteprocs:
                print()
                if nfen < total:
                    print(f"Warning: only processed {nfen} fens out of {total}")
                break
        else:
            nfen += len(fens)
            for fen in fens:
                tokens = tok.encode(fen)
                n_tokens += len(tokens)
                histo.update(tokens)
            #print(histo.most_common(5), end="\r")
            if nloops % 10 == 0:
                print(f"{100*nfen/total:.3f}% done", end="\r")

    add_p.join()
    for p in writeprocs:
        p.join()

    return histo, n_tokens / nfen


def test_tokenizer(tokenizer_fn, pqfile, batch_size, max_tokens):
    with open(tokenizer_fn) as f:
        data = json.loads(f.read())
    
    byte_vocab = {k.encode(): v for k, v in data['model']['vocab'].items() if v < max_tokens}
    print(f"Number of tokens: {len(byte_vocab)}")

    tok = tiktoken.Encoding(name='mimictokens', pat_str='\w+|[^\w\s]+', mergeable_ranks=byte_vocab, special_tokens={})

    histo, avg_n_tokens = calc_fen_token_histo(pqfile, os.cpu_count() - 1, batch_size, tok, byte_vocab)

    hs, es = np.histogram(list(histo.values()), [0, 0.1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])

    for e in es[1:]:
        print(f'{int(e):.1e}'.rjust(10), end="")
    print()
    for h in hs:
        print(str(int(h)).rjust(10), end="")
    print()
    print(f"Average number of tokens per fen: {avg_n_tokens:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_fn", default="lichess_db_2025_04_tokens.json")
    parser.add_argument("--pqfile", default="datasets/lichess_db/2025-04/data.parquet")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=int(1e10))
    args = parser.parse_args()
    test_tokenizer(args.tokenizer_fn, args.pqfile, args.batch_size, args.max_tokens)

