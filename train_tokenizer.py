from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import io
import chess.pgn
import random
from multiprocessing import Queue, Process
import pyarrow.parquet as pq
import os
import itertools

def pgns_to_fens(inq, outq):
    while True:
        pgns = inq.get()
        if len(pgns) == 0:
            outq.put([])
            return

        fens = []
        for pgn in pgns:
            pgn = pgn.as_py()
            nmoves = len(pgn.split())
            skew = 4 
            uni = 1/(nmoves+1)
            minv = 1/(skew*(nmoves+1))
            crem = (uni-minv)*(2/nmoves)
            cumsum = 0

            game = chess.pgn.read_game(io.StringIO(pgn))
            board = game.board()
            r = random.random()

            for i, move in enumerate(game.mainline_moves()):
                cumsum += minv + i*crem
                if r < cumsum:
                    break
                board.push(move)

            fens.append(board.fen())

        outq.put(fens)

def add_pgns(pqfile, inq, batch_size, nwriteprocs):
    for batch in pq.ParquetFile(pqfile).iter_batches(batch_size=batch_size, columns=['moves']):
        inq.put(batch['moves'])

    for _ in range(nwriteprocs):
        inq.put([])

def create_fens(pqfile, nwriteprocs, batch_size, fen_fn='fens.raw'):
    inq = Queue()
    outq = Queue()

    add_p = Process(target=add_pgns, args=(pqfile, inq, batch_size, nwriteprocs))
    add_p.daemon = True
    add_p.start()

    writeprocs = []
    for i in range(nwriteprocs):
        p = Process(target=pgns_to_fens, args=(inq, outq))
        p.daemon = True
        p.start()
        writeprocs.append(p)

    total = pq.ParquetFile(pqfile).metadata.num_rows
    nfen = 0
    with open(fen_fn, 'w') as ff:
        while True:
            fens = outq.get()
            if len(fens)==0:
                ndone += 1
                if ndone == nwriteprocs:
                    break
            else:
                nfen += len(fens)
                ff.write('\n'.join(fens) + '\n')
                print(f'{100*nfen/total:.2f}% done', end='\r')

    add_p.join()
    for p in writeprocs:
        p.join()

    return fen_fn

class PgnIterator:
    def __init__(self, pqfile, batch_size):
        self.iter = pq.ParquetFile(pqfile).iter_batches(batch_size=batch_size, columns=['moves'])
    def __iter__(self):
        for batch in self.iter:
            for pgn in batch['moves']:
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

def train_bpe(pqfile, fen_file, batch_size, tokenizer_fn='tokenizer.json'):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[START]", "[NOOP]"])
    tokenizer.train_from_iterator(PgnFenIterator(pqfile, fen_file, batch_size), trainer=trainer)
    tokenizer.save(tokenizer_fn)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pqfile', type=str)
    parser.add_argument('--nprocs', type=int, default=os.cpu_count()-1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--fen_file', type=str, default='fens.raw')
    parser.add_argument('--tokenizer_fn', type=str, default='tokenizer.json')
    args = parser.parse_args()

    if not os.path.exists(args.fen_file):
        print('generating fens...')
        create_fens(args.pqfile, args.nprocs, args.batch_size, args.fen_file)
    else:
        print(f'using existing fen file: {args.fen_file}')

    print('training tokenizer...')
    train_bpe(args.pqfile, args.fen_file, args.batch_size, args.tokenizer_fn)
    print(f'tokenizer saved to {args.tokenizer_fn}')