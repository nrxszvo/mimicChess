from lib.pgnutils import BoardState
from lib.training.mmcdataset import load_data
import numpy as np

def analyze_positions(dirname):
    data = load_data(dirname)
    train = data['train']
    test = data['test']
    blocks = data['blocks']
    train_pos = set()
    one_p = int(0.01*len(train))
    print('Collecting training positions...')
    for i, (gidx, gs, nmoves, blk) in enumerate(train):
        if i % one_p == 0:
            print(f"{100*i/len(train)}% done", end='\r')
        mvids = blocks[blk]['mvids']
        game = mvids[gs: gs + nmoves]
        board = BoardState()
        for mv in game:
            board.update(mv)
            train_pos.add(board.print())

    print(f'Found {len(train_pos)} unique training positions')

    unique_test = []
    one_p = int(0.01*len(test))
    print('Collecting test positions...')
    for i, (gidx, gs, nmoves, blk) in enumerate(test):
        if i % one_p == 0:
            print(f"{100*i/len(test)}% done", end='\r')
        mvids = blocks[blk]['mvids']
        game = mvids[gs: gs + nmoves]
        board = BoardState()
        unique = []
        for j, mv in enumerate(game):
            board.update(mv)
            if board.print() not in train_pos:
                unique.append(j)

        if len(unique) > 0:
            unique_test.extend([gidx, gs, blk, len(unique), *unique])
    
    np.array(unique_test, dtype='int64').tofile(f'{dirname}/unique_test.npy')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str)
    args = parser.parse_args()
    analyze_positions(args.dirname)