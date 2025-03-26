import argparse
import json
import os
import time
from multiprocessing import Process, Queue

import chess
import numpy as np
import pgnutils.inference as inf
from pgnutils import PrintSafe, get_eta, mvid_to_uci, uci_to_mvid

from mmcdataset import load_data


def get_gain_and_cp_loss(scores, idx):
    if idx % 2 == 0:
        gain = scores[1] - scores[0]
        cp_loss = scores[1] - scores[2]
    else:
        gain = scores[0] - scores[1]
        cp_loss = scores[2] - scores[1]
    return gain, cp_loss


class CheatGenerator:
    def __init__(
        self,
        sfbin,
        mistake_thresh,
        cp_loss_frac,
        depth_limit,
        verbose,
        print_safe,
    ):
        self.engine = chess.engine.SimpleEngine.popen_uci(sfbin)
        self.thresh = mistake_thresh
        self.lf = cp_loss_frac
        self.dl = depth_limit
        self.verbose = verbose
        if self.verbose:
            self.printfn = print_safe
        else:
            self.printfn = lambda a=None, b=None, c=None: None
        self.print_dbg = print_safe

    def close(self):
        self.engine.quit()

    def _is_miss(self, scores, idx):
        gain, cp_loss = get_gain_and_cp_loss(scores, idx)
        return gain >= self.thresh and cp_loss >= self.lf * gain

    def _get_score(self, board, maxval=1000):
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.dl))
        score = info["score"]
        if score.is_mate():
            moves = score.relative.moves
            if moves == 0:
                val = -maxval if score.turn else maxval
            elif (score.turn and moves > 0) or (not score.turn and moves < 0):
                val = maxval
            else:
                val = -maxval
        else:
            val = score.relative.score()
            if not score.turn:
                val = -val
            if val < 0:
                val = max(val, -maxval)
            else:
                val = min(val, maxval)
        return val, info["pv"][0] if "pv" in info else None

    def _print_if_verbose(self, idx, score, miss):
        if self.verbose:
            s = ""
            if idx % 2 == 0:
                s = f"{idx // 2}.".rjust(4)
            score = f"{score}"
            if miss:
                score += "*"
            s += score.rjust(11)
            print(s, end="", flush=True)
            if idx % 2 == 1:
                print()

    def process_game(self, mvids, gid):
        board_state, white, black = inf.board_state()
        board = chess.pgn.Game().board()
        scores = []
        cheat_moves = []
        prev_best = None

        self.printfn(f"Game {gid}")
        for idx, mvid in enumerate(mvids):
            uci = mvid_to_uci(mvid, board_state, white, black, False)
            mv = chess.Move.from_uci(uci)
            if not board.is_legal(mv):
                self.printfn("illegal move".rjust(11))
                break
            board.push(mv)
            score, best_mv = self._get_score(board)
            scores.append(score)
            miss = False
            if idx >= 2 and self._is_miss(scores[-3:], idx):
                miss = True
                gain, cp_loss = get_gain_and_cp_loss(scores[-3:], idx)
                if cp_loss > 2000:
                    self.print_dbg(f"game {gid} had cp_loss={cp_loss}")
                cheat_mvid = uci_to_mvid(prev_best.uci(), white, black)
                cheat_moves.append([idx, cheat_mvid, gain, cp_loss])

            self._print_if_verbose(idx, score, miss)

            mvid_to_uci(mvid, board_state, white, black)  # update states
            prev_best = best_mv

        return np.array(cheat_moves)


def load_games(indices, mvids):
    games = []
    for gs, nmoves, gidx in indices:
        games.append((mvids[gs : gs + nmoves], gidx))
    return games


def load_games_queue(games_q, indices, mvids, nproc):
    for gs, nmoves, gidx in indices:
        games_q.put((mvids[gs : gs + nmoves], gidx))
    for _ in range(nproc):
        games_q.put((None, -1))


def process_game_serial(args, game_data):
    game, gidx = game_data
    generator = CheatGenerator(
        args.sfbin,
        args.cp_mistake_thresh,
        args.cp_loss_fraction,
        args.stockfish_depth,
        args.verbose,
        print,
    )
    cm = generator.process_game(game, gidx)
    generator.close()
    return cm, gidx


def process_games(pid, args, games_q, output_q, print_safe):
    generator = CheatGenerator(
        args.sfbin,
        args.cp_mistake_thresh,
        args.cp_loss_fraction,
        args.stockfish_depth,
        args.verbose,
        print_safe,
    )
    while True:
        game, gidx = games_q.get()
        if gidx == -1:
            break
        cm = generator.process_game(game, gidx)
        output_q.put((cm, gidx))

    generator.close()


def start_processes(sf_args, games_q, output_q, print_safe, nproc):
    procs = []
    for pid in range(nproc):
        p = Process(
            target=process_games, args=((pid, sf_args, games_q, output_q, print_safe))
        )
        p.daemon = True
        p.start()
        procs.append(p)
    return procs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--datadir", help="directory containing dataset files")
    parser.add_argument(
        "--sfbin", help="stockfish binary path", default="/opt/homebrew/bin/stockfish"
    )
    parser.add_argument(
        "--cp_mistake_thresh",
        type=int,
        default=200,
        help="threshold for loss in centipawns for classifying a move as a mistake",
    )
    parser.add_argument(
        "--cp_loss_fraction",
        default=0.5,
        type=float,
        help="fraction of centipawn loss to classify a move as a miss",
    )
    parser.add_argument(
        "--stockfish_depth", default=20, type=int, help="stockfish depth limit"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="verbose printing"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        default=False,
        help="run in serial (non-parallel) mode for debugging",
    )
    parser.add_argument(
        "--nproc",
        default=os.cpu_count() - 1,
        help="number of parallel stockfishes",
        type=int,
    )
    parser.add_argument(
        "--outfn", default="cheatdata.npy", help="name of output npy file"
    )

    args = parser.parse_args()
    data = load_data(args.datadir)
    ngames = len(data["test"])
    cheat_data = {}

    if args.serial:
        games = load_games(data["test"], data["mvids"])
        for cd, gidx in games:
            cheat_data[gidx] = process_game_serial(args, (cd, gidx))
    else:
        games_q = Queue()
        output_q = Queue()
        print_safe = PrintSafe()

        start = time.time()
        load_games_queue(games_q, data["test"], data["mvids"], args.nproc)
        procs = start_processes(args, games_q, output_q, print_safe, args.nproc)
        while True:
            cm, gidx = output_q.get()
            cheat_data[gidx] = cm
            if len(cheat_data) == ngames:
                break

            if len(cheat_data) % args.nproc == 0:
                eta = get_eta(ngames, len(cheat_data), start)
                print_safe(
                    f"Processed {len(cheat_data)} of {ngames} games (eta: {eta})"
                )
            else:
                print_safe(f"Processed {len(cheat_data)} of {ngames} games", end="\r")

        for p in procs:
            p.join()

    np.save(
        os.path.join(args.datadir, args.outfn),
        cheat_data,
        allow_pickle=True,
    )
    with open(
        os.path.join(args.datadir, f"{os.path.basename(args.outfn)}.md.json"), "w"
    ) as f:
        json.dump(
            {
                "cp_mistake_threshold": args.cp_mistake_thresh,
                "cp_loss_fraction": args.cp_loss_fraction,
                "stockfish_depth": args.stockfish_depth,
            },
            f,
        )
