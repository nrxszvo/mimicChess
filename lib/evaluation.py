import os
from functools import partial
from collections import Counter

import numpy as np
from multiprocessing import Queue, Process

from .training.mmcdataset import NOOP
from .pgnutils.reconstruct import count_invalid

SINGLE_THREAD = False


class LegalGameStats:
    def eval_game(gameq, resultq):
        while True:
            pred, opn, tgt = gameq.get()
            if pred is None:
                break
            nmoves, nfail, reasons = count_invalid(pred[0], opn, tgt)
            nvalid_moves = nmoves - nfail
            resultq.put((nvalid_moves, nmoves, nfail == 0, reasons))

    def __init__(self, nproc=os.cpu_count()-1):
        self.nvalid_games = 0
        self.ntotal_games = 0
        self.nvalid_moves = 0
        self.ntotal_moves = 0
        self.reasons = Counter()
        self.gameq = Queue()
        self.resultq = Queue()
        self.procs = []
        if not SINGLE_THREAD:
            for _ in range(nproc):
                p = Process(
                    target=LegalGameStats.eval_game, args=(
                        self.gameq, self.resultq)
                )
                p.daemon = True
                p.start()
                self.procs.append(p)

    def eval(self, tokens, opening, tgts):
        self.ntotal_games += tokens.shape[0]
        ngames = len(tokens)
        for game in zip(tokens, opening, tgts):
            if SINGLE_THREAD:
                pred, opn, tgt = game
                nmoves, nfail, reasons = count_invalid(pred[0], opn, tgt)
                nvalid_moves = nmoves - nfail
                self.nvalid_moves += nvalid_moves
                self.ntotal_moves += nmoves
                if nfail == 0:
                    self.nvalid_games += 1
                else:
                    self.reasons.update(reasons)
            else:
                self.gameq.put(game)

        if not SINGLE_THREAD:
            for _ in range(ngames):
                nvalid_moves, nmoves, is_valid_game, reasons = self.resultq.get()
                self.nvalid_moves += nvalid_moves
                self.ntotal_moves += nmoves
                if is_valid_game:
                    self.nvalid_games += 1
                else:
                    self.reasons.update(reasons)

    def __del__(self):
        for _ in range(len(self.procs)):
            self.gameq.put((None, None, None))
        for p in self.procs:
            p.join()

    def report(self):
        lines = []
        lines.append(
            f"Legal game frequency: {100 * self.nvalid_games / self.ntotal_games:.3f}%"
        )
        lines.append(
            f"Legal move frequency: {100 * self.nvalid_moves / self.ntotal_moves:.3f}%"
        )
        for reason, count in self.reasons.items():
            lines.append(f'illegal {reason} moves: {count}')

        return lines


class TargetStats:
    def __init__(self):
        self.total_preds = 0
        self.sum_t = 0
        self.sum_adj = 0

    def eval(self, tprobs, adjprobs, tgts, cdf_scores):
        if tprobs is not None:
            tprobs[tgts == NOOP] = 0
            adjprobs[tgts == NOOP] = 0
            self.sum_t += tprobs.sum()
            self.sum_adj += adjprobs.sum()
        if cdf_scores is not None:
            cdf_scores[tgts == NOOP] = 0
            self.sum_t += cdf_scores.sum()

        self.total_preds += (tgts != NOOP).sum()

    def report(self):
        return [
            f"Mean target probability/cdf score: {100 * self.sum_t / self.total_preds:.2f}%",
            f"Mean adjacent probability: {100 * self.sum_adj / self.total_preds:.2f}%",
        ]


class AccuracyStats:
    def __init__(self, seq_len, elo_edges, min_prob, ns=[1, 3]):
        self.stats = [{"n": n, "matches": np.zeros(seq_len)} for n in ns]
        self.min_prob = min_prob
        self.total_preds = np.zeros(seq_len)
        self.elo_edges = elo_edges
        n_groups = len(elo_edges)
        self.histo = np.zeros((n_groups, n_groups))
        self.acc_mtx = np.zeros((n_groups, n_groups, 2))
        self.err_mtx = np.zeros((n_groups, n_groups, 2))
        self.move_mtx = np.zeros((n_groups, n_groups, 2))

    def eval(self, tokens, probs, tgts):
        tokens[probs < self.min_prob] = -1
        for i in range(tgts.shape[0]):
            self.histo[tgts[i, 0], tgts[i, 1]] += 1

        npred = (tgts != NOOP).sum(axis=0)
        self.total_preds += npred
        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts[:, None]).sum(axis=1)
            move_matches[tgts == NOOP] = 0
            s["matches"] += move_matches.sum(axis=0)

            if s["n"] == 1:
                for i in range(tgts.shape[0]):
                    wE = tgts[i, 0]
                    bE = tgts[i, 1]

                    wM = move_matches[i, ::2].sum()
                    bM = move_matches[i, 1::2].sum()
                    self.acc_mtx[wE, bE, 0] += wM
                    self.acc_mtx[wE, bE, 1] += bM

                    errors = (tokens[i, 0] - tgts[i]).abs()
                    errors[tgts[i] == NOOP] = 0
                    wM = errors[::2].sum()
                    bM = errors[1::2].sum()
                    self.err_mtx[wE, bE, 0] += wM
                    self.err_mtx[wE, bE, 1] += bM

                    wN = (tgts[i, ::2] != NOOP).sum()
                    bN = (tgts[i, 1::2] != NOOP).sum()
                    self.move_mtx[wE, bE, 0] += wN
                    self.move_mtx[wE, bE, 1] += bN

    def report(self, SEQ_AVG=10):
        if self.total_preds.sum() == 0:
            return []
        lines = []
        for s in self.stats:
            lines.append(f"Top {s['n']} accuracy:")
            for i, v in enumerate(self.total_preds):
                if i % SEQ_AVG == 0:
                    cum_acc = 0
                    cum_v = 0
                if v > 0:
                    cum_acc += s["matches"][i]
                    cum_v += v
                if i % SEQ_AVG == (SEQ_AVG - 1) or i == len(self.total_preds) - 1:
                    lines.append(
                        f"\t{SEQ_AVG * int(i / SEQ_AVG)}: {100 * cum_acc / cum_v:.1f}%"
                    )

        def gen_mtx_report(mtx, fn, COLWIDTH=10):
            lines.append("")
            for wbin, row in reversed(list(enumerate(mtx))):
                rowstr = f"{self.elo_edges[wbin]}".rjust(COLWIDTH)
                for bbin, n in enumerate(row):
                    rowstr += fn(n, wbin, bbin).rjust(COLWIDTH)
                lines.append(rowstr)
                lines.append("")
            ax = "".rjust(COLWIDTH)
            for e in self.elo_edges:
                ax += f"{e}".rjust(COLWIDTH)
            lines.append(ax)
            lines.append("")

        lines.append("Elo histogram:")
        gen_mtx_report(self.histo, lambda data, *args: f"{int(data)}")

        def fmt_acc(mul, cum_accs, wbin, bbin):
            n_row = self.move_mtx[wbin]
            Nw, Nb = n_row[bbin]
            accW, accB = cum_accs
            acc_str = ""
            if Nw > 0:
                acc = accW / Nw
                acc_str += f"{mul * acc:.1f}, "
            else:
                acc_str += "- ,"
            if Nb > 0:
                acc = accB / Nb
                acc_str += f"{mul * acc:.1f}"
            else:
                acc_str += "- "
            return acc_str

        lines.append("Avg Error Matrix:")
        gen_mtx_report(self.err_mtx, partial(fmt_acc, 1), COLWIDTH=20)

        lines.append("Accuracy Matrix:")
        gen_mtx_report(self.acc_mtx, partial(fmt_acc, 100), COLWIDTH=20)

        return lines


class MoveStats:
    def __init__(self, ns=[1, 3, 5]):
        self.stats = [{"n": n, "matches": 0} for n in ns]
        self.total_preds = 0

    def eval(self, tokens, tgts):
        self.total_preds += (tgts != NOOP).sum()

        for s in self.stats:
            move_matches = (tokens[:, : s["n"]] == tgts[:, None]).sum(axis=1)
            move_matches[tgts == NOOP] = 0
            s["matches"] += move_matches.sum()

    def report(self):
        lines = []
        for s in self.stats:
            lines.append(
                f"Top {s['n']} move accuracy: {100 * s['matches'] / self.total_preds:.2f}%"
            )
        return lines


class CheatStats:
    class EloStats:
        def __init__(self, elo):
            self.elo = elo
            self.below = [0, 0]
            self.above = [0, 0]

    def __init__(self, elo_edges):
        self.stats = [CheatStats.EloStats(elo) for elo in elo_edges]

    def eval(self, head_probs, cheat_probs, cheatdata, heads):
        for tps, cps, cd, hs in zip(head_probs, cheat_probs, cheatdata, heads):
            for i, cp in enumerate(cps):
                offset = cd[i, 0]
                tp = tps[offset]
                head = hs[0] if offset % 2 == 0 else hs[1]
                stats = self.stats[head]
                if cp < tp:
                    stats.below[0] += cp / tp
                    stats.below[1] += 1
                else:
                    stats.above[0] += cp / tp
                    stats.above[1] += 1

    def report(self):
        lines = []
        lines.append("Stockfish / Target mean probability ratios:")
        for stats in self.stats:
            lines.append(f"\tElo < {stats.elo}")
            if stats.below[1] > 0:
                lines.append(
                    f"\t\tstockfish < target: {stats.below[0] / stats.below[1]:.4f} ({stats.below[1]} total moves)"
                )
            if stats.above[1] > 0:
                lines.append(
                    f"\t\tstockfish > target: {stats.above[0] / stats.above[1]:.4f} ({stats.above[1]} total moves)"
                )
        return lines
