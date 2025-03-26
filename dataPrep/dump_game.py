import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", help="pgn filename", required=True)
    parser.add_argument("--line", help="starting line of game", type=int, required=True)
    parser.add_argument("--out", help="pgn output filename", required=True)
    parser.add_argument("--n", help="number of consecutive games", default=1, type=int)
    args = parser.parse_args()

    assert args.line > 0, "line number starts at 1"
    n = 0
    with open(args.pgn, "r") as fin:
        lineno = 0
        for line in fin:
            lineno += 1
            if lineno == args.line:
                break
        data = []
        for line in fin:
            data.append(line)
            if line[0] == "1":
                n += 1
                if n == args.n:
                    break

    with open(args.out, "w") as fout:
        fout.writelines(data)


if __name__ == "__main__":
    main()
