import pyarrow.parquet as pq
import os


def test_elo_dir(elo_dir, elo_edges):
    prev_welo = 0
    for i, welo in enumerate(elo_edges):
        if i > 0:
            prev_welo = elo_edges[i - 1]
        prev_belo = 0
        for j, belo in enumerate(elo_edges):
            if j > 0:
                prev_belo = elo_edges[j - 1]
            pqfile = pq.ParquetFile(
                os.path.join(elo_dir, str(welo), str(belo), "data.parquet")
            )
            for batch in pqfile.iter_batches():
                assert (
                    batch["welo"].to_numpy().min() >= prev_welo
                ), f'{batch["welo"].to_numpy().min()} < {prev_welo}'
                assert (
                    batch["belo"].to_numpy().min() >= prev_belo
                ), f'{batch["belo"].to_numpy().min()} < {prev_belo}'
                assert (
                    batch["welo"].to_numpy().max() < welo
                ), f'{batch["welo"].to_numpy().max()} >= {welo}'
                assert (
                    batch["belo"].to_numpy().max() < belo
                ), f'{batch["belo"].to_numpy().max()} >= {belo}'

    print("All tests passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--elo_dir", type=str, required=True)
    parser.add_argument(
        "--elo_edges",
        type=int,
        nargs="+",
        default=[
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
            3000,
            4000,
        ],
    )
    args = parser.parse_args()
    test_elo_dir(args.elo_dir, args.elo_edges)
