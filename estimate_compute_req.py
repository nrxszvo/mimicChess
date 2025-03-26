import argparse


def compute(tau=None, T=None, N=None, D=None):
    # tau*T = 6*N*D
    lhs = 1
    rhs = 6
    n_none = 4
    if N is not None:
        rhs *= N * 10**6
        n_none -= 1
    if D is not None:
        rhs *= D * 10**9
        n_none -= 1
    if tau is not None:
        lhs *= tau * 10**12
        n_none -= 1
    if T is not None:
        lhs *= T * 24 * 3600
        n_none -= 1

    assert n_none == 1, "Must have exactly one of N, D, tau, T equal to None"

    if N is None:
        print(f"Model size: {lhs/rhs:.2e} weights")
    if D is None:
        print(f"Dataset size: {lhs/rhs:.2e} tokens")
    if tau is None:
        print(f"FLOPS requirement: {rhs/lhs/1e12:.2f} teraFLOPs")
    if T is None:
        print(f"Training time: {rhs/lhs/(24*3600):.2f} days")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", help="compute in teraFLOPs", default=None, type=float)
    parser.add_argument("--T", help="time in days", default=None, type=float)
    parser.add_argument(
        "--N", help="number of parameters in millions", default=None, type=float
    )
    parser.add_argument(
        "--D", help="number of training tokens in billions", default=None, type=float
    )
    args = parser.parse_args()
    compute(args.tau, args.T, args.N, args.D)
