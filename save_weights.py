import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("cp")
parser.add_argument("outfn")

if __name__ == "__main__":
    args = parser.parse_args()
    cp = torch.load(args.cp, map_location=torch.device("cpu"), weights_only=False)
    torch.save(cp["state_dict"], args.outfn)
