torchrun --standalone --nnodes 1 --nproc-per-node 8 train.py --train_heads --core_ckpt  ~/mimicChessData/models/v0.3/ckpt/step-100k/v0.3-core-train_loss\=1.37.ckpt --cfg cfghead.yml --outfn v0.3
