from dataclasses import dataclass
from typing import Optional, Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

from .mmcdataset import NOOP
from .model import ModelArgs, Transformer
from ..pgnutils.reconstruct import count_invalid


@dataclass
class MMCModuleArgs:
    name: str
    model_args: ModelArgs
    lr_scheduler_params: Any
    max_steps: int
    val_check_steps: int
    accumulate_grad_batches: int
    random_seed: Optional[int] = 0
    strategy: Optional[str] = 'auto'
    devices: Optional[int] = 0
    outdir: Optional[str] = None
    pretrain_cp: Optional[str] = None
    num_nodes: Optional[int] = 1


class MimicChessModule(L.LightningModule):
    def __init__(self, params: MMCModuleArgs):
        super().__init__()
        self.save_hyperparameters()
        self.params = params
        L.seed_everything(params.random_seed, workers=True)
        self.model_args = params.model_args
        self.val_check_steps = params.val_check_steps*params.accumulate_grad_batches
        self.max_steps = params.max_steps
        if params.name:
            logger = TensorBoardLogger(".", name="L", version=params.name)
        else:
            logger = None
        val_check_interval = min(
            self.val_check_steps, params.max_steps*params.accumulate_grad_batches)
        self.lr_scheduler_params = params.lr_scheduler_params
        if torch.cuda.is_available():
            precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else 16
            accelerator = "gpu"
        else:
            precision = 32
            accelerator = "cpu"

        self.trainer_kwargs = {
            'num_nodes': params.num_nodes,
            "logger": logger,
            "max_steps": params.max_steps,
            "val_check_interval": val_check_interval,
            "check_val_every_n_epoch": None,
            "strategy": params.strategy,
            "devices": params.devices,
            "precision": precision,
            "accelerator": accelerator,
            "callbacks": [TQDMProgressBar()],
            'accumulate_grad_batches': params.accumulate_grad_batches
        }
        if params.outdir is not None:
            self.trainer_kwargs["callbacks"].append(
                ModelCheckpoint(
                    dirpath=params.outdir,
                    filename=params.name + "-{valid_loss:.2f}",
                    monitor="valid_loss",
                )
            )

        self._init_model()
        self.trainer = L.Trainer(**self.trainer_kwargs)

    def _init_model(self):
        if not hasattr(self, 'model'):
            self.model = Transformer(self.model_args)

    def num_params(self):
        nparams = 0
        nwflops = 0
        for name, w in self.model.named_parameters():
            if w.requires_grad:
                nparams += w.numel()
                if (
                    "embeddings" not in name
                    and "norm" not in name
                    and "bias" not in name
                ):
                    nwflops += w.numel()
        return nparams, nwflops

    def configure_optimizers(self):
        lr = self.lr_scheduler_params.lr
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=lr)
        name = self.lr_scheduler_params.name
        if name == "Cosine":
            min_lr = self.lr_scheduler_params.min_lr
            scheduler = CosineAnnealingLR(
                optimizer=optimizer, T_max=self.max_steps, eta_min=min_lr
            )
            freq = 1
        elif name == "WarmUpCosine":
            warmup_steps = self.lr_scheduler_params.warmup_steps
            warmupLR = LinearLR(
                optimizer=optimizer,
                start_factor=1 / warmup_steps,
                end_factor=1,
                total_iters=warmup_steps,
            )
            min_lr = self.lr_scheduler_params.min_lr
            cosineLR = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_steps - warmup_steps,
                eta_min=min_lr,
            )
            scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmupLR, cosineLR],
                milestones=[warmup_steps],
            )
            freq = 1
        else:
            raise Exception(f"unsupported scheduler: {name}")

        config = {
            "scheduler": scheduler,
            "frequency": freq,
            "interval": "step",
            "monitor": "valid_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": config}

    def forward(self, tokens):
        return self.model(tokens)

    def _get_loss(self, move_pred, batch):
        move_pred = move_pred[:, 1:].permute(0,2,1)
        loss = F.cross_entropy(
            move_pred, batch["move_target"], ignore_index=NOOP)
        return loss

    def training_step(self, batch, batch_idx):
        move_pred = self(batch["input"])
        loss = self._get_loss(
            move_pred, batch)
        self.log("train_loss", loss, sync_dist=True)
        self.log("train_move_loss", loss,
                     prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def get_top_5_moves(self, move_pred):
        probs = torch.softmax(move_pred, dim=1)
        sprobs, smoves = torch.sort(probs, dim=1, descending=True)
        sprobs = sprobs[:, :5]
        smoves = smoves[:, :5]
        return probs, sprobs, smoves

    def validation_step(self, batch, batch_idx):
        move_pred = self(batch["input"])
        valid_loss = self._get_loss(
            move_pred, batch
        )

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, sync_dist=True)

        #total_moves = 0
        #total_legal_moves = 0
        #_, _, preds = self.get_top_5_moves(move_pred[:,1:].permute(0,2,1))
        #for pred, tgt in zip(preds, batch["move_target"]):
        #    nmoves, nfails, _ = count_invalid(pred.cpu().numpy(), [], tgt.cpu().numpy())
        #    total_moves += nmoves
        #    total_legal_moves += nmoves - nfails[0]

        #self.log('valid_legal_prob', total_legal_moves/total_moves, sync_dist=True)

        return valid_loss

    def predict_step(self, batch, batch_idx):
        move_pred = self(batch["input"])
        move_loss = self._get_loss(
            move_pred, batch)

        def to_numpy(torchd):
            npd = {}
            for k, v in torchd.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                npd[k] = v
            return npd


        fmt_probs, fmt_top_probs, fmt_top_moves = self.get_top_5_moves(move_pred)

        tgts = batch["move_target"]
        mask = F.one_hot(tgts, fmt_probs.shape[1]).permute(0, 2, 1)
        fmt_tgt_probs = (fmt_probs*mask).sum(dim=1)

        return to_numpy({
            "sorted_tokens": fmt_top_moves,
            "sorted_probs": fmt_top_probs,
            "target_probs": fmt_tgt_probs,
            "targets": tgts,
            "loss": move_loss.item(),
        })

    def fit(self, datamodule, ckpt=None):
        self.trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt)

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        trainer = L.Trainer(**tkargs)
        outputs = trainer.predict(self, datamodule)
        return outputs
