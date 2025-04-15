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


@dataclass
class MMCModuleArgs:
    name: str
    elo_params: Any
    model_args: ModelArgs
    opening_moves: int
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
        self.opening_moves = params.opening_moves
        self.val_check_steps = params.val_check_steps*params.accumulate_grad_batches
        self.max_steps = params.max_steps
        self.elo_params = params.elo_params
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
                    filename=params.name + "-{valid_move_loss:.2f}",
                    monitor="valid_move_loss",
                )
            )

        self._init_model(params.pretrain_cp)
        self.trainer = L.Trainer(**self.trainer_kwargs)

    def _init_model(self, pretrain_cp):
        if not hasattr(self, 'model'):
            self.model = Transformer(self.model_args)
        if pretrain_cp:
            weights = torch.load(pretrain_cp)
            for i in range(self.params.model_args.n_timecontrol_heads):
                for j in range(self.params.model_args.n_elo_heads):
                    for color in ['white', 'black']:
                        weights[f'model.{color}_heads.{i}.{j}.norm.weight'] = weights['model.move_heads.0.0.norm.weight']
                        weights[f'model.{color}_heads.{i}.{j}.output.weight'] = weights['model.move_heads.0.0.output.weight']
            del weights['model.move_heads.0.0.norm.weight']
            del weights['model.move_heads.0.0.output.weight']

            self.load_state_dict(weights)

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

    def _get_elo_warmup_stage(self):
        if self.elo_params.constant_var:
            return "WARMUP_VAR"
        else:
            if self.global_step < self.elo_params.warmup_elo_steps:
                return "WARMUP_ELO"
            elif (
                self.elo_params.loss == "gaussian_nll"
                and self.global_step
                < self.elo_params.warmup_elo_steps + self.elo_params.warmup_var_steps
            ):
                return "WARMUP_VAR"
            else:
                return "COMPLETE"

    def _format_elo_pred(self, pred, batch):
        if self.params.model_args.n_timecontrol_heads == 1:
            pred = pred[:, :, 0]
        else:
            tc_groups = batch["tc_groups"]
            bs, seqlen, _, ndim = pred.shape
            index = tc_groups[:, None, None, None].expand(
                [bs, seqlen, 1, ndim])
            pred = torch.gather(pred, 2, index).squeeze(2)

        pred = pred.permute(0, 2, 1)
        return pred[:, :, self.opening_moves:]

    def _get_elo_loss(self, elo_pred, batch):
        elo_pred = self._format_elo_pred(elo_pred, batch)
        if self.elo_params.loss == "cross_entropy":
            loss = F.cross_entropy(
                elo_pred, batch["elo_target"], ignore_index=NOOP)
        elif self.elo_params.loss == "gaussian_nll":
            exp = elo_pred[:, 0]
            var = elo_pred[:, 1]
            if self._get_elo_warmup_stage() == "WARMUP_VAR":
                var = torch.ones_like(var) * self.elo_params.initial_var
            exp[batch["elo_target"] == NOOP] = NOOP
            var[batch["elo_target"] == NOOP] = 0
            loss = F.gaussian_nll_loss(exp, batch["elo_target"], var)
        elif self.elo_params.loss == "mse":
            loss = F.mse_loss(elo_pred.squeeze(1), batch["elo_target"])
        else:
            raise Exception(f"unsupported Elo loss: {self.elo_params.loss}")

        return loss, elo_pred

    def _select_tc_head_outputs(self, pred, tc_groups):
        if self.params.model_args.n_timecontrol_heads == 1:
            return pred[:, :, 0]
        bs, seqlen, _, nelo, ncolor, ndim = pred.shape
        index = tc_groups[:, None, None, None, None, None].expand([
            bs,
            seqlen,
            1,
            nelo,
            ncolor,
            ndim,
        ])
        return torch.gather(pred, 2, index).squeeze(2)

    def _select_elo_head_outputs(self, pred, elo_groups):
        if self.params.model_args.n_elo_heads == 1:
            return [pred[:, :, 0, 0].permute(0, 2, 1), None]

        bs, seqlen, _, _, ndim = pred.shape
        preds = []
        for i in [0, 1]:
            subpred = pred[:, :, :, i]
            index = elo_groups[:, i, None, None, None].expand([
                bs,
                seqlen,
                1,
                ndim,
            ])
            subpred = torch.gather(subpred, 2, index).squeeze(2)
            subpred = subpred.permute(0, 2, 1)
            subpred = subpred[:, :, self.opening_moves:]
            preds.append(subpred)

        return preds

    def _format_move_pred(self, pred, batch):
        pred = self._select_tc_head_outputs(pred, batch['tc_groups'])
        preds = self._select_elo_head_outputs(pred, batch['elo_groups'])
        return preds

    def _get_move_loss(self, move_pred, batch):
        w_preds, b_preds = self._format_move_pred(move_pred, batch)
        loss = F.cross_entropy(
            w_preds, batch["move_target"], ignore_index=NOOP)
        if b_preds is not None:
            loss += F.cross_entropy(
                b_preds, batch["move_target"], ignore_index=NOOP)
            loss /= 2
        return loss

    def _get_loss(self, move_pred, elo_pred, batch):
        move_loss = self._get_move_loss(move_pred, batch)
        loss = move_loss.clone()
        elo_loss = torch.zeros_like(elo_pred[0, 0, 0, 0])
        if self._get_elo_warmup_stage() != "WARMUP_ELO":
            elo_loss, elo_pred = self._get_elo_loss(elo_pred, batch)
            loss += self.elo_params.weight * elo_loss

        return loss, move_loss, elo_loss, elo_pred

    def training_step(self, batch, batch_idx):
        move_pred, elo_pred = self(batch["input"])
        loss, move_loss, elo_loss, elo_pred = self._get_loss(
            move_pred, elo_pred, batch)
        self.log("train_loss", loss, sync_dist=True)
        if move_loss is not None:
            self.log("train_move_loss", move_loss,
                     prog_bar=True, sync_dist=True)
        if elo_loss is not None:
            self.log("train_elo_loss", elo_loss, sync_dist=True)
        if (
            self.elo_params.loss == "gaussian_nll"
            and self._get_elo_warmup_stage() == "COMPLETE"
        ):
            self.log(
                "train_avg_std",
                self._get_avg_std(elo_pred, batch["elo_target"])[0],
                sync_dist=True,
            )

        cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        move_pred, elo_pred = self(batch["input"])
        valid_loss, move_loss, elo_loss, elo_pred = self._get_loss(
            move_pred, elo_pred, batch
        )

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        self.log("valid_loss", valid_loss, sync_dist=True)
        if move_loss is not None:
            self.log("valid_move_loss", move_loss,
                     prog_bar=True, sync_dist=True)
        if elo_loss is not None:
            self.log("valid_elo_loss", elo_loss, sync_dist=True)
        if (
            self.elo_params.loss == "gaussian_nll"
            and self._get_elo_warmup_stage() == "COMPLETE"
        ):
            self.log(
                "valid_avg_std",
                self._get_avg_std(elo_pred, batch["elo_target"])[0],
                sync_dist=True,
            )
        return valid_loss

    def _get_avg_std(self, elo_pred, tgts):
        _, std = self.elo_params.whiten_params
        npred = (tgts != NOOP).sum()
        u_std_preds = (elo_pred[:, 1] ** 0.5) * std
        u_std_preds[tgts == NOOP] = 0
        avg_std = u_std_preds.sum() / npred
        return avg_std, u_std_preds

    def _random_move_pred(self, pred):
        bs, _, ntc, nelo, _, _ = pred.shape
        tc_groups = torch.randint(0, ntc, (bs,)).to(pred.device)
        pred = self._select_tc_head_outputs(pred, tc_groups)
        elo_groups = torch.randint(0, nelo, (bs, 2)).to(pred.device)
        return self._select_elo_head_outputs(pred, elo_groups)

    def predict_step(self, batch, batch_idx):
        move_pred, elo_pred = self(batch["input"])
        _, move_loss, elo_loss, elo_pred = self._get_loss(
            move_pred, elo_pred, batch)

        def to_numpy(torchd):
            npd = {}
            for k, v in torchd.items():
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                npd[k] = v
            return npd

        def get_top_5_moves(move_pred):
            probs = torch.softmax(move_pred, dim=1)
            sprobs, smoves = torch.sort(probs, dim=1, descending=True)
            sprobs = sprobs[:, :5]
            smoves = smoves[:, :5]
            return probs, sprobs, smoves

        def combine_moves(w_pred, b_pred):
            if b_pred is None:
                return w_pred
            bs, seqlen, dim = w_pred.shape
            w_pred = w_pred[:, ::2]
            b_pred = b_pred[:, 1::2]
            if seqlen % 2 == 1:
                b_pred = torch.cat(
                    [b_pred, torch.zeros_like(b_pred[:, :1])], dim=1)
            fmt_pred = torch.stack(
                [w_pred, b_pred], dim=2).reshape(bs, -1, dim)
            if seqlen % 2 == 1:
                fmt_pred = fmt_pred[:, :-1]
            return fmt_pred

        def get_move_data():
            w_pred, b_pred = self._format_move_pred(move_pred, batch)
            fmt_pred = combine_moves(w_pred, b_pred)
            fmt_probs, fmt_top_probs, fmt_top_moves = get_top_5_moves(fmt_pred)

            w_rand, b_rand = self._random_move_pred(move_pred)
            rand_pred = combine_moves(w_rand, b_rand)
            rand_probs, rand_top_probs, rand_top_moves = get_top_5_moves(
                rand_pred)

            tgts = batch["move_target"]
            mask = F.one_hot(tgts, fmt_probs.shape[1]).permute(0, 2, 1)
            fmt_tgt_probs = (fmt_probs*mask).sum(dim=1)
            rand_tgt_probs = (rand_probs*mask).sum(dim=1)

            return to_numpy({
                "sorted_tokens": fmt_top_moves,
                "sorted_probs": fmt_top_probs,
                "target_probs": fmt_tgt_probs,
                'rand_tokens': rand_top_moves,
                'rand_probs': rand_top_probs,
                'rand_target_probs': rand_tgt_probs,
                "openings": batch["opening"],
                "targets": tgts,
                "loss": move_loss.item(),
            })

        def get_elo_data():
            tgts = batch["elo_target"]
            mean, std = self.elo_params.whiten_params
            utgts = (tgts * std) + mean

            npred = (tgts != NOOP).sum()

            u_loc_preds = (elo_pred[:, 0] * std) + mean
            u_loc_preds[tgts == NOOP] = 0
            loc_err = (utgts - u_loc_preds).abs()
            loc_err[tgts == NOOP] = 0
            loc_err = loc_err.sum() / npred

            avg_std, u_std_preds = self._get_avg_std(elo_pred, tgts)

            return to_numpy({
                "target_groups": tgts,
                "loss": elo_loss.item(),
                "location_error": loc_err,
                "average_std": avg_std,
                "elo_mean": u_loc_preds,
                "elo_std": u_std_preds,
            })

        return get_move_data(), get_elo_data()

    def fit(self, datamodule, ckpt=None):
        self.trainer.fit(self, datamodule=datamodule, ckpt_path=ckpt)

    def predict(self, datamodule):
        tkargs = self.trainer_kwargs
        trainer = L.Trainer(**tkargs)
        outputs = trainer.predict(self, datamodule)
        return outputs
