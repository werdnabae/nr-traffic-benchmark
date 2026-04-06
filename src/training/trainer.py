"""
Unified training loop for all PyTorch DL models.

Supports four training strategies:
  "standard"         Standard MSE loss on all target (step, link) pairs
  "weighted_loss"    MSE loss with per-(step, link) weight = causal_fixed
                     upweighted by nr_loss_weight for confirmed NR steps
  "nr_finetune"      Two-phase: standard pretraining → fine-tune on NR samples
  "multi_objective"  Speed MSE + auxiliary BCE on causal_fixed NR labels

All strategies use:
  • Adam optimiser with weight decay
  • Gradient clipping
  • Early stopping on validation MAE (overall)
  • Checkpoint saving (best model by val MAE)

Usage
─────
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
    preds, targets, nrs, regimes = trainer.predict(test_loader, nd)
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, inverse_transform


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────


def _mse_loss(
    pred: torch.Tensor,  # (B, L_out, N)
    target: torch.Tensor,  # (B, L_out, N)  normalised
    weight: torch.Tensor | None = None,  # (B, L_out, N)
) -> torch.Tensor:
    err = (pred - target) ** 2
    if weight is not None:
        err = err * weight
    return err.mean()


def _multi_obj_loss(
    pred_speed: torch.Tensor,  # (B, L_out, N)
    target_speed: torch.Tensor,  # (B, L_out, N)
    pred_logit: torch.Tensor,  # (B, L_out, N)  raw logits for NR class
    target_nr: torch.Tensor,  # (B, L_out, N)  causal_fixed labels [0/1]
    nr_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    speed_loss = _mse_loss(pred_speed, target_speed)
    nr_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_logit, target_nr.float()
    )
    total = speed_loss + nr_weight * nr_loss
    return total, speed_loss, nr_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """
    Trains and evaluates a PyTorch model.

    Parameters
    ----------
    model      : nn.Module — must implement forward(batch) → (B, L_out, N)
    nd         : NetworkData — for inverse transform in validation
    config     : training sub-dict from config.yaml
    strategy   : "standard" | "weighted_loss" | "nr_finetune" | "multi_objective"
    checkpoint_dir : where to save best model; None = no saving
    """

    def __init__(
        self,
        model: nn.Module,
        nd,
        config: dict,
        strategy: str = "standard",
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        assert strategy in (
            "standard",
            "weighted_loss",
            "nr_finetune",
            "multi_objective",
        ), f"Unknown strategy '{strategy}'"

        self.model = model
        self.nd = nd
        self.cfg = config
        self.strategy = strategy
        self.ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None

        device_str = config.get("device", "cpu")
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        elif device_str == "mps" and not torch.backends.mps.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)

        self.model.to(self.device)

        # Mixed precision: enabled on CUDA unless the model explicitly opts out
        # (e.g. DCRNN uses sparse mm which has no fp16 kernel; ASTGCN/DSTAGNN
        # have attention softmax overflow in fp16).
        model_disables_amp = getattr(model, "disable_amp", False)
        self._use_amp = (self.device.type == "cuda") and not model_disables_amp
        self._scaler = torch.cuda.amp.GradScaler() if self._use_amp else None
        if self._use_amp:
            # Let cuDNN auto-tune kernel selection for fixed input sizes
            torch.backends.cudnn.benchmark = True

        self.lr = float(config.get("lr", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.clip_grad = float(config.get("clip_grad", 5.0))
        self.epochs = int(config.get("epochs", 100))
        self.patience = int(config.get("patience", 15))
        # multi_objective uses a smaller NR weight — nr_weight=20 dominates speed loss
        if strategy == "multi_objective":
            self.nr_weight = float(config.get("multi_obj_nr_weight", 1.0))
        else:
            self.nr_weight = float(config.get("nr_loss_weight", 5.0))
        self.ft_lr_mult = float(config.get("finetune_lr_multiplier", 0.1))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Multi-objective: auxiliary NR head
        self._nr_head: nn.Linear | None = None
        if strategy == "multi_objective":
            # Assumes model has a hidden_dim attribute
            hidden_dim = getattr(model, "hidden_dim", 64)
            # Head takes raw speed prediction (size 1) as proxy signal
            self._nr_head = nn.Linear(1, 1).to(self.device)
            self.optimizer.add_param_group({"params": self._nr_head.parameters()})

        self.history: dict[str, list] = {
            "train_loss": [],
            "val_mae_overall": [],
            "val_mae_nr": [],
            "val_mae_rec": [],
        }
        self._best_val_mae = float("inf")
        self._no_improve = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch_d = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            self.optimizer.zero_grad()

            # ── Forward pass (fp16 on CUDA via AMP) ──────────────────────────
            with torch.cuda.amp.autocast(enabled=self._use_amp):
                pred = self.model(batch_d)  # (B, L_out, N)
                target_norm = batch_d["y"]  # (B, L_out, N) normalised

                if self.strategy == "standard":
                    loss = _mse_loss(pred, target_norm)

                elif self.strategy == "weighted_loss":
                    nr_tgt = batch_d["full_nr"]
                    weight = 1.0 + (self.nr_weight - 1.0) * nr_tgt
                    loss = _mse_loss(pred, target_norm, weight=weight)

                elif self.strategy == "nr_finetune":
                    loss = _mse_loss(pred, target_norm)

                elif self.strategy == "multi_objective":
                    nr_tgt = batch_d["full_nr"]
                    nr_logit = self._nr_head(pred.unsqueeze(-1)).squeeze(-1)
                    loss, _, _ = _multi_obj_loss(
                        pred, target_norm, nr_logit, nr_tgt, self.nr_weight
                    )
                else:
                    loss = _mse_loss(pred, target_norm)

            # Skip NaN/Inf losses (e.g. AGCRN on bad initialisation)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()
                continue

            # ── Backward pass (fp32 gradients via GradScaler) ────────────────
            if self._scaler is not None:
                self._scaler.scale(loss).backward()
                if self.clip_grad > 0:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            # Every batch produced a non-finite loss and was skipped.
            # Return nan (not 0.0) so fit() can detect this condition and
            # reinitialise the model rather than silently doing nothing.
            return float("nan")
        return total_loss / n_batches

    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_pred, all_target, all_nr = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch_d = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                pred_norm = self.model(batch_d).cpu().numpy()
                pred_orig = inverse_transform(
                    pred_norm, self.nd.speed_mean, self.nd.speed_std
                )
                all_pred.append(pred_orig)
                all_target.append(batch["y_orig"].numpy())
                all_nr.append(batch["full_nr"].numpy())

        pred = np.concatenate(all_pred, axis=0)
        target = np.concatenate(all_target, axis=0)
        nr = np.concatenate(all_nr, axis=0)

        return compute_metrics(pred, target, nr)

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        finetune_loader: DataLoader | None = None,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        train_loader     : standard training DataLoader
        val_loader       : validation DataLoader
        finetune_loader  : for "nr_finetune" — DataLoader of NR-only samples;
                           if None, one is auto-built by filtering train_loader
        """
        model_name = getattr(self.model, "name", type(self.model).__name__)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n{'─' * 60}", flush=True)
        print(f"  Training : {model_name}  ({n_params:,} params)", flush=True)
        print(f"  Strategy : {self.strategy}", flush=True)
        print(f"  Device   : {self.device}", flush=True)
        print(
            f"  Epochs   : up to {self.epochs}  (patience={self.patience})", flush=True
        )
        print(f"{'─' * 60}", flush=True)

        # ── Phase 1: standard training (all strategies) ───────────────────────
        phase1_epochs = (
            self.epochs // 2 if self.strategy == "nr_finetune" else self.epochs
        )

        for epoch in range(1, phase1_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)

            # ── Detect bad initialisation: all batches skipped (NaN output) ──
            # This can happen with models whose attention softmax overflows with
            # certain weight initialisations (e.g. ASTGCN on large graphs).
            # Retry up to _MAX_REINIT times with a new seed before giving up.
            _MAX_REINIT = 5
            if np.isnan(train_loss) and epoch == 1:
                for _attempt in range(1, _MAX_REINIT + 1):
                    print(
                        f"  [Trainer] WARNING: all batches produced non-finite loss "
                        f"(bad initialisation, attempt {_attempt}/{_MAX_REINIT}). "
                        f"Re-initialising with new seed.",
                        flush=True,
                    )
                    _reinit_model(self.model)
                    # Reset optimizer so stale momentum doesn't carry over
                    self.optimizer = torch.optim.Adam(
                        list(self.model.parameters())
                        + (
                            list(self._nr_head.parameters())
                            if self._nr_head is not None
                            else []
                        ),
                        lr=self.lr,
                        weight_decay=self.weight_decay,
                    )
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode="min", factor=0.5, patience=5
                    )
                    train_loss = self._train_epoch(train_loader)
                    if not np.isnan(train_loss):
                        print(
                            f"  [Trainer] Re-initialisation succeeded on attempt {_attempt}.",
                            flush=True,
                        )
                        break
                else:
                    print(
                        f"  [Trainer] ERROR: model still produces non-finite loss "
                        f"after {_MAX_REINIT} re-initialisations. Aborting training.",
                        flush=True,
                    )
                    return
            val_m = self._validate(val_loader)
            val_mae = val_m["overall_mae_avg"]

            self.scheduler.step(val_mae)
            self.history["train_loss"].append(train_loss)
            self.history["val_mae_overall"].append(val_mae)
            self.history["val_mae_nr"].append(val_m.get("nr_mae_avg", float("nan")))
            self.history["val_mae_rec"].append(
                val_m.get("recurrent_mae_avg", float("nan"))
            )

            improved = val_mae < self._best_val_mae
            if improved:
                self._best_val_mae = val_mae
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._no_improve = 0
                self._save_checkpoint("best_model.pt")
            else:
                self._no_improve += 1

            star = "✓" if improved else " "
            print(
                f"  [{star}] Epoch {epoch:3d}/{phase1_epochs}  "
                f"loss={train_loss:.4f}  "
                f"val={val_mae:.4f}  "
                f"val_NR={val_m.get('nr_mae_avg', float('nan')):.4f}  "
                f"val_rec={val_m.get('recurrent_mae_avg', float('nan')):.4f}  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

            if self._no_improve >= self.patience:
                print(
                    f"  Early stopping at epoch {epoch}  "
                    f"(no improvement for {self.patience} epochs).",
                    flush=True,
                )
                break

        # ── Phase 2: NR fine-tuning (if strategy == "nr_finetune") ────────────
        if self.strategy == "nr_finetune":
            print("\n[Trainer] Phase 2: NR fine-tuning")
            # Restore best checkpoint from phase 1
            if hasattr(self, "_best_state"):
                self.model.load_state_dict(self._best_state)

            # Build NR-only DataLoader if not provided
            if finetune_loader is None:
                finetune_loader = _build_nr_loader(train_loader)

            if finetune_loader is None or len(finetune_loader) == 0:
                print("  No NR samples found in training set; skipping fine-tune.")
            else:
                # Reduced LR for fine-tuning
                for g in self.optimizer.param_groups:
                    g["lr"] = self.lr * self.ft_lr_mult
                self._no_improve = 0

                for epoch in range(1, phase1_epochs + 1):
                    t0 = time.time()
                    train_loss = self._train_epoch(finetune_loader)
                    val_m = self._validate(val_loader)
                    val_mae = val_m["overall_mae_avg"]
                    self.scheduler.step(val_mae)

                    improved = val_mae < self._best_val_mae
                    if improved:
                        self._best_val_mae = val_mae
                        self._best_state = copy.deepcopy(self.model.state_dict())
                        self._no_improve = 0
                        self._save_checkpoint("best_model_finetune.pt")
                    else:
                        self._no_improve += 1

                    star = "✓" if improved else " "
                    print(
                        f"  [{star}] FT Epoch {epoch:3d}  loss={train_loss:.4f}  "
                        f"val={val_mae:.4f}  "
                        f"val_NR={val_m.get('nr_mae_avg', float('nan')):.4f}  "
                        f"{time.time() - t0:.1f}s",
                        flush=True,
                    )
                    if self._no_improve >= self.patience:
                        print(f"  Fine-tune early stopping at epoch {epoch}.")
                        break

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        print(f"  Done. Best val MAE={self._best_val_mae:.4f}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        loader: DataLoader,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run predictions over a DataLoader.

        Returns (pred_orig, target_orig, full_nr, regime)  each (M, L, N).
        """
        self.model.eval()
        preds, targets, nrs, regimes = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                batch_d = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                pred_norm = self.model(batch_d).cpu().numpy()
                pred_orig = inverse_transform(
                    pred_norm, self.nd.speed_mean, self.nd.speed_std
                )
                preds.append(pred_orig)
                targets.append(batch["y_orig"].numpy())
                nrs.append(batch["full_nr"].numpy())
                regimes.append(batch["regime"].numpy())

        return (
            np.concatenate(preds, axis=0),
            np.concatenate(targets, axis=0),
            np.concatenate(nrs, axis=0),
            np.concatenate(regimes, axis=0),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, fname: str) -> None:
        if self.ckpt_dir is None:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / fname
        torch.save(self.model.state_dict(), path)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: re-initialise model weights (for bad-init recovery)
# ─────────────────────────────────────────────────────────────────────────────


def _reinit_model(model: nn.Module) -> None:
    """
    Re-initialise all parameters of a model by calling reset_parameters()
    on every sub-module that has one.  The caller should advance torch's PRNG
    (e.g. via torch.manual_seed) before calling this so the new initialisation
    differs from the original.
    """
    # Advance the seed so we get a genuinely different initialisation
    current_seed = torch.initial_seed()
    torch.manual_seed((current_seed + 1) % (2**32))

    def _reset(m: nn.Module) -> None:
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            try:
                m.reset_parameters()
            except Exception:
                pass  # some custom layers may not support it; skip silently

    model.apply(_reset)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build NR-only DataLoader for fine-tuning
# ─────────────────────────────────────────────────────────────────────────────


def _build_nr_loader(
    train_loader: DataLoader,
    min_nr_steps: int = 1,
) -> DataLoader | None:
    """
    Build a DataLoader containing only samples that have at least min_nr_steps
    NR timestep in the target window (full_nr.any()).
    """
    dataset = train_loader.dataset
    nr_indices = []

    print("[Trainer] Scanning training set for NR samples...")
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["full_nr"].sum() >= min_nr_steps:
            nr_indices.append(i)

    if not nr_indices:
        return None

    nr_subset = Subset(dataset, nr_indices)
    print(f"  NR samples: {len(nr_indices)} / {len(dataset)}")

    return DataLoader(
        nr_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
    )
