"""
Regime-stratified forecasting metrics.

All metrics are computed over individual (sample, horizon_step, link) triples
— NOT over whole windows.  A single window contributes independently to both
the recurrent and NR buckets depending on each target (step, link)'s label.

Functions
─────────
compute_metrics        Main entry point.  Returns a flat dict of all metrics.
inverse_transform      Helper: unnormalise predictions using per-link stats.

Metric keys in the returned dict
─────────────────────────────────
{regime}_{metric}_{horizon}
  regime  ∈ {overall, recurrent, nr}
  metric  ∈ {mae, rmse, mape}
  horizon ∈ {h1, h2, h3, h4, h5, h6, avg}   (h1=5min … h6=30min)

Example keys
    overall_mae_avg      overall_rmse_h3      nr_mape_avg
    recurrent_mae_h1     nr_rmse_h6
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-6  # avoid div-by-zero in MAPE


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - true)))


def _rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - true) ** 2)))


def _mape(pred: np.ndarray, true: np.ndarray) -> float:
    mask = np.abs(true) > _EPS
    if mask.sum() == 0:
        return float("nan")
    return float(
        100.0 * np.nanmean(np.abs(pred[mask] - true[mask]) / np.abs(true[mask]))
    )


def inverse_transform(
    pred_norm: np.ndarray,  # (..., N)
    speed_mean: np.ndarray,  # (N,)
    speed_std: np.ndarray,  # (N,)
) -> np.ndarray:
    """Undo per-link z-normalisation: pred_orig = pred_norm * std + mean."""
    return pred_norm * speed_std + speed_mean


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(
    pred: np.ndarray,  # (M, output_len, N) — original mph
    target: np.ndarray,  # (M, output_len, N) — original mph
    full_nr: np.ndarray,  # (M, output_len, N) — binary 0/1
    interval_min: int = 5,
) -> dict[str, float]:
    """
    Compute regime-stratified MAE / RMSE / MAPE.

    Stratification is per (sample, horizon_step, link):
      - recurrent : full_nr == 0
      - nr        : full_nr == 1
      - overall   : all predictions

    Parameters
    ----------
    pred         : (M, output_len, N) predictions in original mph
    target       : (M, output_len, N) ground truth in original mph
    full_nr      : (M, output_len, N) binary NR label for each (step, link)
    interval_min : minutes per step (default 5)

    Returns
    -------
    dict of float, keys follow the pattern {regime}_{metric}_{horizon}
    """
    assert pred.shape == target.shape == full_nr.shape, (
        f"Shape mismatch: pred={pred.shape} target={target.shape} "
        f"full_nr={full_nr.shape}"
    )

    M, L_out, N = pred.shape
    results: dict[str, float] = {}

    nr_mask = full_nr.astype(bool)  # (M, L_out, N)
    rec_mask = ~nr_mask

    regimes = {
        "overall": np.ones((M, L_out, N), dtype=bool),
        "recurrent": rec_mask,
        "nr": nr_mask,
    }

    for regime_name, mask in regimes.items():
        per_horizon_mae = []
        per_horizon_rmse = []
        per_horizon_mape = []

        for k in range(L_out):
            h_mask = mask[:, k, :]  # (M, N)
            p_h = pred[:, k, :][h_mask]
            t_h = target[:, k, :][h_mask]

            if h_mask.sum() == 0:
                mae_k = rmse_k = mape_k = float("nan")
            else:
                mae_k = _mae(p_h, t_h)
                rmse_k = _rmse(p_h, t_h)
                mape_k = _mape(p_h, t_h)

            h_key = f"h{k + 1}"
            results[f"{regime_name}_mae_{h_key}"] = round(mae_k, 4)
            results[f"{regime_name}_rmse_{h_key}"] = round(rmse_k, 4)
            results[f"{regime_name}_mape_{h_key}"] = round(mape_k, 4)

            per_horizon_mae.append(mae_k)
            per_horizon_rmse.append(rmse_k)
            per_horizon_mape.append(mape_k)

        # Horizon-averaged (ignoring NaN horizons)
        def _nanmean_safe(lst: list) -> float:
            vals = [v for v in lst if not np.isnan(v)]
            return float(np.mean(vals)) if vals else float("nan")

        results[f"{regime_name}_mae_avg"] = round(_nanmean_safe(per_horizon_mae), 4)
        results[f"{regime_name}_rmse_avg"] = round(_nanmean_safe(per_horizon_rmse), 4)
        results[f"{regime_name}_mape_avg"] = round(_nanmean_safe(per_horizon_mape), 4)

        # Sample counts for this regime
        results[f"{regime_name}_n_predictions"] = int(mask.sum())

    # NR rate in the test set
    results["nr_rate_pct"] = round(100.0 * float(nr_mask.mean()), 3)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: collect predictions from a DataLoader
# ─────────────────────────────────────────────────────────────────────────────


def collect_predictions(
    model_fn,  # callable: batch_dict -> (M, L_out, N) numpy or torch
    dataloader,  # torch DataLoader
    nd,  # NetworkData (for inverse transform)
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model_fn over a DataLoader and collect (pred, target, full_nr).

    model_fn receives a batch dict (values on `device`) and should return
    predictions as a (batch, output_len, N) array or tensor in normalised
    units.  This function handles denormalisation automatically.

    Returns
    -------
    pred    : (M_total, output_len, N)  original mph
    target  : (M_total, output_len, N)  original mph
    full_nr : (M_total, output_len, N)  binary 0/1
    """
    import torch

    all_pred, all_target, all_nr = [], [], []

    for batch in dataloader:
        # Move to device if applicable
        batch_d = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            pred_norm = model_fn(batch_d)  # (B, L_out, N) normalised

        if isinstance(pred_norm, torch.Tensor):
            pred_norm = pred_norm.cpu().numpy()

        # Denormalise: (B, L_out, N)
        pred_orig = inverse_transform(pred_norm, nd.speed_mean, nd.speed_std)

        all_pred.append(pred_orig)
        all_target.append(batch["y_orig"].numpy())
        all_nr.append(batch["full_nr"].numpy())

    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_target, axis=0),
        np.concatenate(all_nr, axis=0),
    )
