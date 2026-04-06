"""
Transition analysis: onset (0→1) vs. persistence (1→1) and anomaly prediction.

Two complementary analyses:

1. Regime-stratified metrics
   ──────────────────────────
   For each test sample the (output_len × N) regime array already computed
   by the Dataset (REGIME_RECURRENT / UNOBSERVED_ONSET / CONFIRMED_NR) is
   used to stratify MAE / RMSE per regime across the whole test set.

2. Anomaly prediction accuracy
   ─────────────────────────────
   Given predictions in mph and per-link v_rec thresholds, classify each
   predicted (step, link) as "predicted NR" or not.  Compare against the
   ground-truth full_nr label.  Report precision / recall / F1 overall and
   separately for onset vs. persistence steps.

   This answers: "Can the model's speed forecast implicitly signal an
   upcoming or ongoing NR event?"
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import (
    REGIME_RECURRENT,
    REGIME_UNOBSERVED_ONSET,
    REGIME_CONFIRMED_NR,
)
from src.evaluation.metrics import _mae, _rmse, _mape


# ─────────────────────────────────────────────────────────────────────────────
# 1. Regime-stratified metrics from pre-classified regime arrays
# ─────────────────────────────────────────────────────────────────────────────


def compute_regime_metrics(
    pred: np.ndarray,  # (M, output_len, N)  original mph
    target: np.ndarray,  # (M, output_len, N)  original mph
    regime: np.ndarray,  # (M, output_len, N)  int8 regime codes
    interval_min: int = 5,
) -> dict[str, float | int]:
    """
    Compute MAE / RMSE / MAPE stratified by the four transition regimes.

    Also reports the count (n_*) for each regime so the reader can assess
    how rare each category is.

    Returns
    ───────
    dict with keys:
      {regime_name}_{metric}   where regime_name ∈
        {unobserved_onset, confirmed_nr, recurrent}
      and metric ∈ {mae, rmse, mape, n}
    """
    mapping = {
        "recurrent": REGIME_RECURRENT,
        "unobserved_onset": REGIME_UNOBSERVED_ONSET,
        "confirmed_nr": REGIME_CONFIRMED_NR,
    }

    out: dict[str, float | int] = {}
    for name, code in mapping.items():
        mask = regime == code  # (M, L, N)
        n = int(mask.sum())
        out[f"{name}_n"] = n
        if n == 0:
            out[f"{name}_mae"] = float("nan")
            out[f"{name}_rmse"] = float("nan")
            out[f"{name}_mape"] = float("nan")
        else:
            p = pred[mask]
            t = target[mask]
            out[f"{name}_mae"] = round(_mae(p, t), 4)
            out[f"{name}_rmse"] = round(_rmse(p, t), 4)
            out[f"{name}_mape"] = round(_mape(p, t), 4)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Anomaly prediction accuracy
# ─────────────────────────────────────────────────────────────────────────────


def _pr_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return dict(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        tp=tp,
        fp=fp,
        fn=fn,
    )


def anomaly_prediction_analysis(
    pred: np.ndarray,  # (M, output_len, N)  original mph
    full_nr: np.ndarray,  # (M, output_len, N)  binary 0/1
    v_rec: np.ndarray,  # (output_len, N) or (N,)  mph thresholds
    regime: np.ndarray,  # (M, output_len, N)  int8 regime codes
) -> dict[str, float | int]:
    """
    Binary anomaly detection accuracy using predicted speed vs. v_rec.

    A (step, link) is classified as "predicted NR" iff pred < v_rec.

    Results are reported overall, and separately for:
      • onset    — ground-truth REGIME_UNOBSERVED_ONSET steps
      • confirmed— ground-truth REGIME_CONFIRMED_NR steps
      • all_nr   — all ground-truth NR steps (union of above)

    v_rec shape: if (N,) it is broadcast over all (M, L) positions.
                 if (output_len, N) it is broadcast over M.

    Returns
    ───────
    dict with keys: {scope}_precision, {scope}_recall, {scope}_f1, {scope}_tp, ...
    """
    M, L, N = pred.shape

    # Broadcast v_rec to (M, L, N)
    if v_rec.ndim == 1:
        vr = v_rec[np.newaxis, np.newaxis, :]  # (1, 1, N)
    elif v_rec.ndim == 2:
        vr = v_rec[np.newaxis, :, :]  # (1, L, N)
    else:
        vr = v_rec

    pred_nr = (pred < vr).astype(bool)  # (M, L, N) predicted NR
    true_nr = full_nr.astype(bool)  # (M, L, N) ground truth NR

    def _counts(mask: np.ndarray) -> tuple[int, int, int]:
        """TP, FP, FN within a boolean mask."""
        tp = int((pred_nr & true_nr & mask).sum())
        fp = int((pred_nr & ~true_nr & mask).sum())
        fn = int((~pred_nr & true_nr & mask).sum())
        return tp, fp, fn

    out: dict[str, float | int] = {}

    # Overall
    everything = np.ones((M, L, N), dtype=bool)
    tp, fp, fn = _counts(everything)
    for k, v in _pr_f1(tp, fp, fn).items():
        out[f"overall_{k}"] = v

    # All NR
    nr_mask = true_nr
    tp, fp, fn = _counts(nr_mask)
    for k, v in _pr_f1(tp, fp, fn).items():
        out[f"all_nr_{k}"] = v

    # Onset (unobserved)
    onset_mask = regime == REGIME_UNOBSERVED_ONSET
    tp, fp, fn = _counts(onset_mask)
    for k, v in _pr_f1(tp, fp, fn).items():
        out[f"onset_{k}"] = v

    # Confirmed NR
    conf_mask = regime == REGIME_CONFIRMED_NR
    tp, fp, fn = _counts(conf_mask)
    for k, v in _pr_f1(tp, fp, fn).items():
        out[f"confirmed_{k}"] = v

    # Counts
    out["n_onset_steps"] = int(onset_mask.sum())
    out["n_confirmed_steps"] = int(conf_mask.sum())
    out["n_recurrent_steps"] = int((regime == REGIME_RECURRENT).sum())

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Collect regime arrays from a DataLoader
# ─────────────────────────────────────────────────────────────────────────────


def collect_regime_arrays(
    dataloader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather regime and full_nr arrays from a DataLoader without running a model.

    Returns
    ───────
    regime  : (M_total, output_len, N) int8
    full_nr : (M_total, output_len, N) float32
    """
    all_regime, all_nr = [], []
    for batch in dataloader:
        all_regime.append(batch["regime"].numpy())
        all_nr.append(batch["full_nr"].numpy())
    return (
        np.concatenate(all_regime, axis=0),
        np.concatenate(all_nr, axis=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full transition report (combines both analyses)
# ─────────────────────────────────────────────────────────────────────────────


def full_transition_report(
    pred: np.ndarray,  # (M, L, N) original mph
    target: np.ndarray,  # (M, L, N) original mph
    full_nr: np.ndarray,  # (M, L, N) binary
    regime: np.ndarray,  # (M, L, N) int8
    v_rec: np.ndarray,  # (N,) or (L, N) mph thresholds
) -> dict[str, float | int]:
    """
    Convenience wrapper: returns the union of regime_metrics and
    anomaly_prediction_analysis in a single flat dict.
    """
    out: dict[str, float | int] = {}
    out.update(compute_regime_metrics(pred, target, regime))
    out.update(anomaly_prediction_analysis(pred, full_nr, v_rec, regime))
    return out
