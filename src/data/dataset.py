"""
TrafficDataset: session-aware sliding-window PyTorch Dataset.

Each sample is a (input_len, output_len) window pair.  Windows that would
span an overnight / weekend gap (session boundary) are excluded.

__getitem__ returns a dict with:

  x            (input_len,  N, C_in)  float32   normalised input features
               If feature config includes "nr_causal", the back-filled
               context-dependent causal NR label is appended as the last
               channel at runtime.

  y            (output_len, N)        float32   normalised target speed
               (used for training loss)

  y_orig       (output_len, N)        float32   original speed in mph
               (used for evaluation metrics)

  full_nr      (output_len, N)        float32   ground-truth NR labels
               for the target window  [0/1]

  causal_fixed_end  (N,)              float32   causal_fixed[T, :] where
               T = last input step; used for regime classification

  nr_backfill  (input_len,  N)        float32   back-filled causal NR label
               for the input context (context-dependent version B)
               Only non-zero when "nr_causal" in feature config, but always
               returned for convenience.

  regime       (output_len, N)        int8      per-(step, link) regime code:
                 0 = recurrent
                 1 = unobserved onset  (NR in target, causal_fixed_end = 0)
                 2 = confirmed NR      (NR in target, causal_fixed_end = 1)

  sample_idx   scalar int             original start index in full time axis
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object  # fallback base class so class definition works
    _TORCH_AVAILABLE = False

from src.data.loader import NetworkData
from src.data.features import FeatureBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Regime codes (exported for use in evaluation modules)
# ─────────────────────────────────────────────────────────────────────────────

REGIME_RECURRENT = 0
REGIME_UNOBSERVED_ONSET = 1
REGIME_CONFIRMED_NR = 2


def classify_regimes(
    causal_fixed_end: np.ndarray,  # (N,)   bool / float
    full_nr_target: np.ndarray,  # (output_len, N) binary
) -> np.ndarray:
    """
    Per-(step, link) regime classification.

    Parameters
    ----------
    causal_fixed_end : (N,) — causal_fixed at last input step
    full_nr_target   : (output_len, N) — ground-truth NR in target window

    Returns
    -------
    regime : (output_len, N) int8
      0 = recurrent
      1 = unobserved onset
      2 = confirmed NR
    """
    output_len, N = full_nr_target.shape
    confirmed = (causal_fixed_end > 0).astype(bool)  # (N,)
    confirmed_bc = confirmed[np.newaxis, :]  # (1, N)

    is_nr = full_nr_target > 0  # (output_len, N)

    regime = np.zeros((output_len, N), dtype=np.int8)
    regime[is_nr & ~confirmed_bc] = REGIME_UNOBSERVED_ONSET
    regime[is_nr & confirmed_bc] = REGIME_CONFIRMED_NR

    return regime


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class TrafficDataset(Dataset):
    """
    Session-aware sliding-window dataset for traffic speed forecasting.

    Parameters
    ----------
    nd             : NetworkData instance
    feature_builder: FeatureBuilder instance (defines which features to use)
    split          : "train" | "val" | "test"
    input_len      : number of input timesteps (default 9 = 45 min)
    output_len     : number of output timesteps (default 6 = 30 min)
    """

    def __init__(
        self,
        nd: NetworkData,
        feature_builder: FeatureBuilder,
        split: str = "train",
        input_len: int = 9,
        output_len: int = 6,
    ) -> None:
        assert split in ("train", "val", "test"), (
            f"split must be 'train', 'val', or 'test', got '{split}'"
        )

        self.nd = nd
        self.fb = feature_builder
        self.split = split
        self.L_in = input_len
        self.L_out = output_len

        self._valid_starts: List[int] = self._find_valid_starts()

    # ─────────────────────────────────────────────────────────────────────────

    def _find_valid_starts(self) -> List[int]:
        """
        Collect all valid window start indices for this split.

        A window [start, start + L_in + L_out) is valid iff:
          1. It lies entirely within the split's time range.
          2. It does not span a session boundary (no overnight gap).
        """
        T = self.nd.T
        L = self.L_in + self.L_out
        mask = self.nd.split_mask[self.split]
        sids = self.nd.session_ids

        # Find the contiguous block of indices for this split
        # (the split mask is a contiguous slice, but we check explicitly)
        split_indices = np.where(mask)[0]
        if len(split_indices) < L:
            return []

        first = int(split_indices[0])
        last = int(split_indices[-1])

        valid = []
        for s in range(first, last - L + 2):
            # Check split boundary
            if not mask[s] or not mask[s + L - 1]:
                continue
            # No session boundary within the window
            if sids[s] == sids[s + L - 1]:
                valid.append(s)

        return valid

    # ─────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __getitem__(self, idx: int) -> dict:
        start = self._valid_starts[idx]
        end_in = start + self.L_in  # exclusive
        end_out = end_in + self.L_out  # exclusive
        T_last = end_in - 1  # last input timestep index

        # ── static features ──────────────────────────────────────────────────
        x_static = self.fb.features[start:end_in]  # (L_in, N, C_static)

        # ── back-filled causal NR (context-dependent) ─────────────────────
        # nr_backfill[i, n] = 1 iff full_nr[t, n]==1 AND obs_time[t, n] <= T_last
        nr_slice = self.nd.full_nr[start:end_in]  # (L_in, N)
        obs_slice = self.nd.observation_time[start:end_in]  # (L_in, N) float64
        nr_backfill = ((nr_slice == 1) & (obs_slice <= T_last)).astype(np.float32)

        # ── assemble full input tensor ────────────────────────────────────
        if self.fb.has_nr_causal:
            x = np.concatenate(
                [x_static, nr_backfill[:, :, np.newaxis]], axis=2
            )  # (L_in, N, C_static + 1)
        else:
            x = x_static  # (L_in, N, C_static)

        # ── targets ───────────────────────────────────────────────────────
        y_orig = self.nd.speed_raw[end_in:end_out]  # (L_out, N) mph
        y_norm = self.nd.speed_norm[end_in:end_out]  # (L_out, N) normalised

        # ── NR evaluation arrays ──────────────────────────────────────────
        full_nr_target = self.nd.full_nr[end_in:end_out]  # (L_out, N)
        causal_fixed_end = self.nd.causal_fixed[T_last]  # (N,)

        regime = classify_regimes(causal_fixed_end, full_nr_target)  # (L_out, N)

        if _TORCH_AVAILABLE:
            return {
                "x": torch.from_numpy(x).float(),
                "y": torch.from_numpy(y_norm).float(),
                "y_orig": torch.from_numpy(y_orig).float(),
                "full_nr": torch.from_numpy(full_nr_target).float(),
                "causal_fixed_end": torch.from_numpy(causal_fixed_end).float(),
                "nr_backfill": torch.from_numpy(nr_backfill).float(),
                "regime": torch.from_numpy(regime).to(torch.int8),
                "sample_idx": start,
            }
        else:
            return {
                "x": x.astype(np.float32),
                "y": y_norm.astype(np.float32),
                "y_orig": y_orig.astype(np.float32),
                "full_nr": full_nr_target.astype(np.float32),
                "causal_fixed_end": causal_fixed_end.astype(np.float32),
                "nr_backfill": nr_backfill.astype(np.float32),
                "regime": regime.astype(np.int8),
                "sample_idx": start,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────


def make_datasets(
    nd: NetworkData,
    feature_config: list[str],
    input_len: int = 9,
    output_len: int = 6,
) -> tuple[TrafficDataset, TrafficDataset, TrafficDataset, FeatureBuilder]:
    """
    Build train / val / test datasets from a single NetworkData object.

    Returns (train_ds, val_ds, test_ds, feature_builder).
    """
    fb = FeatureBuilder(nd, feature_config)
    train_ds = TrafficDataset(nd, fb, "train", input_len, output_len)
    val_ds = TrafficDataset(nd, fb, "val", input_len, output_len)
    test_ds = TrafficDataset(nd, fb, "test", input_len, output_len)

    print(
        f"[Dataset] {nd.network.upper()} / {feature_config} / "
        f"C={fb.C}  |  "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, fb
