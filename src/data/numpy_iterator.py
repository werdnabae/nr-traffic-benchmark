"""
NumpySlidingWindowIterator: pure-numpy equivalent of the PyTorch TrafficDataset.

Generates batches as plain dicts of numpy arrays — no torch dependency.
Used for running statistical / ML baselines locally without PyTorch.

Each batch dict matches the TrafficDataset schema exactly so that all model
predict() methods work unchanged (they accept either torch.Tensors or numpy
arrays via the `_to_numpy` helper in each model).
"""

from __future__ import annotations

import numpy as np

from src.data.loader import NetworkData

# Regime codes — duplicated here to avoid importing torch-dependent dataset.py
REGIME_RECURRENT = 0
REGIME_UNOBSERVED_ONSET = 1
REGIME_CONFIRMED_NR = 2


def classify_regimes(
    causal_fixed_end: np.ndarray,  # (N,)
    full_nr_target: np.ndarray,  # (output_len, N)
) -> np.ndarray:
    """Per-(step, link) regime classification (no torch dependency)."""
    confirmed = (causal_fixed_end > 0).astype(bool)
    confirmed_bc = confirmed[np.newaxis, :]
    is_nr = full_nr_target > 0
    regime = np.zeros_like(full_nr_target, dtype=np.int8)
    regime[is_nr & ~confirmed_bc] = REGIME_UNOBSERVED_ONSET
    regime[is_nr & confirmed_bc] = REGIME_CONFIRMED_NR
    return regime


class NumpySlidingWindowIterator:
    """
    Session-aware sliding-window batch iterator (no torch dependency).

    Parameters
    ----------
    nd             : NetworkData
    feature_arrays : (T, N, C) numpy array of input features
    split          : "train" | "val" | "test"
    input_len      : lookback window
    output_len     : forecast horizon
    batch_size     : samples per batch
    has_nr_causal  : whether to append back-filled causal NR as last channel
    """

    def __init__(
        self,
        nd: NetworkData,
        feature_arrays: np.ndarray,  # (T, N, C_static)
        split: str = "train",
        input_len: int = 9,
        output_len: int = 6,
        batch_size: int = 256,
        has_nr_causal: bool = False,
    ) -> None:
        self.nd = nd
        self.features = feature_arrays
        self.split = split
        self.L_in = input_len
        self.L_out = output_len
        self.batch_size = batch_size
        self.has_nr_causal = has_nr_causal

        self._valid_starts = self._find_valid_starts()

    # ─────────────────────────────────────────────────────────────────────────

    def _find_valid_starts(self) -> list[int]:
        T = self.nd.T
        L = self.L_in + self.L_out
        mask = self.nd.split_mask[self.split]
        sids = self.nd.session_ids

        split_indices = np.where(mask)[0]
        if len(split_indices) < L:
            return []

        first = int(split_indices[0])
        last = int(split_indices[-1])

        valid = []
        for s in range(first, last - L + 2):
            if not mask[s] or not mask[s + L - 1]:
                continue
            if sids[s] == sids[s + L - 1]:
                valid.append(s)
        return valid

    def __len__(self) -> int:
        return len(self._valid_starts)

    def __iter__(self):
        """Yield batches as dicts of numpy arrays."""
        starts = self._valid_starts
        for i in range(0, len(starts), self.batch_size):
            batch_starts = starts[i : i + self.batch_size]
            yield self._make_batch(batch_starts)

    # ─────────────────────────────────────────────────────────────────────────

    def _make_batch(self, batch_starts: list[int]) -> dict:
        B = len(batch_starts)
        L_in, L_out = self.L_in, self.L_out
        nd = self.nd
        C_static = self.features.shape[2]
        C_total = C_static + (1 if self.has_nr_causal else 0)

        x = np.zeros((B, L_in, nd.N, C_total), dtype=np.float32)
        y = np.zeros((B, L_out, nd.N), dtype=np.float32)
        y_orig = np.zeros((B, L_out, nd.N), dtype=np.float32)
        full_nr = np.zeros((B, L_out, nd.N), dtype=np.float32)
        causal_fixed_end = np.zeros((B, nd.N), dtype=np.float32)
        nr_backfill = np.zeros((B, L_in, nd.N), dtype=np.float32)
        regime = np.zeros((B, L_out, nd.N), dtype=np.int8)
        sample_indices = np.array(batch_starts, dtype=np.int64)

        for b, start in enumerate(batch_starts):
            end_in = start + L_in
            end_out = end_in + L_out
            T_last = end_in - 1

            # Static features
            x_static = self.features[start:end_in]  # (L_in, N, C_static)

            # Back-filled causal NR (context-dependent)
            nr_slice = nd.full_nr[start:end_in]
            obs_slice = nd.observation_time[start:end_in]
            backfill = ((nr_slice == 1) & (obs_slice <= T_last)).astype(np.float32)
            nr_backfill[b] = backfill

            if self.has_nr_causal:
                x[b] = np.concatenate([x_static, backfill[:, :, np.newaxis]], axis=2)
            else:
                x[b] = x_static

            y[b] = nd.speed_norm[end_in:end_out]
            y_orig[b] = nd.speed_raw[end_in:end_out]

            full_nr_t = nd.full_nr[end_in:end_out]
            causal_end = nd.causal_fixed[T_last]
            full_nr[b] = full_nr_t
            causal_fixed_end[b] = causal_end
            regime[b] = classify_regimes(causal_end, full_nr_t)

        return {
            "x": x,
            "y": y,
            "y_orig": y_orig,
            "full_nr": full_nr,
            "causal_fixed_end": causal_fixed_end,
            "nr_backfill": nr_backfill,
            "regime": regime,
            "sample_idx": sample_indices,
        }


def make_numpy_iterators(
    nd: NetworkData,
    feature_config: list[str],
    input_len: int = 9,
    output_len: int = 6,
    batch_size: int = 256,
) -> tuple[
    "NumpySlidingWindowIterator",
    "NumpySlidingWindowIterator",
    "NumpySlidingWindowIterator",
    object,  # FeatureBuilder — imported inside function
]:
    """Build train/val/test iterators and the static feature array."""
    from src.data.features import FeatureBuilder

    fb = FeatureBuilder(nd, feature_config)

    has_nr = fb.has_nr_causal

    train_it = NumpySlidingWindowIterator(
        nd, fb.features, "train", input_len, output_len, batch_size, has_nr
    )
    val_it = NumpySlidingWindowIterator(
        nd, fb.features, "val", input_len, output_len, batch_size, has_nr
    )
    test_it = NumpySlidingWindowIterator(
        nd, fb.features, "test", input_len, output_len, batch_size, has_nr
    )

    print(
        f"[NumpyIterator] {nd.network.upper()} / {feature_config} / C={fb.C}  "
        f"train={len(train_it)}  val={len(val_it)}  test={len(test_it)}"
    )
    return train_it, val_it, test_it, fb
