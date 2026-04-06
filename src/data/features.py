"""
FeatureBuilder: assembles the (T, N, C) input feature tensor.

Handles all TIME-INDEPENDENT features (speed, weather, incidents, time).
The window-dependent back-filled causal NR label (nr_causal) is NOT
included here; the Dataset handles that per-sample in __getitem__.

Feature config strings
──────────────────────
"speed"              normalised per-link speed         → C += 1
"weather"            normalised weather (broadcast)    → C += W
"incidents"          binary Waze incident indicator    → C += 1
"time"               raw time-of-day and day-of-week   → C += 2
                     channel 0: tod = (h*60+m)/1440 ∈ [0,1]
                     channel 1: dow = dayofweek/7    ∈ [0,1]
                     Works natively with DGCRN and D2STGNN which read
                     these channels as position encodings.
"nr_causal"          flag: Dataset will append the     → C += 1  (count only)
                     back-filled causal label at
                     runtime (not stored here)

Usage
─────
    from src.data.loader import load_network
    from src.data.features import FeatureBuilder

    nd = load_network("tsmo")
    fb = FeatureBuilder(nd, ["speed", "weather"])
    # fb.features : (T, N, C) float32
    # fb.C        : total feature dimension
    # fb.has_nr_causal : True if "nr_causal" in config
"""

from __future__ import annotations

import numpy as np
from src.data.loader import NetworkData


class FeatureBuilder:
    """
    Build the static (T, N, C) feature tensor from a NetworkData object.

    Parameters
    ----------
    nd             : NetworkData instance
    feature_config : list of feature keys, e.g. ["speed", "weather"]
    """

    VALID_KEYS = {"speed", "weather", "incidents", "time", "nr_causal"}

    def __init__(
        self,
        nd: NetworkData,
        feature_config: list[str],
    ) -> None:
        unknown = set(feature_config) - self.VALID_KEYS
        if unknown:
            raise ValueError(
                f"Unknown feature keys: {unknown}. Valid keys: {self.VALID_KEYS}"
            )

        self.nd = nd
        self.config = list(feature_config)
        self.has_nr_causal = "nr_causal" in feature_config

        self._build()

    # ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        T, N = self.nd.T, self.nd.N
        parts: list[np.ndarray] = []

        if "speed" in self.config:
            # (T, N) → (T, N, 1)
            parts.append(self.nd.speed_norm[:, :, np.newaxis])

        if "time" in self.config:
            # Raw time-of-day and day-of-week, broadcast over N links.
            # Placed IMMEDIATELY after speed so DGCRN (reads channels 1,2 as
            # TOD/DOW) and D2STGNN (reads channels num_feat, num_feat+1) work
            # correctly regardless of what other features follow.
            # tod ∈ [0,1]: fraction of day elapsed at each 5-min step.
            # dow ∈ [0,1]: dayofweek / 7.
            import pandas as pd

            ts = pd.DatetimeIndex(self.nd.timestamps)
            tod = ((ts.hour * 60 + ts.minute) / 1440.0).to_numpy().astype(np.float32)
            dow = (ts.dayofweek / 7.0).to_numpy().astype(np.float32)
            time_arr = np.stack([tod, dow], axis=1)  # (T, 2)
            time_expanded = np.tile(time_arr[:, np.newaxis, :], (1, N, 1))  # (T, N, 2)
            parts.append(time_expanded)

        if "weather" in self.config:
            if self.nd.W == 0:
                raise ValueError(
                    f"'weather' requested but no weather data available for "
                    f"{self.nd.network}."
                )
            # Broadcast (T, W) to (T, N, W)
            weather_expanded = np.tile(self.nd.weather[:, np.newaxis, :], (1, N, 1))
            parts.append(weather_expanded)

        if "incidents" in self.config:
            parts.append(self.nd.incidents[:, :, np.newaxis])

        if not parts:
            raise ValueError(
                "Feature config results in zero static features. "
                "At least one of {speed, weather, incidents} is required."
            )

        self.features: np.ndarray = np.concatenate(parts, axis=2).astype(
            np.float32
        )  # (T, N, C_static)

        # Total C includes the runtime nr_causal slot if requested
        self._C_static = self.features.shape[2]
        self._C_total = self._C_static + (1 if self.has_nr_causal else 0)

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def C(self) -> int:
        """Total input feature dimension including nr_causal (if requested)."""
        return self._C_total

    @property
    def C_static(self) -> int:
        """Feature dimension of the static (pre-computed) part."""
        return self._C_static

    def __repr__(self) -> str:
        return (
            f"FeatureBuilder(network={self.nd.network}, "
            f"config={self.config}, C={self.C}, "
            f"has_nr_causal={self.has_nr_causal})"
        )
