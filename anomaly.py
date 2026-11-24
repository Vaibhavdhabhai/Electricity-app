"""Peak and anomaly detection on the daily demand series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect(
    series: pd.Series,
    z_thr: float = 3.5,
    peak_quantile: float = 0.95,
    contamination: float = 0.03,
) -> pd.DataFrame:
    """Flag unusual days.

    A day is an anomaly if either:
      - its robust z-score against the local weekly level exceeds `z_thr`, or
      - an Isolation Forest flags it as unusual on (demand, deviation from local level).
    A day is a peak if it is in the top `1 - peak_quantile` share of demand.
    """
    s = series.astype(float)
    local_level = s.rolling(7, center=True, min_periods=3).median()
    dev = (s - local_level).fillna(0.0)

    centred = dev - dev.median()
    abs_dev = centred.abs()
    mad = np.median(abs_dev[abs_dev > 1e-6])  # ignore flat stretches
    scale = mad if np.isfinite(mad) and mad > 0 else 1.0
    z = 0.6745 * centred / scale

    iso = IsolationForest(contamination=contamination, random_state=42).fit_predict(
        np.c_[s.values, dev.values]
    ) == -1

    df = pd.DataFrame(
        {"value": s, "z": z, "zscore_flag": z.abs() > z_thr, "iso_flag": iso},
        index=s.index,
    )
    df["anomaly"] = df["zscore_flag"] | df["iso_flag"]
    df["peak"] = s >= s.quantile(peak_quantile)
    return df
