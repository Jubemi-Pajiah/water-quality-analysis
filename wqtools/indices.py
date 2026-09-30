"""Water Quality Index (WQI) and Water Pollution Index (WPI).

Both functions take a sample table (``Sample ID`` plus one column per
parameter) and a standards table as returned by :func:`wqtools.io.load_standards`.
Only parameters present in both tables are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import IDEAL, LIMIT, PARAMETER

PH_NEUTRAL = 7.0
PH_LOW = 6.5
PH_HIGH = 8.5

WQI_BANDS = [
    (50, "Excellent"),
    (100, "Good"),
    (200, "Poor"),
    (300, "Very poor"),
    (np.inf, "Unsuitable"),
]


def _is_ph(name: str) -> bool:
    return name.strip().lower() == "ph"


def _shared_parameters(samples: pd.DataFrame, standards: pd.DataFrame) -> pd.DataFrame:
    used = standards[standards[PARAMETER].isin(samples.columns)]
    if used.empty:
        raise ValueError("No parameter in the standards table matches a column in the sample table")
    return used.set_index(PARAMETER)


def _ph_rating(ph: pd.Series) -> pd.Series:
    """Distance from neutral as a fraction of the allowed range on that side."""
    above = (ph - PH_NEUTRAL) / (PH_HIGH - PH_NEUTRAL)
    below = (PH_NEUTRAL - ph) / (PH_NEUTRAL - PH_LOW)
    return above.where(ph >= PH_NEUTRAL, below)


def wqi(samples: pd.DataFrame, standards: pd.DataFrame, details: bool = False) -> pd.DataFrame:
    """Weighted arithmetic Water Quality Index.

    Wi = K / Si with K = 1 / sum(1 / Si).
    Qi = 100 * (Va - Vi) / (Si - Vi), where Va is the measured value and Vi the
    ideal value. For pH, Qi = 100 * (pH - 7) / (8.5 - 7) at or above 7 and
    100 * (7 - pH) / (7 - 6.5) below 7.
    WQI = sum(Qi * Wi) / sum(Wi), taken per sample over the parameters that are
    not NaN, so the weights renormalise row by row.

    Returns a copy of ``samples`` with a ``WQI`` column. With ``details=True``
    it also has ``<param>_Qi`` and ``<param>_Ii`` (= Qi * Wi) columns.
    """
    used = _shared_parameters(samples, standards)
    si = used[LIMIT]
    weights = (1.0 / si) / (1.0 / si).sum()

    ratings = {}
    for param, row in used.iterrows():
        values = samples[param]
        if _is_ph(param):
            ratings[param] = 100.0 * _ph_rating(values)
        else:
            if row[LIMIT] == row[IDEAL]:
                raise ValueError(f"{param}: permissible limit equals ideal value")
            ratings[param] = 100.0 * (values - row[IDEAL]) / (row[LIMIT] - row[IDEAL])
    q = pd.DataFrame(ratings, index=samples.index)

    sub_index = q * weights
    weight_sum = q.notna().mul(weights).sum(axis=1)
    index = sub_index.sum(axis=1, min_count=1) / weight_sum.replace(0.0, np.nan)

    out = samples.copy()
    if details:
        for param in q.columns:
            out[f"{param}_Qi"] = q[param]
            out[f"{param}_Ii"] = sub_index[param]
    out["WQI"] = index
    return out


def wqi_class(value):
    """Conventional WQI bands. Accepts a number or a Series."""
    if isinstance(value, pd.Series):
        return value.map(wqi_class)
    if value is None or pd.isna(value):
        return None
    for upper, label in WQI_BANDS:
        if value < upper:
            return label
    return WQI_BANDS[-1][1]


def wpi(samples: pd.DataFrame, standards: pd.DataFrame, details: bool = False) -> pd.DataFrame:
    """Water Pollution Index.

    For each parameter, the pollution load is PL = 1 + (C - Si) / Si, where C is
    the measured value. For pH, PL = (7 - pH) / (7 - 6.5) below 7 and
    (pH - 7) / (8.5 - 7) above 7. WPI is the mean PL over the parameters of a
    sample. A value of exactly 0 is treated as not detected and left out of the
    mean, as are NaN values.

    Returns a copy of ``samples`` with ``WPI`` and ``n_parameters`` columns, and
    ``<param>_PL`` columns when ``details=True``.
    """
    used = _shared_parameters(samples, standards)

    loads = {}
    for param, row in used.iterrows():
        values = samples[param].where(samples[param] != 0)
        if _is_ph(param):
            loads[param] = _ph_rating(values)
        else:
            loads[param] = 1.0 + (values - row[LIMIT]) / row[LIMIT]
    pl = pd.DataFrame(loads, index=samples.index)

    out = samples.copy()
    if details:
        for param in pl.columns:
            out[f"{param}_PL"] = pl[param]
    out["WPI"] = pl.mean(axis=1, skipna=True)
    out["n_parameters"] = pl.notna().sum(axis=1)
    return out
