"""Loading and checking sample and standards tables."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

ID_COLUMN = "Sample ID"
PARAMETER = "Parameter"
LIMIT = "Permissible Limit (Si)"
IDEAL = "Ideal Value (Ci)"
SOURCE = "Source"

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls"}


def _read_table(path, sheet=None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if sheet is not None:
            raise ValueError(f"{path.name} is a CSV file; --sheet only applies to Excel files")
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str,
                             keep_default_na=False)
    raise ValueError(f"Unsupported file type {suffix!r}; use .csv or .xlsx")


def _to_numeric(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a column of strings. Returns the numbers and a mask of cells that failed."""
    text = raw.astype(str).str.strip()
    blank = text.eq("") | text.str.lower().isin({"nan", "none", "na", "n/a"})
    values = pd.to_numeric(text.where(~blank), errors="coerce")
    failed = values.isna() & ~blank
    return values, failed


def load_samples(path, sheet=None) -> pd.DataFrame:
    """Load a sample table from CSV or Excel.

    The table needs a ``Sample ID`` column. Every other column is parsed as a
    number. Blank cells become NaN silently. Cells that are not blank but cannot
    be parsed also become NaN, and all of them are reported in one warning that
    names the sample and the column.
    """
    raw = _read_table(path, sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    if ID_COLUMN not in raw.columns:
        raise ValueError(f"{Path(path).name}: missing required column {ID_COLUMN!r}")

    df = pd.DataFrame({ID_COLUMN: raw[ID_COLUMN].astype(str).str.strip()})
    problems = []
    for column in raw.columns:
        if column == ID_COLUMN:
            continue
        values, failed = _to_numeric(raw[column])
        df[column] = values.astype(float)
        for idx in failed[failed].index:
            problems.append(f"{df.at[idx, ID_COLUMN]} / {column}: {raw.at[idx, column]!r}")

    if problems:
        warnings.warn(
            f"{Path(path).name}: {len(problems)} cell(s) could not be read as numbers and were "
            "set to NaN (sample / column: value): " + "; ".join(problems),
            stacklevel=2,
        )
    return df


def load_standards(path, sheet=None) -> pd.DataFrame:
    """Load a standards table.

    Required columns: ``Parameter`` and ``Permissible Limit (Si)``.
    ``Ideal Value (Ci)`` is optional and defaults to 0. Rows with a blank limit
    are left out (with a warning), so a parameter with no limit in the chosen
    standard is simply not used by the indices. Non-positive limits are rejected.
    """
    raw = _read_table(path, sheet)
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in (PARAMETER, LIMIT) if c not in raw.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: missing required column(s) {missing}")

    table = pd.DataFrame({PARAMETER: raw[PARAMETER].astype(str).str.strip()})
    limit, bad_limit = _to_numeric(raw[LIMIT])
    if IDEAL in raw.columns:
        ideal, bad_ideal = _to_numeric(raw[IDEAL])
    else:
        ideal, bad_ideal = pd.Series(0.0, index=raw.index), pd.Series(False, index=raw.index)

    bad = table.loc[bad_limit | bad_ideal, PARAMETER].tolist()
    if bad:
        raise ValueError(f"{Path(path).name}: non-numeric limit or ideal value for {bad}")

    table[LIMIT] = limit.astype(float)
    table[IDEAL] = ideal.fillna(0.0).astype(float)
    table[SOURCE] = raw[SOURCE].astype(str).str.strip() if SOURCE in raw.columns else ""

    no_limit = table[LIMIT].isna()
    if no_limit.any():
        warnings.warn(
            f"{Path(path).name}: no permissible limit for {table.loc[no_limit, PARAMETER].tolist()}; "
            "these parameters are left out of WQI and WPI",
            stacklevel=2,
        )
        table = table.loc[~no_limit]

    non_positive = table.loc[table[LIMIT] <= 0, PARAMETER].tolist()
    if non_positive:
        raise ValueError(f"{Path(path).name}: permissible limit must be positive for {non_positive}")

    duplicated = table.loc[table[PARAMETER].duplicated(), PARAMETER].tolist()
    if duplicated:
        raise ValueError(f"{Path(path).name}: parameter listed more than once: {duplicated}")

    return table.reset_index(drop=True)


def value_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the sample ID."""
    return [c for c in df.columns if c != ID_COLUMN]


def drop_constant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Split off columns that take a single value (ignoring NaN).

    Returns the frame without those columns and the list of dropped names. The
    ``Sample ID`` column is kept if present.
    """
    constant = [c for c in value_columns(df) if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=constant), constant
