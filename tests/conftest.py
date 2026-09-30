from pathlib import Path

import pandas as pd
import pytest

from wqtools.io import IDEAL, LIMIT, PARAMETER, load_samples

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def groundwater():
    return load_samples(DATA / "groundwater_metals.csv")


@pytest.fixture
def storage_water():
    return load_samples(DATA / "storagewater_metals.csv")


def standards_table(rows):
    """Build a standards table from (parameter, limit, ideal) tuples. Test values only."""
    return pd.DataFrame(rows, columns=[PARAMETER, LIMIT, IDEAL])


def samples_table(**columns):
    n = len(next(iter(columns.values())))
    return pd.DataFrame({"Sample ID": [f"S{i + 1}" for i in range(n)], **columns})
