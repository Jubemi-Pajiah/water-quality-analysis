import warnings

import pandas as pd
import pytest

from wqtools.io import IDEAL, LIMIT, drop_constant, load_samples, load_standards
from wqtools.stats import METALS

from .conftest import DATA


@pytest.mark.parametrize("name, prefix", [("groundwater_metals.csv", "US"),
                                          ("storagewater_metals.csv", "LS")])
def test_bundled_data_loads(name, prefix):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = load_samples(DATA / name)
    assert len(df) == 19
    assert list(df.columns) == ["Sample ID", *METALS]
    assert df["Sample ID"].tolist() == [f"{prefix}{i}" for i in range(1, 20)]
    for metal in METALS:
        assert pd.api.types.is_float_dtype(df[metal])
    assert not df[METALS].isna().any().any()


def test_bad_cell_is_reported(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("Sample ID,Fe,Zn\nA,0.1,0.2\nB,abc,0.3\nC,0.2,\n")
    with pytest.warns(UserWarning, match=r"B / Fe: 'abc'") as record:
        df = load_samples(path)
    assert len(record) == 1
    assert pd.isna(df.loc[1, "Fe"])
    # A blank cell is missing data, not an error.
    assert "C / Zn" not in str(record[0].message)
    assert pd.isna(df.loc[2, "Zn"])


def test_missing_sample_id_column(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("ID,Fe\nA,0.1\n")
    with pytest.raises(ValueError, match="Sample ID"):
        load_samples(path)


def test_excel_sheet(tmp_path):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="Other", index=False)
        pd.DataFrame({"Sample ID": ["A", "B"], "Fe": [0.1, 0.2]}).to_excel(
            writer, sheet_name="GroundWater", index=False)
    df = load_samples(path, sheet="GroundWater")
    assert df["Fe"].tolist() == [0.1, 0.2]


def test_standards_ideal_defaults_to_zero(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text("Parameter,Permissible Limit (Si)\nFe,0.3\n")
    table = load_standards(path)
    assert table.loc[0, LIMIT] == 0.3
    assert table.loc[0, IDEAL] == 0


def test_standards_blank_limit_is_left_out(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text("Parameter,Permissible Limit (Si),Ideal Value (Ci)\nFe,0.3,0\nCo,,0\n")
    with pytest.warns(UserWarning, match="Co"):
        table = load_standards(path)
    assert table["Parameter"].tolist() == ["Fe"]


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_standards_reject_non_positive_limit(tmp_path, limit):
    path = tmp_path / "standards.csv"
    path.write_text(f"Parameter,Permissible Limit (Si)\nFe,{limit}\n")
    with pytest.raises(ValueError, match="positive"):
        load_standards(path)


def test_standards_require_columns(tmp_path):
    path = tmp_path / "standards.csv"
    path.write_text("Parameter,Standard Limit (Si)\nFe,0.3\n")
    with pytest.raises(ValueError, match="Permissible Limit"):
        load_standards(path)


def test_drop_constant(groundwater):
    varying, dropped = drop_constant(groundwater)
    assert dropped == ["Cd"]
    assert "Cd" not in varying.columns
    assert "Sample ID" in varying.columns
