import numpy as np
import pandas as pd
import pytest

from wqtools.indices import wpi, wqi, wqi_class

from .conftest import samples_table, standards_table

# Limits in these tests are chosen to make the arithmetic easy to check by hand.
# They are not taken from any drinking water standard.


def test_wqi_single_parameter_at_limit_is_100():
    result = wqi(samples_table(Fe=[0.3]), standards_table([("Fe", 0.3, 0)]), details=True)
    assert result.loc[0, "Fe_Qi"] == pytest.approx(100)
    assert result.loc[0, "WQI"] == pytest.approx(100)


def test_wqi_two_parameters_weighted_by_inverse_limit():
    samples = samples_table(Fe=[0.15], Zn=[3.0])
    standards = standards_table([("Fe", 0.3, 0), ("Zn", 3, 0)])
    result = wqi(samples, standards, details=True)
    assert result.loc[0, "Fe_Qi"] == pytest.approx(50)
    assert result.loc[0, "Zn_Qi"] == pytest.approx(100)
    # (50 * 1/0.3 + 100 * 1/3) / (1/0.3 + 1/3)
    assert result.loc[0, "WQI"] == pytest.approx(54.545, abs=5e-4)


@pytest.mark.parametrize("ph, expected", [(7.0, 0), (8.5, 100), (6.5, 100), (7.75, 50), (6.75, 50)])
def test_wqi_ph_is_two_sided(ph, expected):
    # Regression for the original script, whose pH branch never ran because it
    # compared a lowercased name with "pH". pH 6.5 then fell through to the
    # generic formula and came out at (6.5 - 7) / (8.5 - 7) * 100 = -33.3.
    result = wqi(samples_table(pH=[ph]), standards_table([("pH", 8.5, 7)]), details=True)
    assert result.loc[0, "pH_Qi"] == pytest.approx(expected)


def test_wqi_ph_name_is_case_insensitive():
    result = wqi(samples_table(PH=[6.5]), standards_table([("PH", 8.5, 7)]), details=True)
    assert result.loc[0, "PH_Qi"] == pytest.approx(100)


def test_wqi_skips_missing_value_and_renormalises():
    samples = samples_table(Fe=[0.15, 0.15], Zn=[3.0, np.nan])
    standards = standards_table([("Fe", 0.3, 0), ("Zn", 3, 0)])
    result = wqi(samples, standards)
    assert result.loc[0, "WQI"] == pytest.approx(54.545, abs=5e-4)
    # The second sample has only Fe, so its WQI is Fe's rating alone.
    assert result.loc[1, "WQI"] == pytest.approx(50)


def test_wqi_all_missing_gives_nan():
    result = wqi(samples_table(Fe=[np.nan]), standards_table([("Fe", 0.3, 0)]))
    assert np.isnan(result.loc[0, "WQI"])


def test_wqi_ignores_standards_without_data_column():
    samples = samples_table(Fe=[0.3])
    standards = standards_table([("Fe", 0.3, 0), ("Zn", 3, 0)])
    assert wqi(samples, standards).loc[0, "WQI"] == pytest.approx(100)


@pytest.mark.parametrize("value, label", [(0, "Excellent"), (49.9, "Excellent"), (50, "Good"),
                                          (99.9, "Good"), (100, "Poor"), (200, "Very poor"),
                                          (300, "Unsuitable"), (1000, "Unsuitable")])
def test_wqi_class_bands(value, label):
    assert wqi_class(value) == label


def test_wqi_class_series_and_nan():
    labels = wqi_class(pd.Series([10, np.nan, 150]))
    assert labels[0] == "Excellent"
    assert pd.isna(labels[1])
    assert labels[2] == "Poor"


def test_wpi_metal_load():
    result = wpi(samples_table(Fe=[0.6]), standards_table([("Fe", 0.3, 0)]), details=True)
    assert result.loc[0, "Fe_PL"] == pytest.approx(2)
    assert result.loc[0, "WPI"] == pytest.approx(2)
    assert result.loc[0, "n_parameters"] == 1


def test_wpi_excludes_zero_from_mean():
    samples = samples_table(Fe=[0.6], Pb=[0.0])
    result = wpi(samples, standards_table([("Fe", 0.3, 0), ("Pb", 0.01, 0)]))
    assert result.loc[0, "WPI"] == pytest.approx(2)
    assert result.loc[0, "n_parameters"] == 1


@pytest.mark.parametrize("ph, expected", [(6.0, 2.0), (8.5, 1.0), (7.0, 0.0)])
def test_wpi_ph_branch_runs(ph, expected):
    # Regression for the original script, which never put pH in its parameter
    # list, so the pH formula could not run.
    result = wpi(samples_table(pH=[ph]), standards_table([("pH", 8.5, 7)]), details=True)
    assert result.loc[0, "pH_PL"] == pytest.approx(expected)
    assert result.loc[0, "n_parameters"] == 1


def test_wpi_mixes_ph_and_metals():
    samples = samples_table(Fe=[0.6], pH=[6.0])
    result = wpi(samples, standards_table([("Fe", 0.3, 0), ("pH", 8.5, 7)]))
    assert result.loc[0, "WPI"] == pytest.approx(2.0)
    assert result.loc[0, "n_parameters"] == 2


def test_wqi_and_wpi_share_one_standards_table():
    standards = standards_table([("Fe", 0.3, 0), ("Zn", 3, 0), ("pH", 8.5, 7)])
    samples = samples_table(Fe=[0.3, 0.15], Zn=[3.0, 1.5], pH=[7.0, 8.5])
    assert wqi(samples, standards)["WQI"].notna().all()
    assert wpi(samples, standards)["WPI"].notna().all()


def test_no_shared_parameter_raises():
    with pytest.raises(ValueError, match="No parameter"):
        wqi(samples_table(Fe=[0.3]), standards_table([("Zn", 3, 0)]))
