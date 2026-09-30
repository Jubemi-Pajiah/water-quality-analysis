import numpy as np
import pandas as pd
import pytest

from wqtools import stats
from wqtools.stats import CLUSTER_METALS, METALS


def test_correlation_drops_constant_column(groundwater):
    corr = stats.correlation_matrix(groundwater, METALS)
    assert corr.dropped == ["Cd"]
    assert "Cd" not in corr.r.columns
    assert not corr.r.isna().any().any()
    assert np.allclose(corr.r, corr.r.T)
    assert np.allclose(np.diag(corr.r), 1)
    assert ((corr.p >= 0) & (corr.p <= 1)).all().all()


def test_correlation_matches_pandas(groundwater):
    corr = stats.correlation_matrix(groundwater, CLUSTER_METALS)
    expected = groundwater[CLUSTER_METALS].corr()
    assert np.allclose(corr.r, expected)


def test_elbow_wcss_non_increasing(groundwater, storage_water):
    for df in (groundwater, storage_water):
        for scale in (True, False):
            wcss = stats.elbow_wcss(df, scale=scale)
            assert list(wcss.index) == list(range(1, 11))
            assert (np.diff(wcss.to_numpy()) <= 1e-9).all()


def test_silhouette_in_range(groundwater):
    scores = stats.silhouette_by_k(groundwater)
    assert list(scores.index) == list(range(2, 11))
    assert ((scores >= -1) & (scores <= 1)).all()


def test_elbow_k_on_clear_elbow():
    wcss = pd.Series([100, 20, 15, 12, 10, 9], index=range(1, 7))
    assert stats.elbow_k(wcss) == 2


@pytest.mark.parametrize("scale", [True, False])
def test_kmeans_repeatable_and_ordered(storage_water, scale):
    first = stats.kmeans_clusters(storage_water, 3, scale=scale)
    second = stats.kmeans_clusters(storage_water, 3, scale=scale)
    assert first.equals(second)
    fe_means = storage_water["Fe"].groupby(first).mean()
    assert fe_means.idxmin() == 0
    assert fe_means.is_monotonic_increasing


def test_kmeans_rejects_missing_values(groundwater):
    df = groundwater.copy()
    df.loc[0, "Fe"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        stats.kmeans_clusters(df)


def test_cluster_summary_counts(groundwater):
    labels = stats.kmeans_clusters(groundwater, 2)
    summary = stats.cluster_summary(groundwater, labels)
    assert summary["n_samples"].sum() == len(groundwater)


def test_factor_functions_drop_constant(groundwater):
    suitability = stats.factor_suitability(groundwater, METALS)
    assert suitability.dropped == ["Cd"]
    assert "Cd" not in suitability.kmo_per_variable.index
    assert 0 <= suitability.kmo_overall <= 1
    assert 0 <= suitability.bartlett_p <= 1

    eigenvalues, dropped = stats.factor_eigenvalues(groundwater, METALS)
    assert dropped == ["Cd"]
    assert len(eigenvalues) == len(METALS) - 1
    # Eigenvalues of a correlation matrix sum to the number of variables.
    assert eigenvalues.sum() == pytest.approx(len(METALS) - 1)
    assert eigenvalues.is_monotonic_decreasing

    factors = stats.factor_loadings(groundwater, METALS)
    assert factors.dropped == ["Cd"]
    assert "Cd" not in factors.loadings.index


def test_factor_loadings_shape_and_sign(storage_water):
    factors = stats.factor_loadings(storage_water, CLUSTER_METALS)
    assert factors.loadings.shape == (len(CLUSTER_METALS), 2)
    assert factors.scores.shape == (len(storage_water), 2)
    assert list(factors.scores.index) == storage_water["Sample ID"].tolist()
    # Communality is the row sum of squared loadings.
    assert np.allclose(factors.communalities, (factors.loadings ** 2).sum(axis=1))
    # Sign convention: the largest absolute loading of each factor is positive.
    for column in factors.loadings:
        col = factors.loadings[column]
        assert col[col.abs().idxmax()] > 0
    assert factors.variance["Cumulative variance"].iloc[-1] <= 1
