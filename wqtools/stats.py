"""Correlation, K-means clustering and factor analysis."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import NamedTuple

import factor_analyzer as fa
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .io import drop_constant, value_columns

METALS = ["Fe", "Zn", "Cu", "Mn", "Ni", "Cr", "Co", "Pb", "Cd"]
# Pb and Cd are left out of clustering: Cd is never detected and Pb only in a few samples.
CLUSTER_METALS = ["Fe", "Zn", "Cu", "Mn", "Ni", "Cr", "Co"]
RANDOM_STATE = 42
N_INIT = 10


def _columns(df: pd.DataFrame, columns) -> list[str]:
    columns = list(columns) if columns is not None else value_columns(df)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not in the data: {missing}")
    return columns


def _varying(df: pd.DataFrame, columns) -> tuple[pd.DataFrame, list[str]]:
    varying, dropped = drop_constant(df[_columns(df, columns)])
    return varying, dropped


def _matrix(df: pd.DataFrame, columns, scale: bool) -> np.ndarray:
    x = df[_columns(df, columns)]
    if x.isna().any().any():
        raise ValueError("Missing values in the clustering columns; fill or drop them first")
    x = x.to_numpy(dtype=float)
    return StandardScaler().fit_transform(x) if scale else x


# Correlation -----------------------------------------------------------------

class Correlation(NamedTuple):
    r: pd.DataFrame
    p: pd.DataFrame
    dropped: list[str]


_CORR_TESTS = {"pearson": sps.pearsonr, "spearman": sps.spearmanr, "kendall": sps.kendalltau}


def correlation_matrix(df: pd.DataFrame, columns=None, method: str = "pearson") -> Correlation:
    """Correlation coefficients and two-sided p-values.

    Constant columns have no defined correlation, so they are dropped and listed
    in ``dropped``. Pairs are computed on the samples where both values are
    present.
    """
    if method not in _CORR_TESTS:
        raise ValueError(f"method must be one of {sorted(_CORR_TESTS)}")
    x, dropped = _varying(df, columns)
    names = list(x.columns)
    r = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    p = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = x[[a, b]].dropna()
            res = _CORR_TESTS[method](pair[a], pair[b])
            r.loc[a, b] = r.loc[b, a] = res[0]
            p.loc[a, b] = p.loc[b, a] = res[1]
    return Correlation(r, p, dropped)


# K-means ---------------------------------------------------------------------

def _fit_kmeans(x: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(x)


def kmeans_clusters(df: pd.DataFrame, n_clusters: int = 2, columns=CLUSTER_METALS,
                    scale: bool = True, order_by: str = "Fe") -> pd.Series:
    """K-means cluster labels, one per sample.

    Inputs are standardised (zero mean, unit variance per metal) unless
    ``scale=False``. Without scaling, Fe has much larger values than the other
    metals and dominates the distance. Labels are renumbered so that cluster 0
    has the lowest mean ``order_by`` value in the original units, which keeps
    the numbering stable between runs and between scaled and unscaled fits.
    """
    x = _matrix(df, columns, scale)
    raw = _fit_kmeans(x, n_clusters).labels_
    means = df[order_by].groupby(raw).mean().sort_values(kind="stable")
    mapping = {old: new for new, old in enumerate(means.index)}
    return pd.Series([mapping[v] for v in raw], index=df.index, name="cluster")


def cluster_summary(df: pd.DataFrame, labels: pd.Series, columns=CLUSTER_METALS) -> pd.DataFrame:
    """Sample count and mean concentration of each metal per cluster."""
    columns = _columns(df, columns)
    table = df[columns].groupby(labels).mean()
    table.insert(0, "n_samples", labels.value_counts().sort_index())
    table.index.name = "cluster"
    return table


def elbow_wcss(df: pd.DataFrame, columns=CLUSTER_METALS, scale: bool = True,
               k_values=range(1, 11)) -> pd.Series:
    """Within-cluster sum of squares (K-means inertia) for each k."""
    x = _matrix(df, columns, scale)
    return pd.Series({k: _fit_kmeans(x, k).inertia_ for k in k_values}, name="WCSS").rename_axis("k")


def elbow_k(wcss: pd.Series) -> int:
    """k at the elbow: the point furthest from the straight line joining the ends.

    Both axes are rescaled to [0, 1] first so the result does not depend on units.
    """
    k = np.asarray(wcss.index, dtype=float)
    w = wcss.to_numpy(dtype=float)
    kx = (k - k[0]) / (k[-1] - k[0])
    wy = (w - w[-1]) / (w[0] - w[-1])
    # Distance from each point to the line from (0, 1) to (1, 0).
    distance = np.abs(kx + wy - 1) / np.sqrt(2)
    return int(wcss.index[int(np.argmax(distance))])


def silhouette_by_k(df: pd.DataFrame, columns=CLUSTER_METALS, scale: bool = True,
                    k_values=range(2, 11)) -> pd.Series:
    """Mean silhouette score for each k (computed in the same space as the fit)."""
    x = _matrix(df, columns, scale)
    scores = {k: silhouette_score(x, _fit_kmeans(x, k).labels_) for k in k_values}
    return pd.Series(scores, name="silhouette").rename_axis("k")


# Factor analysis -------------------------------------------------------------

@contextmanager
def _quiet_factor_analyzer():
    # factor_analyzer 0.5.1 passes a keyword that scikit-learn 1.6 and 1.7 have
    # deprecated; silence that one warning so it does not drown real ones.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
        yield


class Suitability(NamedTuple):
    kmo_overall: float
    kmo_per_variable: pd.Series
    bartlett_chi2: float
    bartlett_p: float
    dropped: list[str]

    def summary(self) -> pd.DataFrame:
        rows = [("KMO overall", self.kmo_overall), ("Bartlett chi-square", self.bartlett_chi2),
                ("Bartlett p-value", self.bartlett_p)]
        rows += [(f"KMO {name}", value) for name, value in self.kmo_per_variable.items()]
        rows += [(f"Dropped (constant) {name}", np.nan) for name in self.dropped]
        return pd.DataFrame(rows, columns=["statistic", "value"])


def factor_suitability(df: pd.DataFrame, columns=None) -> Suitability:
    """Kaiser-Meyer-Olkin measure and Bartlett's test of sphericity."""
    x, dropped = _varying(df, columns)
    # Both statistics depend only on the correlation matrix. Standardising first
    # avoids a near-singular covariance matrix when concentrations are small.
    z = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    with _quiet_factor_analyzer():
        kmo_all, kmo_model = fa.calculate_kmo(z)
        chi2, p = fa.calculate_bartlett_sphericity(z)
    return Suitability(float(kmo_model), pd.Series(kmo_all, index=x.columns, name="KMO"),
                       float(chi2), float(p), dropped)


def factor_eigenvalues(df: pd.DataFrame, columns=None) -> tuple[pd.Series, list[str]]:
    """Eigenvalues of the correlation matrix, largest first."""
    x, dropped = _varying(df, columns)
    eig = np.sort(np.linalg.eigvalsh(np.corrcoef(x.to_numpy(dtype=float), rowvar=False)))[::-1]
    return pd.Series(eig, index=range(1, len(eig) + 1), name="eigenvalue").rename_axis("factor"), dropped


class Factors(NamedTuple):
    loadings: pd.DataFrame
    communalities: pd.Series
    variance: pd.DataFrame
    scores: pd.DataFrame
    dropped: list[str]


def factor_loadings(df: pd.DataFrame, columns=None, n_factors: int = 2,
                    rotation: str = "varimax") -> Factors:
    """Factor analysis (minimum residual) on standardised data with rotation.

    The sign of a factor is arbitrary, so each factor is flipped if needed to
    make its largest absolute loading positive. Scores are flipped with it.
    """
    x, dropped = _varying(df, columns)
    z = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    model = fa.FactorAnalyzer(n_factors=n_factors, rotation=rotation)
    with _quiet_factor_analyzer():
        model.fit(z)
        scores = model.transform(z)

    names = [f"Factor {i + 1}" for i in range(n_factors)]
    loadings = model.loadings_.copy()
    signs = np.sign(loadings[np.argmax(np.abs(loadings), axis=0), range(n_factors)])
    signs[signs == 0] = 1
    loadings *= signs
    scores *= signs

    ss, prop, cum = model.get_factor_variance()
    variance = pd.DataFrame({"SS loadings": ss, "Proportion of variance": prop,
                             "Cumulative variance": cum}, index=names)
    index = df["Sample ID"] if "Sample ID" in df.columns else df.index
    return Factors(
        pd.DataFrame(loadings, index=x.columns, columns=names),
        pd.Series(model.get_communalities(), index=x.columns, name="communality"),
        variance,
        pd.DataFrame(scores, index=pd.Index(index, name="Sample ID"), columns=names),
        dropped,
    )
