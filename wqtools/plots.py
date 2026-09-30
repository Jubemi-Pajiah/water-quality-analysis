"""Figures. Every function returns a matplotlib Figure and never calls plt.show()."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler

from .io import ID_COLUMN

UNIT = "mg/L"

# Okabe-Ito palette, readable with the common forms of colour blindness.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DIVERGING = "RdBu_r"

# Same pairs as the original cluster-analysis script.
SCATTER_PAIRS = [("Fe", "Cu"), ("Cr", "Mn"), ("Fe", "Zn"), ("Co", "Fe"), ("Ni", "Co"), ("Mn", "Zn")]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)


def correlation_heatmap(r: pd.DataFrame, p: pd.DataFrame | None = None, title: str = "",
                        alpha: float = 0.05):
    """Lower triangle of a correlation matrix. An asterisk marks p < alpha."""
    n = len(r)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    values = np.ma.masked_array(r.to_numpy(), mask=mask)

    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.6 + 0.7 * n), layout="constrained")
    image = ax.imshow(values, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            value = r.iat[i, j]
            star = "*" if p is not None and i != j and p.iat[i, j] < alpha else ""
            ax.text(j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else "black")
    ax.set_xticks(range(n), r.columns)
    ax.set_yticks(range(n), r.index)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Correlation coefficient r")
    note = f"* p < {alpha:g}" if p is not None else ""
    ax.set_title(title or "Correlation matrix", loc="left")
    if note:
        ax.set_xlabel(note, loc="right", fontsize=8)
    return fig


def elbow_plot(wcss: pd.Series, chosen_k: int | None = None, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(wcss.index, wcss.values, marker="o", color=PALETTE[0])
    if chosen_k is not None:
        ax.plot([chosen_k], [wcss[chosen_k]], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor=PALETTE[5], markeredgewidth=2, linestyle="none",
                label=f"Elbow at k = {chosen_k}")
        ax.legend(frameon=False)
    ax.set_xticks(list(wcss.index))
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Within-cluster sum of squares")
    ax.set_title(title or "Elbow method", loc="left")
    _style(ax)
    return fig


def silhouette_plot(scores: pd.Series, title: str = ""):
    best = int(scores.idxmax())
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(scores.index, scores.values, marker="o", color=PALETTE[0])
    ax.plot([best], [scores[best]], marker="o", markersize=12, markerfacecolor="none",
            markeredgecolor=PALETTE[5], markeredgewidth=2, linestyle="none",
            label=f"Maximum at k = {best} ({scores[best]:.2f})")
    ax.set_xticks(list(scores.index))
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title(title or "Silhouette score", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    return fig


def cluster_scatter(df: pd.DataFrame, labels: pd.Series, pairs=SCATTER_PAIRS, title: str = "",
                    unit: str = UNIT):
    clusters = sorted(labels.unique())
    ncols = 2
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.3 * nrows), layout="constrained")
    for ax, (x, y) in zip(axes.flat, pairs):
        for c in clusters:
            part = df[labels == c]
            ax.scatter(part[x], part[y], color=PALETTE[c % len(PALETTE)],
                       marker=MARKERS[c % len(MARKERS)], s=40, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(f"{x} ({unit})")
        ax.set_ylabel(f"{y} ({unit})")
        _style(ax)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=PALETTE[c % len(PALETTE)], marker=MARKERS[c % len(MARKERS)],
                      linestyle="none", label=f"Cluster {c} (n = {(labels == c).sum()})")
               for c in clusters]
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(clusters), 4),
               frameon=False)
    fig.suptitle(title or "K-means clusters", x=0.02, ha="left")
    return fig


def scree_plot(eigenvalues: pd.Series, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.plot(eigenvalues.index, eigenvalues.values, marker="o", color=PALETTE[0], label="Eigenvalue")
    ax.axhline(1.0, color=PALETTE[5], linestyle="--", linewidth=1.2, label="Eigenvalue = 1")
    ax.set_xticks(list(eigenvalues.index))
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue of the correlation matrix")
    ax.set_title(title or "Scree plot", loc="left")
    ax.legend(frameon=False)
    _style(ax)
    return fig


def loadings_heatmap(loadings: pd.DataFrame, title: str = ""):
    n, m = loadings.shape
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * m, 0.8 + 0.5 * n), layout="constrained")
    image = ax.imshow(loadings.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(m):
            value = loadings.iat[i, j]
            ax.text(j, i, f"{value + 0.0:.2f}".replace("-0.00", "0.00"), ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.6 else "black")
    ax.set_xticks(range(m), loadings.columns)
    ax.set_yticks(range(n), loadings.index)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(image, ax=ax).set_label("Loading")
    ax.set_title(title or "Rotated factor loadings", loc="left")
    return fig


def biplot(scores: pd.DataFrame, loadings: pd.DataFrame, title: str = ""):
    """Sample scores as points, variable loadings as arrows.

    Arrows are multiplied by one common factor so the longest arrow reaches 80%
    of the largest score. Only their directions and relative lengths matter.
    """
    fx, fy = scores.columns[:2]
    s = scores[[fx, fy]].to_numpy()
    load = loadings[[fx, fy]].to_numpy()
    stretch = 0.8 * np.abs(s).max() / np.abs(load).max()
    tips = load * stretch

    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8)
    ax.scatter(s[:, 0], s[:, 1], color=PALETTE[0], s=30, zorder=3, label="Samples")
    for lx, ly in tips:
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=PALETTE[5], linewidth=1.4))
    ax.plot([], [], color=PALETTE[5], label=f"Loadings (x {stretch:.1f})")
    both = np.vstack([s, tips])
    pad = 0.12 * (both.max(axis=0) - both.min(axis=0))
    ax.set_xlim(both[:, 0].min() - pad[0], both[:, 0].max() + pad[0])
    ax.set_ylim(both[:, 1].min() - pad[1], both[:, 1].max() + pad[1])
    ax.set_xlabel(f"{fx} score")
    ax.set_ylabel(f"{fy} score")
    ax.set_title(title or "Factor biplot", loc="left")
    ax.legend(frameon=False, loc="best")
    _style(ax)

    # Settle the constrained layout, then freeze it so labels keep the pixel
    # positions they were checked against.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    taken = [ax.get_legend().get_window_extent(renderer)]
    taken += [_point_box(ax, x, y) for x, y in s]
    taken += [_point_box(ax, t * lx, t * ly, half=5)
              for lx, ly in tips for t in np.linspace(0.3, 1.0, 15)]

    for (lx, ly), name in zip(tips, loadings.index):
        direction = np.array([lx, ly]) / (np.hypot(lx, ly) or 1.0)
        normal = np.array([-direction[1], direction[0]])
        offsets = [d * direction + k * normal for d in (10, 18) for k in (0, 10, -10, 20, -20)]
        _place(ax, renderer, taken, (lx, ly), str(name), offsets, ha="center", va="center",
               fontsize=10, fontweight="bold", color=PALETTE[5])
    for (x, y), name in zip(s, scores.index):
        _place(ax, renderer, taken, (x, y), str(name), _OFFSETS, fontsize=7, color="0.3")
    return fig


_OFFSETS = [(4, 4), (4, -10), (-4, 4), (-4, -10), (6, -3), (-6, -3), (0, 7), (0, -13)]


def _point_box(ax, x, y, half=4):
    px, py = ax.transData.transform((x, y))
    return matplotlib.transforms.Bbox.from_extents(px - half, py - half, px + half, py + half)


def _place(ax, renderer, taken, xy, text, offsets, ha=None, va="baseline", **style):
    """Annotate xy with the first offset (in points) whose label overlaps nothing in taken."""
    for dx, dy in offsets:
        label = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points",
                            ha=ha or ("left" if dx >= 0 else "right"), va=va, **style)
        box = label.get_window_extent(renderer)
        if not any(box.overlaps(other) for other in taken):
            break
        label.remove()
    else:
        dx, dy = offsets[0]
        label = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points",
                            ha=ha or "left", va=va, **style)
    taken.append(label.get_window_extent(renderer))


def sample_dendrogram(df: pd.DataFrame, columns, title: str = ""):
    """Ward linkage on standardised concentrations (Euclidean distance)."""
    z = StandardScaler().fit_transform(df[list(columns)].to_numpy(dtype=float))
    tree = linkage(z, method="ward")
    labels = df[ID_COLUMN].tolist() if ID_COLUMN in df.columns else [str(i) for i in df.index]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(df) + 1.5), layout="constrained")
    dendrogram(tree, labels=labels, orientation="right", ax=ax, color_threshold=0,
               above_threshold_color=PALETTE[0])
    ax.set_xlabel("Ward linkage distance (standardised units)")
    ax.set_ylabel("Sample")
    ax.set_title(title or "Dendrogram of samples", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def variable_dendrogram(r: pd.DataFrame, title: str = ""):
    """Average linkage on the distance 1 - r between variables."""
    distance = 1.0 - r.to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(r) + 1.5), layout="constrained")
    dendrogram(tree, labels=list(r.index), orientation="right", ax=ax, color_threshold=0,
               above_threshold_color=PALETTE[0])
    ax.set_xlabel("Distance (1 - Pearson r), average linkage")
    ax.set_ylabel("Metal")
    ax.set_title(title or "Dendrogram of metals", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    return fig


def index_bars(values: pd.Series, labels, column: str, title: str = "", reference: float | None = None,
               reference_label: str = ""):
    """Bar chart of one index value per sample."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(list(labels), values.to_numpy(), color=PALETTE[0])
    if reference is not None:
        ax.axhline(reference, color=PALETTE[5], linestyle="--", linewidth=1.2, label=reference_label)
        ax.legend(frameon=False)
    ax.set_xlabel("Sample")
    ax.set_ylabel(column)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(title or column, loc="left")
    _style(ax)
    return fig
