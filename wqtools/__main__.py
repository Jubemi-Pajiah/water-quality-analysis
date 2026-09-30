"""Command line interface: python -m wqtools <command> [options]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import indices, plots, stats  # noqa: E402
from .io import ID_COLUMN, load_samples, load_standards, value_columns  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASETS = {
    "groundwater": (DATA_DIR / "groundwater_metals.csv", "Groundwater"),
    "storage_water": (DATA_DIR / "storagewater_metals.csv", "Storage water"),
}
STANDARDS = DATA_DIR / "standards.csv"
DPI = 200


def _save_fig(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=DPI)
    plt.close(fig)


def _save_csv(table: pd.DataFrame, out: Path, name: str, index: bool = False) -> None:
    table.to_csv(out / name, index=index, float_format="%.6g", lineterminator="\n")


def _multivariate_columns(df: pd.DataFrame) -> list[str]:
    columns = [c for c in stats.CLUSTER_METALS if c in df.columns]
    return columns or value_columns(df)


def _report_dropped(dropped: list[str], step: str) -> None:
    if dropped:
        print(f"  {step}: dropped constant column(s) {', '.join(dropped)}")


def run_wqi(df, standards, out, label):
    result = indices.wqi(df, standards, details=True)
    result["WQI class"] = indices.wqi_class(result["WQI"])
    _save_csv(result, out, "wqi.csv")
    _save_fig(plots.index_bars(result["WQI"], result[ID_COLUMN], "WQI",
                               title=f"{label}: Water Quality Index", reference=100,
                               reference_label="WQI = 100 (upper bound of 'good')"), out, "wqi.png")
    print(f"  WQI: {result['WQI'].min():.1f} to {result['WQI'].max():.1f}")


def run_wpi(df, standards, out, label):
    result = indices.wpi(df, standards, details=True)
    _save_csv(result, out, "wpi.csv")
    _save_fig(plots.index_bars(result["WPI"], result[ID_COLUMN], "WPI",
                               title=f"{label}: Water Pollution Index", reference=1,
                               reference_label="WPI = 1 (mean value at the limit)"), out, "wpi.png")
    print(f"  WPI: {result['WPI'].min():.2f} to {result['WPI'].max():.2f}")


def run_corr(df, out, label):
    corr = stats.correlation_matrix(df)
    _report_dropped(corr.dropped, "correlation")
    _save_csv(corr.r, out, "correlation_r.csv", index=True)
    _save_csv(corr.p, out, "correlation_p.csv", index=True)
    _save_fig(plots.correlation_heatmap(corr.r, corr.p, title=f"{label}: Pearson correlation"),
              out, "correlation.png")


def run_cluster(df, out, label, scale=True, k=None):
    columns = _multivariate_columns(df)
    wcss = stats.elbow_wcss(df, columns, scale=scale)
    silhouette = stats.silhouette_by_k(df, columns, scale=scale)
    elbow_k = stats.elbow_k(wcss)
    best_k = int(silhouette.idxmax())
    k = k or best_k
    labels = stats.kmeans_clusters(df, k, columns, scale=scale)

    space = "standardised" if scale else "unscaled"
    _save_csv(pd.concat([wcss, silhouette], axis=1), out, "cluster_selection.csv", index=True)
    _save_csv(pd.DataFrame({ID_COLUMN: df[ID_COLUMN], "cluster": labels}), out, "cluster_labels.csv")
    _save_csv(stats.cluster_summary(df, labels, columns), out, "cluster_means.csv", index=True)
    _save_fig(plots.elbow_plot(wcss, elbow_k, title=f"{label}: elbow method ({space})"), out, "elbow.png")
    _save_fig(plots.silhouette_plot(silhouette, title=f"{label}: silhouette score ({space})"),
              out, "silhouette.png")
    _save_fig(plots.cluster_scatter(df, labels, title=f"{label}: K-means, k = {k} ({space})"),
              out, "clusters.png")
    print(f"  clusters ({space}): elbow k = {elbow_k}, silhouette best k = {best_k} "
          f"({silhouette[best_k]:.2f}), used k = {k}")


def run_factor(df, out, label):
    columns = _multivariate_columns(df)
    suitability = stats.factor_suitability(df, columns)
    eigenvalues, _ = stats.factor_eigenvalues(df, columns)
    factors = stats.factor_loadings(df, columns)
    _report_dropped(factors.dropped, "factor analysis")

    _save_csv(suitability.summary(), out, "factor_suitability.csv")
    _save_csv(eigenvalues.to_frame(), out, "factor_eigenvalues.csv", index=True)
    _save_csv(factors.loadings.join(factors.communalities), out, "factor_loadings.csv", index=True)
    _save_csv(factors.variance, out, "factor_variance.csv", index=True)
    _save_csv(factors.scores, out, "factor_scores.csv", index=True)
    _save_fig(plots.scree_plot(eigenvalues, title=f"{label}: scree plot"), out, "scree.png")
    _save_fig(plots.loadings_heatmap(factors.loadings, title=f"{label}: varimax loadings"),
              out, "loadings.png")
    _save_fig(plots.biplot(factors.scores, factors.loadings, title=f"{label}: factor biplot"),
              out, "biplot.png")
    print(f"  factor analysis: KMO = {suitability.kmo_overall:.2f}, "
          f"Bartlett p = {suitability.bartlett_p:.1e}, "
          f"cumulative variance = {factors.variance.iloc[-1, -1]:.0%}")


def run_dendro(df, out, label):
    columns = _multivariate_columns(df)
    corr = stats.correlation_matrix(df, columns)
    _report_dropped(corr.dropped, "dendrogram")
    kept = [c for c in columns if c not in corr.dropped]
    _save_fig(plots.sample_dendrogram(df, kept, title=f"{label}: samples (Ward, standardised)"),
              out, "dendrogram_samples.png")
    _save_fig(plots.variable_dendrogram(corr.r, title=f"{label}: metals (1 - r, average linkage)"),
              out, "dendrogram_metals.png")


def _load_standards_or_none(path, sheet):
    if path is None or not Path(path).exists():
        print(f"  WQI/WPI skipped: standards file {path} not found")
        return None
    standards = load_standards(path, sheet)
    if standards.empty:
        print(f"  WQI/WPI skipped: no permissible limits filled in {path}")
        return None
    return standards


def run_all_for(df, out, label, standards, scale):
    out.mkdir(parents=True, exist_ok=True)
    print(f"{label} ({len(df)} samples) -> {out}")
    if standards is not None:
        run_wqi(df, standards, out, label)
        run_wpi(df, standards, out, label)
    run_corr(df, out, label)
    run_cluster(df, out, label, scale=scale)
    run_factor(df, out, label)
    run_dendro(df, out, label)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m wqtools",
                                     description="Water quality indices and multivariate analysis.")
    commands = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(DATASETS["groundwater"][0]),
                        help="sample table, CSV or Excel (default: bundled groundwater data)")
    common.add_argument("--sheet", default=None, help="sheet name when --data is an Excel file")
    common.add_argument("--standards", default=str(STANDARDS),
                        help="standards table, CSV or Excel (default: data/standards.csv)")
    common.add_argument("--standards-sheet", default=None,
                        help="sheet name when --standards is an Excel file")
    common.add_argument("--out", default="outputs", help="output folder (default: outputs)")
    common.add_argument("--no-scale", action="store_true",
                        help="cluster on unscaled concentrations (reproduces the original analysis)")
    common.add_argument("--label", default=None, help="dataset name used in figure titles")

    commands.add_parser("all", parents=[common],
                        help="run every analysis on both bundled datasets")
    commands.add_parser("wqi", parents=[common], help="Water Quality Index")
    commands.add_parser("wpi", parents=[common], help="Water Pollution Index")
    commands.add_parser("corr", parents=[common], help="correlation matrix with p-values")
    cluster = commands.add_parser("cluster", parents=[common], help="K-means, elbow and silhouette")
    cluster.add_argument("--k", type=int, default=None,
                         help="number of clusters (default: k with the highest silhouette score)")
    commands.add_parser("factor", parents=[common], help="KMO, Bartlett, scree, loadings, biplot")
    commands.add_parser("dendro", parents=[common], help="sample and metal dendrograms")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    scale = not args.no_scale
    out = Path(args.out)

    if args.command == "all":
        standards = _load_standards_or_none(args.standards, args.standards_sheet)
        for folder, (path, label) in DATASETS.items():
            run_all_for(load_samples(path), out / folder, label, standards, scale)
        return 0

    df = load_samples(args.data, args.sheet)
    label = args.label or Path(args.data).stem
    out.mkdir(parents=True, exist_ok=True)
    if args.command in ("wqi", "wpi"):
        standards = load_standards(args.standards, args.standards_sheet)
        if standards.empty:
            print(f"No permissible limits filled in {args.standards}", file=sys.stderr)
            return 1
        (run_wqi if args.command == "wqi" else run_wpi)(df, standards, out, label)
    elif args.command == "corr":
        run_corr(df, out, label)
    elif args.command == "cluster":
        run_cluster(df, out, label, scale=scale, k=args.k)
    elif args.command == "factor":
        run_factor(df, out, label)
    elif args.command == "dendro":
        run_dendro(df, out, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
