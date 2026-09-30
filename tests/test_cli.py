import matplotlib

matplotlib.use("Agg")

from wqtools.__main__ import main  # noqa: E402

CORE_FILES = [
    "correlation.png", "correlation_r.csv", "correlation_p.csv",
    "elbow.png", "silhouette.png", "clusters.png",
    "cluster_selection.csv", "cluster_labels.csv", "cluster_means.csv",
    "scree.png", "loadings.png", "biplot.png",
    "factor_suitability.csv", "factor_eigenvalues.csv", "factor_loadings.csv",
    "factor_variance.csv", "factor_scores.csv",
    "dendrogram_samples.png", "dendrogram_metals.png",
]
INDEX_FILES = ["wqi.csv", "wqi.png", "wpi.csv", "wpi.png"]


def _test_standards(tmp_path):
    # Round test values, not a drinking water standard.
    path = tmp_path / "standards.csv"
    path.write_text("Parameter,Permissible Limit (Si),Ideal Value (Ci)\n"
                    "Fe,1,0\nZn,1,0\nCu,1,0\nMn,1,0\n")
    return path


def test_all_writes_expected_files(tmp_path):
    out = tmp_path / "out"
    assert main(["all", "--out", str(out), "--standards", str(_test_standards(tmp_path))]) == 0
    for folder in ("groundwater", "storage_water"):
        written = {p.name for p in (out / folder).iterdir()}
        missing = set(CORE_FILES + INDEX_FILES) - written
        assert not missing, f"{folder}: missing {sorted(missing)}"


def test_single_command_no_scale(tmp_path):
    assert main(["cluster", "--out", str(tmp_path), "--no-scale", "--k", "2"]) == 0
    assert (tmp_path / "clusters.png").exists()
