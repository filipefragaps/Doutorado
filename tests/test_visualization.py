import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from harmonic_signatures.organize.long_format import to_long
from harmonic_signatures.organize.matrix_format import to_matrix
from harmonic_signatures.visualization.dimensionality import compute_embedding, plot_embedding_2d
from harmonic_signatures.visualization.distributions import plot_boxplot, plot_histogram, plot_violin
from harmonic_signatures.visualization.filters import filter_long
from harmonic_signatures.visualization.heatmaps import plot_correlation_heatmap, plot_signature_heatmap


@pytest.fixture
def long_and_matrix(small_harmonicas_sheet):
    long_df = to_long(small_harmonicas_sheet)
    matrix = to_matrix(long_df)
    return long_df, matrix


def test_filter_long_by_multiple_dimensions(long_and_matrix):
    long_df, _ = long_and_matrix
    sub = filter_long(long_df, family="H", variable="V", phase="a", statistic="avg")
    assert (sub["family"] == "H").all()
    assert (sub["variable"] == "V").all()
    assert (sub["phase"] == "a").all()
    assert (sub["statistic"] == "avg").all()


def test_plot_signature_heatmap_runs(long_and_matrix):
    long_df, _ = long_and_matrix
    fig = plot_signature_heatmap(long_df, variable="V", phase="a", statistic="avg", family="H")
    assert fig is not None


def test_plot_histogram_boxplot_violin_run(long_and_matrix):
    long_df, _ = long_and_matrix
    assert plot_histogram(long_df, family="H", variable="V", phase="a", statistic="avg") is not None
    assert plot_boxplot(long_df, family="H", variable="V", phase="a", statistic="avg") is not None
    assert plot_violin(long_df, family="H", variable="V", phase="a", statistic="avg") is not None


def test_plot_histogram_raises_on_empty_filter(long_and_matrix):
    long_df, _ = long_and_matrix
    with pytest.raises(ValueError):
        plot_histogram(long_df, family="IH")


def test_compute_embedding_pca_shape(long_and_matrix):
    _, matrix = long_and_matrix
    feature_cols = [c for c in matrix.columns if c not in ("sample_id", "label", "timestamp")]
    emb = compute_embedding(matrix, feature_cols, method="pca", n_components=2)
    assert emb.shape[0] == len(matrix)
    assert {"dim1", "dim2", "label", "sample_id"}.issubset(emb.columns)
    assert plot_embedding_2d(emb) is not None


def test_plot_correlation_heatmap_runs(long_and_matrix):
    _, matrix = long_and_matrix
    feature_cols = [c for c in matrix.columns if c not in ("sample_id", "label", "timestamp")]
    fig = plot_correlation_heatmap(matrix, feature_cols=feature_cols)
    assert fig is not None
