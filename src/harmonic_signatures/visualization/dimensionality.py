from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None


def compute_embedding(
    matrix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """PCA / t-SNE / UMAP embedding of a sample-level matrix. NaNs are mean-imputed
    per column — module 8 (feature selection) is where a more principled missing-data
    strategy belongs; here the goal is a quick visual read of separability."""

    X = matrix_df[feature_cols].to_numpy(dtype="float64").copy()
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    X = StandardScaler().fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(X)
        explained = reducer.explained_variance_ratio_
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state, **kwargs)
        embedding = reducer.fit_transform(X)
        explained = None
    elif method == "umap":
        if umap is None:
            raise ImportError("umap-learn não está instalado")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        embedding = reducer.fit_transform(X)
        explained = None
    else:
        raise ValueError(f"método desconhecido: {method}")

    cols = [f"dim{i+1}" for i in range(n_components)]
    out = pd.DataFrame(embedding, columns=cols, index=matrix_df.index)
    out["label"] = matrix_df["label"].values
    id_col = "sample_id" if "sample_id" in matrix_df.columns else next(
        (c for c in matrix_df.columns if c.startswith("sample_id")), None
    )
    out["sample_id"] = matrix_df[id_col].values if id_col else matrix_df.index
    out.attrs["explained_variance_ratio"] = explained
    out.attrs["method"] = method
    return out


def plot_embedding_2d(embedding_df: pd.DataFrame, *, color_by: str = "label", ax: plt.Axes | None = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    categories = embedding_df[color_by].astype(str).unique()
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, palette))

    for cat in categories:
        sub = embedding_df[embedding_df[color_by].astype(str) == cat]
        ax.scatter(sub["dim1"], sub["dim2"], s=8, alpha=0.6, label=cat, color=color_map[cat])

    method = embedding_df.attrs.get("method", "")
    explained = embedding_df.attrs.get("explained_variance_ratio")
    xlabel, ylabel = "dim 1", "dim 2"
    if explained is not None:
        xlabel = f"PC1 ({explained[0]*100:.1f}%)"
        ylabel = f"PC2 ({explained[1]*100:.1f}%)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Embedding ({method})")
    ax.legend(fontsize=7, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig
