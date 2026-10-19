from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .filters import filter_long


def plot_signature_heatmap(
    long_df: pd.DataFrame,
    *,
    variable: str,
    phase: str,
    statistic: str = "avg",
    family: str | list[str] = ("H", "IH"),
    labels: list[str] | None = None,
    order_range: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """'Mapa de assinatura': mean value per (label, order) — one row per label,
    one column per harmonic/inter-harmonic order. This is the visual signature
    the whole thesis is organized around."""

    sub = filter_long(
        long_df, family=family, variable=variable, phase=phase,
        statistic=statistic, labels=labels,
    )
    sub = sub[sub["order"].notna()]
    if order_range is not None:
        lo, hi = order_range
        sub = sub[(sub["order"] >= lo) & (sub["order"] <= hi)]

    pivot = sub.groupby(["label", "order"], observed=True)["value"].mean().unstack("order")
    pivot = pivot.sort_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * pivot.shape[1]), max(4, 0.35 * pivot.shape[0])))
    else:
        fig = ax.figure

    sns.heatmap(pivot, cmap="rocket_r", cbar_kws={"label": f"{variable}{phase} {statistic} [%]"}, ax=ax)
    ax.set_xlabel("ordem")
    ax.set_ylabel("rótulo")
    ax.set_title(f"Assinatura {variable}{phase} {statistic} por rótulo")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    matrix_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    method: str = "pearson",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    meta_cols = {"sample_id", "label", "timestamp"}
    if feature_cols is None:
        feature_cols = [c for c in matrix_df.columns if c not in meta_cols]

    corr = matrix_df[feature_cols].corr(method=method)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(feature_cols)), max(5, 0.25 * len(feature_cols))))
    else:
        fig = ax.figure

    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, xticklabels=True, yticklabels=True)
    ax.set_title(f"Correlação ({method})")
    fig.tight_layout()
    return fig
