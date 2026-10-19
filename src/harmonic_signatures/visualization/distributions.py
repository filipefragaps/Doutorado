from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .filters import filter_long


def _filtered_values(long_df: pd.DataFrame, **filters) -> pd.DataFrame:
    sub = filter_long(long_df, **filters)
    if sub.empty:
        raise ValueError("Filtro não retornou nenhuma linha — confira domain/family/variable/phase/order/statistic/labels")
    return sub


def plot_histogram(long_df: pd.DataFrame, *, bins: int = 50, ax: plt.Axes | None = None, **filters) -> plt.Figure:
    sub = _filtered_values(long_df, **filters)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    sns.histplot(sub["value"], bins=bins, ax=ax, color="#2a78d6")
    ax.set_xlabel("valor")
    ax.set_title("Distribuição")
    fig.tight_layout()
    return fig


def plot_boxplot(long_df: pd.DataFrame, *, by: str = "label", ax: plt.Axes | None = None, **filters) -> plt.Figure:
    sub = _filtered_values(long_df, **filters)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * sub[by].nunique()), 5))
    else:
        fig = ax.figure
    order = sub.groupby(by, observed=True)["value"].median().sort_values(ascending=False).index
    sns.boxplot(data=sub, x=by, y="value", order=order, ax=ax, color="#2a78d6", fliersize=2)
    ax.tick_params(axis="x", rotation=75)
    ax.set_title(f"Boxplot por {by}")
    fig.tight_layout()
    return fig


def plot_violin(long_df: pd.DataFrame, *, by: str = "label", ax: plt.Axes | None = None, **filters) -> plt.Figure:
    sub = _filtered_values(long_df, **filters)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * sub[by].nunique()), 5))
    else:
        fig = ax.figure
    order = sub.groupby(by, observed=True)["value"].median().sort_values(ascending=False).index
    sns.violinplot(data=sub, x=by, y="value", order=order, ax=ax, color="#2a78d6", cut=0)
    ax.tick_params(axis="x", rotation=75)
    ax.set_title(f"Violin plot por {by}")
    fig.tight_layout()
    return fig
