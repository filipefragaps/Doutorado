from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..visualization.filters import filter_long

_CHANNEL_COLS = ["domain", "family", "variable", "phase", "order", "statistic"]


def welch_ttest(long_df: pd.DataFrame, label_a: str, label_b: str, **filters) -> dict:
    sub = filter_long(long_df, **filters)
    a = sub.loc[sub["label"] == label_a, "value"].dropna()
    b = sub.loc[sub["label"] == label_b, "value"].dropna()
    result = stats.ttest_ind(a, b, equal_var=False)
    return {"test": "welch", "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
            "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def mann_whitney(long_df: pd.DataFrame, label_a: str, label_b: str, **filters) -> dict:
    sub = filter_long(long_df, **filters)
    a = sub.loc[sub["label"] == label_a, "value"].dropna()
    b = sub.loc[sub["label"] == label_b, "value"].dropna()
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"test": "mann_whitney", "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
            "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def one_way_anova(long_df: pd.DataFrame, **filters) -> dict:
    sub = filter_long(long_df, **filters)
    groups = [g["value"].dropna().to_numpy() for _, g in sub.groupby("label", observed=True)]
    groups = [g for g in groups if len(g) > 1]
    result = stats.f_oneway(*groups)
    return {"test": "anova", "n_groups": len(groups), "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def channel_label_groups(sub: pd.DataFrame) -> dict[tuple, dict[str, np.ndarray]]:
    """Group a long-format frame by channel and label in a single pandas
    groupby pass, instead of re-filtering the whole frame once per channel
    (which is O(n_channels * n_rows) and gets slow past a few hundred channels
    over tens of thousands of rows — the original sweep_channels took ~5.5min
    for 666 channels this way; groupby-once brings that down to seconds)."""

    groups: dict[tuple, dict[str, np.ndarray]] = {}
    grouped = sub.groupby([*_CHANNEL_COLS, "label"], observed=True)["value"]
    for key, series in grouped:
        *channel_key, label = key
        groups.setdefault(tuple(channel_key), {})[label] = series.dropna().to_numpy()
    return groups


def sweep_channels(
    long_df: pd.DataFrame,
    *,
    test: str = "anova",
    label_a: str | None = None,
    label_b: str | None = None,
    fdr_alpha: float = 0.05,
    channel_filter: dict | None = None,
) -> pd.DataFrame:
    """Run the same hypothesis test independently across every (domain, family,
    variable, phase, order, statistic) channel, then apply Benjamini-Hochberg FDR
    correction across all channels tested — the standard way to avoid false
    discoveries when testing hundreds of harmonic/inter-harmonic channels at once."""

    sub = filter_long(long_df, **(channel_filter or {}))
    if test in ("welch", "mann_whitney"):
        sub = sub[sub["label"].isin([label_a, label_b])]

    channel_groups = channel_label_groups(sub)

    rows = []
    for channel_key, label_values in channel_groups.items():
        ch_dict = dict(zip(_CHANNEL_COLS, channel_key))

        if test == "anova":
            groups = [g for g in label_values.values() if len(g) > 1]
            if len(groups) < 2:
                continue
            stat_result = stats.f_oneway(*groups)
            result = {
                "test": "anova", "n_groups": len(groups),
                "statistic": float(stat_result.statistic), "p_value": float(stat_result.pvalue),
            }
        elif test == "welch":
            a, b = label_values.get(label_a, np.array([])), label_values.get(label_b, np.array([]))
            stat_result = stats.ttest_ind(a, b, equal_var=False)
            result = {
                "test": "welch", "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
                "statistic": float(stat_result.statistic), "p_value": float(stat_result.pvalue),
            }
        elif test == "mann_whitney":
            a, b = label_values.get(label_a, np.array([])), label_values.get(label_b, np.array([]))
            stat_result = stats.mannwhitneyu(a, b, alternative="two-sided")
            result = {
                "test": "mann_whitney", "label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b),
                "statistic": float(stat_result.statistic), "p_value": float(stat_result.pvalue),
            }
        else:
            raise ValueError(f"teste desconhecido: {test}")

        rows.append({**ch_dict, **result})

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    reject, p_adj, _, _ = multipletests(result_df["p_value"].to_numpy(), alpha=fdr_alpha, method="fdr_bh")
    result_df["p_adj"] = p_adj
    result_df["reject_h0"] = reject
    return result_df.sort_values("p_adj")
