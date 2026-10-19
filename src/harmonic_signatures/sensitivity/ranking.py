from __future__ import annotations

import pandas as pd

from ..stats.group_tests import channel_label_groups
from ..visualization.filters import filter_long
from .effect_size import cliffs_delta, cohens_d, eta_squared, interpret_cliffs_delta, interpret_eta_squared

_CHANNEL_COLS = ["domain", "family", "variable", "phase", "order", "statistic"]


def rank_channels_by_effect_size(
    long_df: pd.DataFrame,
    label_a: str,
    label_b: str,
    *,
    method: str = "cohens_d",
    channel_filter: dict | None = None,
) -> pd.DataFrame:
    """Rank every (domain, family, variable, phase, order, statistic) channel by
    how strongly it separates two labels — the basis for module 8's automatic
    feature selection."""

    sub = filter_long(long_df, **(channel_filter or {}))
    sub = sub[sub["label"].isin([label_a, label_b])]
    channel_groups = channel_label_groups(sub)

    rows = []
    for channel_key, label_values in channel_groups.items():
        ch_dict = dict(zip(_CHANNEL_COLS, channel_key))
        a = label_values.get(label_a)
        b = label_values.get(label_b)
        if a is None or b is None:
            continue

        if method == "cohens_d":
            effect = cohens_d(a, b)
        elif method == "cliffs_delta":
            effect = cliffs_delta(a, b)
        else:
            raise ValueError(f"método desconhecido: {method}")

        rows.append({**ch_dict, "effect_size": effect, "abs_effect_size": abs(effect)})

    result = pd.DataFrame(rows).sort_values("abs_effect_size", ascending=False)
    if method == "cliffs_delta" and not result.empty:
        result["magnitude"] = result["effect_size"].apply(interpret_cliffs_delta)
    return result.reset_index(drop=True)


def rank_channels_by_eta_squared(long_df: pd.DataFrame, *, channel_filter: dict | None = None) -> pd.DataFrame:
    """Rank every channel by eta-squared across all labels present after
    filtering — the multi-group counterpart to `rank_channels_by_effect_size`
    (which only compares two labels at a time). Use this to size the effect
    behind an ANOVA run over the same label set (e.g. `sweep_channels`)."""

    sub = filter_long(long_df, **(channel_filter or {}))
    channel_groups = channel_label_groups(sub)

    rows = []
    for channel_key, label_values in channel_groups.items():
        ch_dict = dict(zip(_CHANNEL_COLS, channel_key))
        effect = eta_squared(list(label_values.values()))
        rows.append({**ch_dict, "eta_squared": effect, "magnitude": interpret_eta_squared(effect)})

    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False).reset_index(drop=True)
