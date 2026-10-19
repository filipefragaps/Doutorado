from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SequenceTensorBundle:
    tensor: np.ndarray  # (n_samples, n_orders, n_channels)
    static: pd.DataFrame  # (n_samples, n_static_features), index=sample_id
    y: np.ndarray  # (n_samples,)
    sample_ids: list[str]
    channel_names: list[str]
    order_values: list[int]


def _channel_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["family"].astype(str)
        + "_"
        + df["variable"].astype(str)
        + df["phase"].astype(str)
        + "_"
        + df["statistic"].astype(str)
    )


def build_sequence_tensor(
    long_df: pd.DataFrame, order_families: tuple[str, ...] = ("H", "IH")
) -> SequenceTensorBundle:
    seq_mask = long_df["family"].isin(order_families) & long_df["order"].notna()
    seq_df = long_df[seq_mask].copy()
    static_df_long = long_df[~seq_mask].copy()

    sample_ids = sorted(long_df["sample_id"].unique())
    sample_index = {s: i for i, s in enumerate(sample_ids)}

    seq_df["channel"] = _channel_key(seq_df)
    channel_names = sorted(seq_df["channel"].unique())
    channel_index = {c: i for i, c in enumerate(channel_names)}

    order_values = sorted(int(o) for o in seq_df["order"].dropna().unique())
    order_index = {o: i for i, o in enumerate(order_values)}

    tensor = np.full(
        (len(sample_ids), len(order_values), len(channel_names)), np.nan, dtype="float32"
    )
    si = seq_df["sample_id"].map(sample_index).to_numpy()
    oi = seq_df["order"].astype(int).map(order_index).to_numpy()
    ci = seq_df["channel"].map(channel_index).to_numpy()
    tensor[si, oi, ci] = seq_df["value"].to_numpy()

    static_df_long["feature"] = _channel_key(static_df_long)
    static_wide = static_df_long.pivot_table(
        index="sample_id", columns="feature", values="value", aggfunc="first"
    ).reindex(sample_ids)

    label_map = long_df.groupby("sample_id", observed=True)["label"].first()
    y = label_map.reindex(sample_ids).to_numpy()

    return SequenceTensorBundle(
        tensor=tensor,
        static=static_wide,
        y=y,
        sample_ids=sample_ids,
        channel_names=channel_names,
        order_values=order_values,
    )
