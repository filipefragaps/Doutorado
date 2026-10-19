from __future__ import annotations

import pandas as pd


def _feature_keys(df: pd.DataFrame) -> pd.Series:
    has_order = df["order"].notna()
    order_str = pd.Series("", index=df.index, dtype="string")
    order_str[has_order] = df.loc[has_order, "order"].astype(int).astype(str).str.zfill(2)
    return (
        df["family"].astype(str)
        + order_str
        + "_"
        + df["variable"].astype(str)
        + df["phase"].astype(str)
        + "_"
        + df["statistic"].astype(str)
    )


def to_matrix(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["feature_key"] = _feature_keys(df)

    pivoted = df.pivot_table(index="sample_id", columns="feature_key", values="value", aggfunc="first")
    meta = df.groupby("sample_id", observed=True)[["label", "timestamp"]].first()

    return meta.join(pivoted).reset_index()
