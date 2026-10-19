from __future__ import annotations

import pandas as pd


def filter_long(
    long_df: pd.DataFrame,
    *,
    domain: str | list[str] | None = None,
    family: str | list[str] | None = None,
    variable: str | list[str] | None = None,
    phase: str | list[str] | None = None,
    order: int | list[int] | None = None,
    statistic: str | list[str] | None = None,
    labels: str | list[str] | None = None,
) -> pd.DataFrame:
    """Single shared filter for domain/family/variable/phase/order/statistic/label,
    reused by every visualization, statistics and sensitivity function so the
    filtering semantics stay identical across modules."""

    out = long_df
    conditions = {
        "domain": domain,
        "family": family,
        "variable": variable,
        "phase": phase,
        "order": order,
        "statistic": statistic,
        "label": labels,
    }
    for col, value in conditions.items():
        if value is None:
            continue
        values = [value] if isinstance(value, (str, int)) else list(value)
        out = out[out[col].isin(values)]
    return out
