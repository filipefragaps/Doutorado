from __future__ import annotations

import pandas as pd

from ..schema.models import RawSheet

_LONG_COLUMNS = [
    "sample_id",
    "label",
    "timestamp",
    "source_sheet",
    "domain",
    "family",
    "variable",
    "phase",
    "order",
    "statistic",
    "value",
]


def to_long(sheet: RawSheet) -> pd.DataFrame:
    measurement_cols = sheet.measurement_columns()

    working = sheet.dataframe[[sheet.label_column, sheet.timestamp_column, *measurement_cols]].copy()
    working.insert(0, "row_id", working.index)

    melted = working.melt(
        id_vars=["row_id", sheet.label_column, sheet.timestamp_column],
        value_vars=measurement_cols,
        var_name="raw_name",
        value_name="value",
    )

    spec_records = [
        {
            "raw_name": name,
            "family": spec.family,
            "domain": spec.domain,
            "variable": spec.variable,
            "phase": spec.phase,
            "order": spec.order,
            "statistic": spec.statistic,
        }
        for name, spec in sheet.columns.items()
        if spec.kind == "measurement"
    ]
    spec_df = pd.DataFrame.from_records(spec_records)

    merged = melted.merge(spec_df, on="raw_name", how="left")

    merged["sample_id"] = sheet.name + "#" + merged["row_id"].astype(str)
    merged["source_sheet"] = sheet.name
    merged = merged.rename(
        columns={sheet.label_column: "label", sheet.timestamp_column: "timestamp"}
    )

    merged["family"] = merged["family"].astype("category")
    merged["domain"] = merged["domain"].astype("category")
    merged["variable"] = merged["variable"].astype("category")
    merged["phase"] = merged["phase"].astype("category")
    merged["statistic"] = merged["statistic"].astype("category")
    merged["order"] = merged["order"].astype("Int8")
    merged["label"] = merged["label"].astype("category")

    return merged[_LONG_COLUMNS]
