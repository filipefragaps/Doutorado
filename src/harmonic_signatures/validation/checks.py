from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_JOIN_TOLERANCE_SECONDS
from ..organize.join import nearest_timestamp_join
from ..organize.long_format import to_long
from ..organize.matrix_format import to_matrix
from ..schema.models import RawSheet
from .report import CrossSheetConsistencyReport, ValidationReport


def validate_sheet(sheet: RawSheet) -> ValidationReport:
    df = sheet.dataframe
    measurement_cols = sheet.measurement_columns()

    missing_pct = df[measurement_cols].isna().mean() * 100
    zero_pct = (df[measurement_cols] == 0).mean() * 100
    n_duplicate_rows = int(df.duplicated().sum())
    label_counts = df[sheet.label_column].value_counts()

    invalid_status_count = 0
    invalid_status_labels = pd.Series(dtype="int64")
    if sheet.status_column is not None:
        invalid_mask = df[sheet.status_column].astype(str).str.lower().str.contains("inv")
        invalid_status_count = int(invalid_mask.sum())
        invalid_status_labels = df.loc[invalid_mask, sheet.label_column].value_counts()

    return ValidationReport(
        sheet_name=sheet.name,
        n_rows=len(df),
        n_columns=len(df.columns),
        missing_pct=missing_pct,
        zero_pct=zero_pct,
        n_duplicate_rows=n_duplicate_rows,
        label_counts=label_counts,
        invalid_status_count=invalid_status_count,
        invalid_status_labels=invalid_status_labels,
    )


def compare_h_max_across_sheets(
    harm_sheet: RawSheet,
    inter_sheet: RawSheet,
    *,
    tolerance_seconds: float = DEFAULT_JOIN_TOLERANCE_SECONDS,
    atol: float = 1e-6,
) -> CrossSheetConsistencyReport:
    """Compare the harmonic 'max' columns that share identical names between the
    two sheets. These are NOT duplicates of each other (confirmed empirically:
    low correlation, no fixed scale ratio) — they are distinct measurement
    sources that happen to reuse the same column-naming convention. Rows are
    matched by nearest timestamp within the same label since the two sheets are
    independent time series."""

    harm_matrix = to_matrix(to_long(harm_sheet))
    inter_matrix = to_matrix(to_long(inter_sheet))

    shared_cols = sorted(
        (set(harm_matrix.columns) & set(inter_matrix.columns)) - {"sample_id", "label", "timestamp"}
    )
    if not shared_cols:
        return CrossSheetConsistencyReport(0, 0, 0.0, 0.0, pd.Series(dtype=float))

    joined = nearest_timestamp_join(harm_matrix, inter_matrix, tolerance_seconds=tolerance_seconds)
    matched = joined[joined["matched"]]

    per_channel_mismatch_pct = {}
    total_compared = 0
    total_mismatch = 0
    max_abs_diff = 0.0
    for col in shared_cols:
        diff = (matched[f"{col}_harm"] - matched[f"{col}_inter"]).abs()
        mismatch_mask = diff > atol
        per_channel_mismatch_pct[col] = mismatch_mask.mean() * 100
        total_compared += len(diff)
        total_mismatch += int(mismatch_mask.sum())
        if len(diff) and float(diff.max()) > max_abs_diff:
            max_abs_diff = float(diff.max())

    mismatch_pct = (total_mismatch / total_compared * 100) if total_compared else 0.0

    return CrossSheetConsistencyReport(
        n_compared_channels=len(shared_cols),
        n_compared_values=total_compared,
        mismatch_pct=mismatch_pct,
        max_abs_diff=max_abs_diff,
        per_channel_mismatch_pct=pd.Series(per_channel_mismatch_pct),
    )
