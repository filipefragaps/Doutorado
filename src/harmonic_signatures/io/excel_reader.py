from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..labels.registry import normalize_label
from ..schema.column_parser import parse_columns
from ..schema.models import ImportResult, RawSheet


class SheetLayoutError(ValueError):
    pass


def find_sheet_names(path: Path) -> tuple[str, str]:
    names = pd.ExcelFile(path, engine="openpyxl").sheet_names
    inter = [n for n in names if "inter" in n.lower()]
    harm = [n for n in names if "inter" not in n.lower() and "harm" in n.lower()]
    if len(inter) != 1 or len(harm) != 1:
        raise SheetLayoutError(
            f"Esperava exatamente uma planilha de harmônicas e uma de "
            f"inter-harmônicas, encontrei: {names}"
        )
    return harm[0], inter[0]


def read_sheet_raw(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def build_raw_sheet(
    name: str, dataframe: pd.DataFrame, domain_hint: str, *, downcast_measurements: bool = True
) -> RawSheet:
    columns = parse_columns(list(dataframe.columns))

    label_cols = [c for c, spec in columns.items() if spec.kind == "label"]
    timestamp_cols = [c for c, spec in columns.items() if spec.kind == "timestamp"]
    status_cols = [c for c, spec in columns.items() if spec.kind == "status"]

    if len(label_cols) != 1:
        raise SheetLayoutError(f"Planilha {name!r}: esperava 1 coluna de rótulo, achei {label_cols}")
    if len(timestamp_cols) != 1:
        raise SheetLayoutError(f"Planilha {name!r}: esperava 1 coluna de timestamp, achei {timestamp_cols}")

    label_column = label_cols[0]
    timestamp_column = timestamp_cols[0]
    status_column = status_cols[0] if status_cols else None

    dataframe = dataframe.copy()
    dataframe[timestamp_column] = pd.to_datetime(dataframe[timestamp_column])
    dataframe[label_column] = dataframe[label_column].map(normalize_label)

    if downcast_measurements:
        measurement_cols = [c for c, spec in columns.items() if spec.kind == "measurement"]
        dataframe[measurement_cols] = dataframe[measurement_cols].astype("float32")

    return RawSheet(
        name=name,
        domain_hint=domain_hint,
        dataframe=dataframe,
        columns=columns,
        label_column=label_column,
        timestamp_column=timestamp_column,
        status_column=status_column,
    )


def read_workbook(path: Path) -> ImportResult:
    path = Path(path)
    harm_name, inter_name = find_sheet_names(path)

    harm_df = read_sheet_raw(path, harm_name)
    inter_df = read_sheet_raw(path, inter_name)

    harmonicas = build_raw_sheet(harm_name, harm_df, "harmonica")
    interharmonicas = build_raw_sheet(inter_name, inter_df, "interharmonica")

    unknown = {}
    for sheet in (harmonicas, interharmonicas):
        unknown_cols = [c for c, spec in sheet.columns.items() if spec.kind == "unknown"]
        if unknown_cols:
            unknown[sheet.name] = unknown_cols

    return ImportResult(
        source_path=path,
        imported_at=datetime.now(),
        harmonicas=harmonicas,
        interharmonicas=interharmonicas,
        unknown_columns=unknown,
    )
