from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

HARMONIC_FAMILIES = {"H", "THD", "THDo", "THDe", "THD3", "DC", "K-Factor"}
INTERHARMONIC_FAMILIES = {"IH", "TID-F"}


@dataclass(frozen=True)
class ColumnSpec:
    raw_name: str
    kind: str  # "measurement" | "timestamp" | "status" | "event" | "label" | "unknown"
    family: str | None = None
    domain: str | None = None  # "harmonica" | "interharmonica"
    order: int | None = None
    variable: str | None = None  # "V" | "I"
    phase: str | None = None  # "a" | "b" | "c"
    statistic: str | None = None  # "avg" | "max"
    unit: str | None = None


@dataclass
class RawSheet:
    name: str
    domain_hint: str  # "harmonica" | "interharmonica"
    dataframe: pd.DataFrame
    columns: dict[str, ColumnSpec]
    label_column: str
    timestamp_column: str
    status_column: str | None = None

    def measurement_columns(self) -> list[str]:
        return [c for c, spec in self.columns.items() if spec.kind == "measurement"]


@dataclass
class ImportResult:
    source_path: Path
    imported_at: datetime
    harmonicas: RawSheet
    interharmonicas: RawSheet
    unknown_columns: dict[str, list[str]] = field(default_factory=dict)
