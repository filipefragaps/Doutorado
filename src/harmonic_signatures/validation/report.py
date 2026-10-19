from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd


@dataclass
class ValidationReport:
    sheet_name: str
    n_rows: int
    n_columns: int
    missing_pct: pd.Series
    zero_pct: pd.Series
    n_duplicate_rows: int
    label_counts: pd.Series
    invalid_status_count: int
    invalid_status_labels: pd.Series
    generated_at: datetime = field(default_factory=datetime.now)

    def worst_missing(self, n: int = 10) -> pd.Series:
        return self.missing_pct.sort_values(ascending=False).head(n)

    def worst_zero(self, n: int = 10) -> pd.Series:
        return self.zero_pct.sort_values(ascending=False).head(n)

    def to_markdown(self) -> str:
        lines = [
            f"# Relatório de validação — {self.sheet_name}",
            f"Gerado em: {self.generated_at:%Y-%m-%d %H:%M:%S}",
            "",
            f"- Linhas: {self.n_rows}",
            f"- Colunas: {self.n_columns}",
            f"- Linhas duplicadas: {self.n_duplicate_rows}",
            f"- Registros com Status inválido: {self.invalid_status_count}",
            "",
            "## Contagem por rótulo",
            self.label_counts.to_frame("n").to_markdown(),
            "",
            "## Top 10 colunas com mais ausentes (%)",
            self.worst_missing().to_frame("missing_pct").to_markdown(),
            "",
            "## Top 10 colunas com mais zeros (%)",
            self.worst_zero().to_frame("zero_pct").to_markdown(),
        ]
        return "\n".join(lines)


@dataclass
class CrossSheetConsistencyReport:
    n_compared_channels: int
    n_compared_values: int
    mismatch_pct: float
    max_abs_diff: float
    per_channel_mismatch_pct: pd.Series

    def to_markdown(self) -> str:
        lines = [
            "# Divergência entre colunas H-max de mesmo nome (HARMÔNICAS vs INTERHARMÔNICAS)",
            "Nota: os dois blocos têm o mesmo nome de coluna mas são fontes de dado",
            "distintas (não duplicatas) — este relatório documenta a divergência,",
            "não indica um erro a ser corrigido.",
            f"- Canais comparados: {self.n_compared_channels}",
            f"- Valores comparados: {self.n_compared_values}",
            f"- % de divergência (valores diferentes): {self.mismatch_pct:.4f}%",
            f"- Maior diferença absoluta observada: {self.max_abs_diff}",
            "",
            "## Divergência por canal (%)",
            self.per_channel_mismatch_pct.sort_values(ascending=False).to_frame("mismatch_pct").to_markdown(),
        ]
        return "\n".join(lines)
