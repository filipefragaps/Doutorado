from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_WORKBOOK = RAW_DIR / "BANCO_DADOS_CONSOLIDADO.xlsx"
LABEL_REGISTRY_PATH = INTERIM_DIR / "label_registry.json"

DEFAULT_JOIN_TOLERANCE_SECONDS = 90
