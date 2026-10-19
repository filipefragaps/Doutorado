import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harmonic_signatures.io.excel_reader import build_raw_sheet
from harmonic_signatures.schema.models import RawSheet


@pytest.fixture
def small_harmonicas_sheet() -> RawSheet:
    df = pd.DataFrame(
        {
            "Registro": pd.to_datetime(
                ["2025-01-01 10:00:00", "2025-01-01 10:01:00", "2025-01-01 10:02:00"]
            ),
            "Status": ["Válido", "Válido", "Inválido"],
            "THD Va avg [%]": [1.0, 2.0, 3.0],
            "DC Va avg [%]": [0.1, 0.2, 0.3],
            "H01 Va avg [%]": [10.0, 20.0, 30.0],
            "H02 Va avg [%]": [5.0, 10.0, 15.0],
            "H00 Va max [%]": [0.5, 0.5, 0.5],
            "H01 Va max [%]": [11.0, 21.0, 31.0],
            "K-Factor Ia avg": [1.5, 1.6, 1.7],
            "Va evt ": [0, 0, 1],
            "Rótulo": ["- DIODO 1", "- DIODO 1", "- DIODO 2"],
        }
    )
    return build_raw_sheet("HARMONICAS_TEST", df, "harmonica")


@pytest.fixture
def small_interharmonicas_sheet() -> RawSheet:
    df = pd.DataFrame(
        {
            "Registro": pd.to_datetime(
                ["2025-01-01 10:00:20", "2025-01-01 10:01:20", "2025-01-01 10:02:20"]
            ),
            "TID-F Ia avg [%]": [0.3, 0.4, 0.5],
            "IH00 Va avg [%]": [0.05, 0.06, 0.07],
            "IH01 Va avg [%]": [0.15, 0.16, 0.17],
            "H01 Va max [%]": [99.0, 98.0, 97.0],
            "Rótulo": ["- DIODO 1", "- DIODO 1", "- DIODO 2"],
        }
    )
    return build_raw_sheet("INTERHARMONICAS_TEST", df, "interharmonica")
