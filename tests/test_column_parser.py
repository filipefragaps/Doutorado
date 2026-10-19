import json
from pathlib import Path

import pytest

from harmonic_signatures.schema.column_parser import parse_column_name, parse_columns

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def real_headers() -> dict[str, list[str]]:
    return json.loads((FIXTURES_DIR / "real_headers.json").read_text(encoding="utf-8"))


def test_all_real_columns_are_recognized(real_headers):
    for sheet_name, columns in real_headers.items():
        specs = parse_columns(columns)
        unknown = [c for c, spec in specs.items() if spec.kind == "unknown"]
        assert unknown == [], f"{sheet_name}: unrecognized columns {unknown}"


def test_real_column_count_matches(real_headers):
    for sheet_name, columns in real_headers.items():
        specs = parse_columns(columns)
        assert len(specs) == len(columns)


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "THD Va avg [%]",
            dict(kind="measurement", family="THD", domain="harmonica", order=None,
                 variable="V", phase="a", statistic="avg", unit="%"),
        ),
        (
            "THD3 Ic max [%]",
            dict(kind="measurement", family="THD3", domain="harmonica", order=None,
                 variable="I", phase="c", statistic="max", unit="%"),
        ),
        (
            "DC Vb avg [%]",
            dict(kind="measurement", family="DC", domain="harmonica", order=None,
                 variable="V", phase="b", statistic="avg", unit="%"),
        ),
        (
            "H01 Va avg [%]",
            dict(kind="measurement", family="H", domain="harmonica", order=1,
                 variable="V", phase="a", statistic="avg", unit="%"),
        ),
        (
            "H00 Va max [%]",
            dict(kind="measurement", family="H", domain="harmonica", order=0,
                 variable="V", phase="a", statistic="max", unit="%"),
        ),
        (
            "IH50 Ic avg [%]",
            dict(kind="measurement", family="IH", domain="interharmonica", order=50,
                 variable="I", phase="c", statistic="avg", unit="%"),
        ),
        (
            "TID-F Ia max [%]",
            dict(kind="measurement", family="TID-F", domain="interharmonica", order=None,
                 variable="I", phase="a", statistic="max", unit="%"),
        ),
        (
            "K-Factor Ia avg",
            dict(kind="measurement", family="K-Factor", domain="harmonica", order=None,
                 variable="I", phase="a", statistic="avg", unit=None),
        ),
        (
            "K-Factor Ia Max",
            dict(kind="measurement", family="K-Factor", domain="harmonica", order=None,
                 variable="I", phase="a", statistic="max", unit=None),
        ),
    ],
)
def test_measurement_columns_parsed_correctly(name, expected):
    spec = parse_column_name(name)
    for field, value in expected.items():
        assert getattr(spec, field) == value, f"{name}: {field} expected {value}, got {getattr(spec, field)}"


@pytest.mark.parametrize(
    "name,kind",
    [
        ("Registro", "timestamp"),
        ("Status", "status"),
        ("Va evt ", "event"),
        ("Rótulo", "label"),
        ("RÓTULO", "label"),
    ],
)
def test_special_columns_classified(name, kind):
    assert parse_column_name(name).kind == kind


def test_unrecognized_column_is_unknown():
    assert parse_column_name("Coluna Totalmente Nova XYZ").kind == "unknown"
