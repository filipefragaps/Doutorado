from __future__ import annotations

import re

from .models import HARMONIC_FAMILIES, INTERHARMONIC_FAMILIES, ColumnSpec

_MEASUREMENT_RE = re.compile(
    r"^(?P<prefix>THD3|THDo|THDe|THD|DC|IH|H|TID-F|K-Factor)"
    r"(?P<order>\d{2})?"
    r"\s+(?P<phasevar>[VI][abc])"
    r"\s+(?P<stat>(?i:avg|max))"
    r"\s*(?:\[%\])?\s*$"
)

_LABEL_NAMES = {"rótulo"}
_TIMESTAMP_NAMES = {"registro"}
_STATUS_NAMES = {"status"}


def _family_domain(prefix: str) -> str | None:
    if prefix in HARMONIC_FAMILIES:
        return "harmonica"
    if prefix in INTERHARMONIC_FAMILIES:
        return "interharmonica"
    return None


def parse_column_name(raw_name: str) -> ColumnSpec:
    stripped = raw_name.strip()
    lowered = stripped.lower()

    if lowered in _LABEL_NAMES:
        return ColumnSpec(raw_name=raw_name, kind="label")
    if lowered in _TIMESTAMP_NAMES:
        return ColumnSpec(raw_name=raw_name, kind="timestamp")
    if lowered in _STATUS_NAMES:
        return ColumnSpec(raw_name=raw_name, kind="status")
    if lowered.endswith("evt"):
        phase = lowered[0] if lowered[0] in "abc" else None
        return ColumnSpec(raw_name=raw_name, kind="event", phase=phase)

    match = _MEASUREMENT_RE.match(stripped)
    if match:
        prefix = match.group("prefix")
        order_raw = match.group("order")
        phasevar = match.group("phasevar")
        stat = match.group("stat").lower()
        return ColumnSpec(
            raw_name=raw_name,
            kind="measurement",
            family=prefix,
            domain=_family_domain(prefix),
            order=int(order_raw) if order_raw is not None else None,
            variable=phasevar[0],
            phase=phasevar[1].lower(),
            statistic=stat,
            unit="%" if "[%]" in stripped else None,
        )

    return ColumnSpec(raw_name=raw_name, kind="unknown")


def parse_columns(names: list[str]) -> dict[str, ColumnSpec]:
    return {name: parse_column_name(name) for name in names}
