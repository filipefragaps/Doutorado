from __future__ import annotations

# Domain taxonomy confirmed empirically + with the user: the "RET N PARAF" labels
# are pure mechanical-imbalance test configurations (no rectifier connected) —
# their harmonic profile matches the healthy-machine baseline, not the
# rectifier-loaded combined condition. Single source of truth for which of the
# 17 experimental labels are mechanical-only, electrical-only, or a combination
# of both, reused by every module that needs to scope an analysis to one group.

MECHANICAL_ONLY_LABELS: tuple[str, ...] = (
    "MÁQUINA BALANCEADA E ALINHADO",
    "MÁQUINA BALANCEADA (S/ PARAF)",
    "MÁQUINA DESALINHADA",
    "MÁQUINA DESBALANCEADA (RET 6 PARAF)",
    "MÁQUINA DESBALANCEADA (RET 14 PARAF)",
    "MÁQUINA DESBALANCEADA, DESALINHADA (RET 15 PARAF)",
)

ELECTRICAL_ONLY_LABELS: tuple[str, ...] = (
    "DIODO 1",
    "DIODO 2",
    "DIODO 3",
    "DIODO 4",
    "DIODO 5",
    "DIODO 6",
    "RETIFICADOR COMPLETO",
)

COMBINED_LABELS: tuple[str, ...] = (
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 2)",
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 3)",
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 6)",
    "MÁQUINA BALANCEADA E ALINHADA + RETIFICADOR COMPLETO",
)

_CATEGORY_BY_LABEL = {
    **{label: "mechanical" for label in MECHANICAL_ONLY_LABELS},
    **{label: "electrical" for label in ELECTRICAL_ONLY_LABELS},
    **{label: "combined" for label in COMBINED_LABELS},
}


def category_of(label: str) -> str:
    try:
        return _CATEGORY_BY_LABEL[label]
    except KeyError:
        raise ValueError(f"Rótulo não reconhecido na taxonomia: {label!r}") from None
