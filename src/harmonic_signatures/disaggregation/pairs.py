from __future__ import annotations

from ..labels.taxonomy import COMBINED_LABELS

# Combined-condition label -> (pure machine-baseline label, pure rectifier/diode-fault label).
# Derived from the experiment naming convention "<máquina> + (-DIODO N)" /
# "<máquina> + RETIFICADOR COMPLETO", where both the machine-only and the
# fault-only runs also exist in isolation in this dataset — the precondition
# that makes supervised disaggregation (rather than blind source separation)
# possible here.
DISAGGREGATION_PAIRS: dict[str, tuple[str, str]] = {
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 2)": ("MÁQUINA BALANCEADA E ALINHADO", "DIODO 2"),
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 3)": ("MÁQUINA BALANCEADA E ALINHADO", "DIODO 3"),
    "MÁQUINA BALANCEADA E ALINHADA + (-DIODO 6)": ("MÁQUINA BALANCEADA E ALINHADO", "DIODO 6"),
    "MÁQUINA BALANCEADA E ALINHADA + RETIFICADOR COMPLETO": ("MÁQUINA BALANCEADA E ALINHADO", "RETIFICADOR COMPLETO"),
}

assert set(DISAGGREGATION_PAIRS) == set(COMBINED_LABELS), (
    "DISAGGREGATION_PAIRS deve cobrir exatamente os rótulos combinados da taxonomia"
)
