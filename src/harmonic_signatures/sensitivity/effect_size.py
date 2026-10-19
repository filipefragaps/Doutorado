from __future__ import annotations

import numpy as np


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size: P(a > b) - P(a < b), in [-1, 1].
    Computed via sort + searchsorted — O(n log n), not the naive O(n*m) loop."""

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")

    b_sorted = np.sort(b)
    less = np.searchsorted(b_sorted, a, side="left")   # count of b < a_i
    not_greater = np.searchsorted(b_sorted, a, side="right")  # count of b <= a_i
    greater = nb - not_greater                          # count of b > a_i

    return float((less.sum() - greater.sum()) / (na * nb))


def interpret_cliffs_delta(delta: float) -> str:
    if np.isnan(delta):
        return "undefined"
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def eta_squared(groups: list[np.ndarray]) -> float:
    """Effect size for a one-way ANOVA (multi-group, unlike Cohen's d/Cliff's
    delta which are pairwise): proportion of total variance explained by group
    membership, SS_between / SS_total. Needed because ANOVA's p-value alone
    says nothing about magnitude, only whether some difference exists."""

    cleaned = [np.asarray(g, dtype="float64") for g in groups]
    cleaned = [g[~np.isnan(g)] for g in cleaned]
    all_values = np.concatenate(cleaned)
    if len(all_values) == 0:
        return float("nan")

    grand_mean = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in cleaned if len(g) > 0)
    ss_total = np.sum((all_values - grand_mean) ** 2)
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def interpret_eta_squared(value: float) -> str:
    if np.isnan(value):
        return "undefined"
    if value < 0.01:
        return "negligible"
    if value < 0.06:
        return "small"
    if value < 0.14:
        return "medium"
    return "large"
