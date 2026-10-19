from __future__ import annotations

import numpy as np
from scipy import stats


def confidence_interval_mean(values: np.ndarray, confidence: float = 0.95) -> dict:
    values = np.asarray(values, dtype="float64")
    values = values[~np.isnan(values)]
    n = len(values)
    mean = values.mean()
    sem = stats.sem(values)
    lo, hi = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    return {"mean": float(mean), "ci_low": float(lo), "ci_high": float(hi), "n": n, "confidence": confidence}


def correlation_with_ci(x: np.ndarray, y: np.ndarray, *, method: str = "pearson", confidence: float = 0.95) -> dict:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"método desconhecido: {method}")

    n = len(x)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3) if n > 3 else np.nan
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    lo, hi = np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)

    return {"method": method, "r": float(r), "p_value": float(p), "n": n,
            "ci_low": float(lo), "ci_high": float(hi), "confidence": confidence}
