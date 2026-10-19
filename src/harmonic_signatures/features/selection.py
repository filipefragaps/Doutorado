from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

from ..schema.models import HARMONIC_FAMILIES, INTERHARMONIC_FAMILIES

_FEATURE_KEY_RE = re.compile(
    r"^(?P<family>[A-Za-z0-9\-]+?)(?P<order>\d{2})?_(?P<variable>[VI])(?P<phase>[abc])_(?P<statistic>avg|max)"
    r"(?P<source>_harm|_inter)?$"
)

_SOURCE_SHEET_BY_SUFFIX = {"_harm": "harmonicas", "_inter": "interharmonicas", None: None}


def parse_feature_key(key: str) -> dict:
    """Parse a matrix feature key back into its structured parts. Handles both
    plain keys (e.g. 'H05_Ia_avg') and the '_harm'/'_inter' suffixed keys that
    `organize.join.nearest_timestamp_join` produces for columns that exist,
    ambiguously, in both source sheets (only the H-max duplicate block today)."""

    match = _FEATURE_KEY_RE.match(key)
    if not match:
        return {"family": None, "order": None, "variable": None, "phase": None, "statistic": None, "source_sheet": None}
    d = match.groupdict()
    d["order"] = int(d["order"]) if d["order"] else None
    d["source_sheet"] = _SOURCE_SHEET_BY_SUFFIX[d.pop("source")]
    return d


@dataclass
class FeatureSelectionResult:
    ranking: pd.DataFrame
    selected_features: list[str]
    dropped_redundant: dict[str, str] = field(default_factory=dict)

    def summarize_by(self, *keys: str) -> pd.DataFrame:
        rows = [{**parse_feature_key(f), "feature": f} for f in self.selected_features]
        df = pd.DataFrame(rows)
        return df.groupby(list(keys), observed=True, dropna=False).size().rename("n_selected").reset_index()


def rank_features_univariate(
    matrix_df: pd.DataFrame, feature_cols: list[str], *, label_col: str = "label", method: str = "f_classif",
    random_state: int = 42,
) -> pd.DataFrame:
    X = matrix_df[feature_cols].copy()
    X = X.fillna(X.mean())
    y = matrix_df[label_col].to_numpy()

    if method == "f_classif":
        scores, p_values = f_classif(X.to_numpy(), y)
        return pd.DataFrame({"feature": feature_cols, "score": scores, "p_value": p_values}).sort_values(
            "score", ascending=False
        ).reset_index(drop=True)
    if method == "mutual_info":
        scores = mutual_info_classif(X.to_numpy(), y, random_state=random_state)
        return pd.DataFrame({"feature": feature_cols, "score": scores}).sort_values(
            "score", ascending=False
        ).reset_index(drop=True)
    raise ValueError(f"método desconhecido: {method}")


def remove_redundant_features(
    matrix_df: pd.DataFrame, ranked_features: list[str], *, corr_threshold: float = 0.95,
) -> tuple[list[str], dict[str, str]]:
    """Greedy redundancy elimination: walk features best-first, drop a feature if
    it correlates above threshold with a feature already kept (which, by
    construction, ranks at least as high)."""

    corr = matrix_df[ranked_features].corr().abs()
    kept: list[str] = []
    dropped_because_of: dict[str, str] = {}

    for feat in ranked_features:
        redundant_with = next((k for k in kept if corr.loc[feat, k] >= corr_threshold), None)
        if redundant_with is not None:
            dropped_because_of[feat] = redundant_with
        else:
            kept.append(feat)

    return kept, dropped_because_of


_META_COLUMNS = {
    "sample_id", "sample_id_harm", "sample_id_inter",
    "label", "timestamp", "timestamp_inter",
    "join_delta_seconds", "matched",
}


def build_candidate_pool(matrix_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Single source of truth for the full spectral candidate pool (harmonic +
    inter-harmonic), used consistently across RQ1 and RQ2 so both draw from the
    same 984-channel space the thesis documents. Classifies every non-meta column
    by spectral family via `parse_feature_key` rather than a naive column-name
    filter (e.g. `startswith('H')`), which both misses inter-harmonic channels
    entirely and mishandles the duplicated H-max block (Fase 1, Seção 3.1).

    For the H-max block that collides between the two source sheets (Seção
    resultados_bd), only the `_harm`-suffixed version is kept as the harmonic
    candidate -- the `_inter`-suffixed duplicate is dropped rather than counted
    as an inter-harmonic channel, matching the canonical-source decision already
    made in notebook 10."""

    candidate_cols = [c for c in matrix_df.columns if c not in _META_COLUMNS]
    specs = {c: parse_feature_key(c) for c in candidate_cols}

    harm_cols = [
        c for c, s in specs.items()
        if s["family"] in HARMONIC_FAMILIES and s["source_sheet"] != "interharmonicas"
    ]
    inter_cols = [c for c, s in specs.items() if s["family"] in INTERHARMONIC_FAMILIES]
    return harm_cols, inter_cols


def select_features(
    matrix_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    label_col: str = "label",
    method: str = "f_classif",
    top_k: int | None = None,
    corr_threshold: float = 0.95,
) -> FeatureSelectionResult:
    ranking = rank_features_univariate(matrix_df, feature_cols, label_col=label_col, method=method)
    kept, dropped = remove_redundant_features(matrix_df, ranking["feature"].tolist(), corr_threshold=corr_threshold)
    if top_k is not None:
        kept = kept[:top_k]
    return FeatureSelectionResult(ranking=ranking, selected_features=kept, dropped_redundant=dropped)
