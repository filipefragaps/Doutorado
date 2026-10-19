import numpy as np
import pandas as pd
import pytest

from harmonic_signatures.validation.splitting import (
    blocked_temporal_kfold,
    temporal_train_test_split,
)


def _synthetic_matrix(n_per_label: int = 20, labels: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rows = []
    for label in labels:
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hash(label) % 5)
        for i in range(n_per_label):
            rows.append({"label": label, "timestamp": base + pd.Timedelta(seconds=10 * i), "value": i})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    return df


def test_temporal_train_test_split_no_overlap_and_full_coverage():
    df = _synthetic_matrix()
    train_idx, test_idx = temporal_train_test_split(df, test_size=0.3)

    assert set(train_idx).isdisjoint(set(test_idx))
    assert set(train_idx) | set(test_idx) == set(range(len(df)))


def test_temporal_train_test_split_respects_chronological_order_per_label():
    df = _synthetic_matrix()
    train_idx, test_idx = temporal_train_test_split(df, test_size=0.3)

    for label, group in df.groupby("label"):
        group_positions = set(group.index)
        train_ts = df.iloc[[i for i in train_idx if i in group_positions]]["timestamp"]
        test_ts = df.iloc[[i for i in test_idx if i in group_positions]]["timestamp"]
        assert train_ts.max() < test_ts.min(), f"label {label}: train não termina antes do test começar"


def test_temporal_train_test_split_size_matches_requested_fraction():
    df = _synthetic_matrix(n_per_label=20)
    train_idx, test_idx = temporal_train_test_split(df, test_size=0.3)

    assert len(test_idx) == pytest.approx(0.3 * len(df), abs=df.groupby("label").ngroups)


def test_blocked_temporal_kfold_covers_every_row_exactly_once():
    df = _synthetic_matrix()
    folds = blocked_temporal_kfold(df, n_splits=5)

    assert len(folds) == 5
    all_test = np.concatenate([test_idx for _, test_idx in folds])
    assert sorted(all_test.tolist()) == list(range(len(df)))
    assert len(set(all_test.tolist())) == len(df)


def test_blocked_temporal_kfold_test_blocks_are_temporally_contiguous_per_label():
    df = _synthetic_matrix()
    folds = blocked_temporal_kfold(df, n_splits=5)

    for _, test_idx in folds:
        for label, group in df.groupby("label"):
            group_test = df.iloc[[i for i in test_idx if i in set(group.index)]].sort_values("timestamp")
            if len(group_test) <= 1:
                continue
            gaps = group_test["timestamp"].diff().dropna()
            assert (gaps == pd.Timedelta(seconds=10)).all(), (
                f"label {label}: bloco de teste não é temporalmente contíguo"
            )


def test_blocked_temporal_kfold_train_and_test_disjoint_within_fold():
    df = _synthetic_matrix()
    folds = blocked_temporal_kfold(df, n_splits=4)

    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
