from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_train_test_split(
    matrix_df: pd.DataFrame,
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    test_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological train/test split, done independently within each label.

    Each of the 17 experimental conditions is a single continuous acquisition
    session (Seção~sec:bd), not several independent runs -- there is no group
    column that would make `GroupKFold` meaningful (group == label). A random
    stratified split therefore tests generalization to temporally-adjacent
    acquisitions of the *same* session, not to a new experimental run. Sorting
    each label's rows by timestamp and holding out the last `test_size` fraction
    as test is the honest fallback until independent repeated sessions exist
    (see the response to the advisor's ponto 1): it at least prevents a test
    row from sitting between two training rows a few seconds apart.

    Returns (train_idx, test_idx) as positional integer arrays into `matrix_df`,
    so callers can do `matrix_df.iloc[train_idx]` / `.iloc[test_idx]`.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size deve estar em (0, 1), recebido {test_size!r}")

    train_positions: list[int] = []
    test_positions: list[int] = []

    positions = pd.Series(np.arange(len(matrix_df)), index=matrix_df.index)
    for _, group in matrix_df.groupby(label_col, observed=True):
        ordered = group.sort_values(timestamp_col)
        n_test = max(1, round(len(ordered) * test_size))
        n_test = min(n_test, len(ordered) - 1) if len(ordered) > 1 else 0
        test_idx = ordered.index[len(ordered) - n_test:] if n_test else ordered.index[:0]
        train_idx = ordered.index.difference(test_idx)
        train_positions.extend(positions.loc[train_idx].tolist())
        test_positions.extend(positions.loc[test_idx].tolist())

    return np.array(sorted(train_positions)), np.array(sorted(test_positions))


def blocked_temporal_kfold(
    matrix_df: pd.DataFrame,
    *,
    label_col: str = "label",
    timestamp_col: str = "timestamp",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Blocked (contiguous) temporal K-fold, done independently within each
    label: each label's chronologically-sorted rows are cut into `n_splits`
    contiguous blocks, and fold `i` tests on block `i` (from every label) while
    training on the other blocks. This is the within-session analogue of
    `StratifiedKFold` for the 5-fold table (Tabela res_cv) -- it keeps each test
    fold temporally contiguous within its label instead of interleaving randomly
    adjacent acquisitions between train and test, for the same reason
    `temporal_train_test_split` replaces the single 70/30 random split.

    Returns a list of (train_idx, test_idx) positional integer arrays, one pair
    per fold, in the same style as `sklearn`'s `.split()` generators.
    """

    if n_splits < 2:
        raise ValueError(f"n_splits deve ser >= 2, recebido {n_splits!r}")

    positions = pd.Series(np.arange(len(matrix_df)), index=matrix_df.index)
    fold_blocks: list[list[int]] = [[] for _ in range(n_splits)]

    for _, group in matrix_df.groupby(label_col, observed=True):
        ordered = group.sort_values(timestamp_col)
        block_ids = (np.arange(len(ordered)) * n_splits) // len(ordered)
        block_ids = np.minimum(block_ids, n_splits - 1)
        ordered_positions = positions.loc[ordered.index].to_numpy()
        for fold in range(n_splits):
            fold_blocks[fold].extend(ordered_positions[block_ids == fold].tolist())

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_positions = np.arange(len(matrix_df))
    for fold in range(n_splits):
        test_idx = np.array(sorted(fold_blocks[fold]))
        train_idx = np.setdiff1d(all_positions, test_idx, assume_unique=False)
        folds.append((train_idx, test_idx))
    return folds
