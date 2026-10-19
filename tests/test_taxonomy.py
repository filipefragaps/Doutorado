import pytest

from harmonic_signatures.labels.taxonomy import (
    COMBINED_LABELS,
    ELECTRICAL_ONLY_LABELS,
    MECHANICAL_ONLY_LABELS,
    category_of,
)

ALL_17_LABELS = MECHANICAL_ONLY_LABELS + ELECTRICAL_ONLY_LABELS + COMBINED_LABELS


def test_taxonomy_partitions_all_17_labels_without_overlap():
    assert len(ALL_17_LABELS) == 17
    assert len(set(ALL_17_LABELS)) == 17  # no duplicates across groups
    assert len(MECHANICAL_ONLY_LABELS) == 6
    assert len(ELECTRICAL_ONLY_LABELS) == 7
    assert len(COMBINED_LABELS) == 4


@pytest.mark.parametrize("label", MECHANICAL_ONLY_LABELS)
def test_category_of_mechanical(label):
    assert category_of(label) == "mechanical"


@pytest.mark.parametrize("label", ELECTRICAL_ONLY_LABELS)
def test_category_of_electrical(label):
    assert category_of(label) == "electrical"


@pytest.mark.parametrize("label", COMBINED_LABELS)
def test_category_of_combined(label):
    assert category_of(label) == "combined"


def test_category_of_unknown_label_raises():
    with pytest.raises(ValueError):
        category_of("RÓTULO INEXISTENTE")
