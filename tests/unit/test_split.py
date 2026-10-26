"""Unit tests U02: Deterministic contiguous block split (`rdai.data.split`)."""

from __future__ import annotations

from rdai.data.split import (
    SampleRecord,
    apply_train_background_cap,
    build_train_manifest_with_oversampling,
    split_records_contiguous_blocks,
)


def _make_synthetic_dataset(n_images: int = 1000) -> list[SampleRecord]:
    """Build synthetic sorted dataset of `n_images` with all 4 classes distributed across blocks."""
    records: list[SampleRecord] = []
    for idx in range(1, n_images + 1):
        stem = f"India_{idx:06d}"
        # Assign classes across blocks so all 4 classes exist, with D10 rare
        if idx % 95 == 0:
            classes = {1}  # D10 rare
        elif idx % 7 == 0:
            classes = set()  # Empty background image
        elif idx % 3 == 0:
            classes = {0, 3}  # D00 + D40
        elif idx % 2 == 0:
            classes = {2}  # D20
        else:
            classes = {3}  # D40
        records.append(SampleRecord(stem=stem, file_number=idx, classes=classes))
    return records


def test_u02_disjoint_and_exact_split_proportions() -> None:
    """U02: No image is in two splits; sizes are within 1 image of 70, 15, 15 percent."""
    records = _make_synthetic_dataset(233)  # Arbitrary non-round count
    splits = split_records_contiguous_blocks(
        records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        block_size=50,
        seed=0,
    )

    train_stems = {r.stem for r in splits.train}
    val_stems = {r.stem for r in splits.val}
    test_stems = {r.stem for r in splits.test}

    # Disjointness
    assert train_stems.isdisjoint(val_stems)
    assert train_stems.isdisjoint(test_stems)
    assert val_stems.isdisjoint(test_stems)
    assert len(train_stems) + len(val_stems) + len(test_stems) == len(records)

    # Sizes within 1 image of 70%, 15%, 15%
    n = len(records)
    assert abs(len(splits.train) - 0.70 * n) <= 1.0
    assert abs(len(splits.val) - 0.15 * n) <= 1.0
    assert abs(len(splits.test) - 0.15 * n) <= 1.0


def test_u02_contiguous_blocks_of_50_stay_together() -> None:
    """U02: Contiguous blocks of 50 file numbers stay together in the same split."""
    records = _make_synthetic_dataset(1000)  # 20 blocks of 50 -> 14 train, 3 val, 3 test
    splits = split_records_contiguous_blocks(
        records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        block_size=50,
        seed=0,
    )

    stem_to_split: dict[str, str] = {}
    for r in splits.train:
        stem_to_split[r.stem] = "train"
    for r in splits.val:
        stem_to_split[r.stem] = "val"
    for r in splits.test:
        stem_to_split[r.stem] = "test"

    # Check every contiguous block of 50 images (1..50, 51..100, ..., 951..1000)
    for block_idx in range(20):
        start = block_idx * 50 + 1
        block_splits = {stem_to_split[f"India_{i:06d}"] for i in range(start, start + 50)}
        assert len(block_splits) == 1, f"Block {block_idx} was split across {block_splits}"


def test_u02_deterministic_seed_reproducibility() -> None:
    """U02: Same seed gives the exact same split; different seed changes block assignment."""
    records = _make_synthetic_dataset(1000)
    split_a = split_records_contiguous_blocks(records, block_size=50, seed=0)
    split_b = split_records_contiguous_blocks(records, block_size=50, seed=0)
    split_c = split_records_contiguous_blocks(records, block_size=50, seed=42)

    assert [r.stem for r in split_a.train] == [r.stem for r in split_b.train]
    assert [r.stem for r in split_a.val] == [r.stem for r in split_b.val]
    assert [r.stem for r in split_a.test] == [r.stem for r in split_b.test]
    assert [r.stem for r in split_a.val] != [r.stem for r in split_c.val]


def test_u02_every_class_appears_in_val_and_test() -> None:
    """U02: Every class (0: D00, 1: D10, 2: D20, 3: D40) appears in validation and test."""
    records = _make_synthetic_dataset(1000)
    splits = split_records_contiguous_blocks(records, block_size=50, seed=0)

    train_classes = set().union(*(r.classes for r in splits.train))
    val_classes = set().union(*(r.classes for r in splits.val))
    test_classes = set().union(*(r.classes for r in splits.test))

    assert train_classes == {0, 1, 2, 3}
    assert val_classes == {0, 1, 2, 3}
    assert test_classes == {0, 1, 2, 3}


def test_u02_background_cap_and_d10_oversampling() -> None:
    """Section 7: Cap empty images in train at ~10% and repeat D10 images in train list."""
    records = [
        SampleRecord(stem=f"pos_{i}", file_number=i, classes={3 if i > 2 else 1})
        for i in range(1, 91)
    ] + [SampleRecord(stem=f"empty_{i}", file_number=100 + i, classes=set()) for i in range(1, 51)]

    filtered_train, dropped_empty = apply_train_background_cap(
        records, max_empty_share=0.10, seed=0
    )
    empty_remaining = [r for r in filtered_train if not r.classes]
    assert dropped_empty > 0
    assert len(empty_remaining) / len(filtered_train) <= 0.10 + 1e-6

    manifest = build_train_manifest_with_oversampling(
        filtered_train, minority_class=1, repeat_factor=3
    )
    d10_stems = {r.stem for r in filtered_train if 1 in r.classes}
    for stem in d10_stems:
        assert manifest.count(stem) == 3
