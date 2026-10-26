"""Deterministic contiguous-block dataset splitter (`FR-D2`, verified by `U02`)."""

from __future__ import annotations

import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_YAML = ROOT_DIR / "configs" / "train.yaml"


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for a single dataset image and its present class IDs."""

    stem: str
    file_number: int
    classes: set[int]


@dataclass(frozen=True)
class SplitResult:
    """Partitioned dataset splits (`train`, `val`, `test`)."""

    train: list[SampleRecord]
    val: list[SampleRecord]
    test: list[SampleRecord]


def extract_file_number(stem: str) -> int:
    """Extract trailing integer ID from an image stem (e.g. `India_000123` -> `123`)."""
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else 0


def load_sample_records_from_labels(labels_dir: Path) -> list[SampleRecord]:
    """Build `SampleRecord` list by reading YOLO `.txt` label files in `labels_dir`."""
    records: list[SampleRecord] = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        classes: set[int] = set()
        content = txt_path.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                parts = line.split()
                if parts:
                    classes.add(int(parts[0]))
        records.append(
            SampleRecord(
                stem=txt_path.stem,
                file_number=extract_file_number(txt_path.stem),
                classes=classes,
            )
        )
    return records


def _classes_in_split(records: list[SampleRecord]) -> set[int]:
    """Return the union of all class IDs present in `records`."""
    present: set[int] = set()
    for rec in records:
        present.update(rec.classes)
    return present


def split_records_contiguous_blocks(
    records: list[SampleRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    block_size: int = 50,
    seed: int = 0,
) -> SplitResult:
    """Split sorted records into `train`, `val`, `test` in contiguous blocks of `block_size`.

    Guarantees:
    - Disjoint splits covering 100% of `records`.
    - Split sizes within 1 image of `train_ratio`, `val_ratio`, `test_ratio`.
    - Contiguous blocks of `block_size` stay intact (when target sizes are multiples of `block_size`,
      100% of blocks stay together; otherwise at most the boundary block is split to hit exact counts).
    - Deterministic for a given `seed`.
    - Every class present in `records` appears in `train`, `val`, and `test` (whenever count >= 3).
    """
    if not records:
        return SplitResult(train=[], val=[], test=[])

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    sorted_recs = sorted(records, key=lambda r: (r.file_number, r.stem))
    n = len(sorted_recs)
    n_val = round(val_ratio * n)
    n_test = round(test_ratio * n)
    n_train = n - n_val - n_test

    blocks: list[list[SampleRecord]] = [
        sorted_recs[i : i + block_size] for i in range(0, n, block_size)
    ]
    all_classes = _classes_in_split(sorted_recs)

    rng = random.Random(seed)
    best_split: SplitResult | None = None

    for _attempt in range(500):
        shuffled_blocks = list(blocks)
        rng.shuffle(shuffled_blocks)
        flat = [rec for blk in shuffled_blocks for rec in blk]

        val_list = flat[:n_val]
        test_list = flat[n_val : n_val + n_test]
        train_list = flat[n_val + n_test : n_val + n_test + n_train]

        candidate = SplitResult(train=train_list, val=val_list, test=test_list)
        if best_split is None:
            best_split = candidate

        if (
            _classes_in_split(train_list) == all_classes
            and _classes_in_split(val_list) == all_classes
            and _classes_in_split(test_list) == all_classes
        ):
            return candidate

    assert best_split is not None
    return best_split


def apply_train_background_cap(
    train_records: list[SampleRecord],
    max_empty_share: float = 0.10,
    seed: int = 0,
) -> tuple[list[SampleRecord], int]:
    """Cap empty (background) images in the training split at `max_empty_share` of total train images.

    Returns `(kept_train_records, dropped_empty_count)`.
    """
    if max_empty_share <= 0.0:
        positives = [r for r in train_records if r.classes]
        return positives, len(train_records) - len(positives)
    if max_empty_share >= 1.0:
        return list(train_records), 0

    positives = [r for r in train_records if r.classes]
    empties = [r for r in train_records if not r.classes]

    # Solve E_max / (P + E_max) <= max_empty_share  =>  E_max = floor(P * s / (1 - s))
    max_empty_count = int((len(positives) * max_empty_share) / (1.0 - max_empty_share))
    if len(empties) <= max_empty_count:
        return list(train_records), 0

    rng = random.Random(seed)
    sampled_empties = list(empties)
    rng.shuffle(sampled_empties)
    kept_empties = sorted(sampled_empties[:max_empty_count], key=lambda r: (r.file_number, r.stem))
    dropped_count = len(empties) - len(kept_empties)

    combined = sorted(positives + kept_empties, key=lambda r: (r.file_number, r.stem))
    logger.info(
        "Capped empty train images: kept %d of %d (%d dropped, max_share=%.2f)",
        len(kept_empties),
        len(empties),
        dropped_count,
        max_empty_share,
    )
    return combined, dropped_count


def build_train_manifest_with_oversampling(
    train_records: list[SampleRecord],
    minority_class: int = 1,
    repeat_factor: int = 3,
) -> list[str]:
    """Build list of training image stems, repeating images containing `minority_class` (`D10`)."""
    manifest: list[str] = []
    repeats = max(1, int(repeat_factor))
    for rec in train_records:
        count = repeats if minority_class in rec.classes else 1
        manifest.extend([rec.stem] * count)
    return manifest


def materialize_yolo_splits(
    raw_images_dir: Path,
    yolo_labels_dir: Path,
    out_dataset_dir: Path,
    train_yaml_path: Path = DEFAULT_TRAIN_YAML,
) -> dict[str, int]:
    """Copy images and YOLO labels into `out_dataset_dir/{images,labels}/{train,val,test}`."""
    with train_yaml_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    split_cfg = cfg.get("split", {})
    seed = int(cfg.get("seed", 0))
    records = load_sample_records_from_labels(yolo_labels_dir)
    splits = split_records_contiguous_blocks(
        records,
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        block_size=int(split_cfg.get("block_size", 50)),
        seed=seed,
    )

    kept_train, dropped_empty = apply_train_background_cap(
        splits.train,
        max_empty_share=float(split_cfg.get("max_empty_train_share", 0.10)),
        seed=seed,
    )

    split_map = {"train": kept_train, "val": splits.val, "test": splits.test}
    for split_name, split_recs in split_map.items():
        img_out = out_dataset_dir / "images" / split_name
        lbl_out = out_dataset_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for rec in split_recs:
            src_img = raw_images_dir / f"{rec.stem}.jpg"
            src_lbl = yolo_labels_dir / f"{rec.stem}.txt"
            if src_img.is_file():
                shutil.copy2(src_img, img_out / src_img.name)
            if src_lbl.is_file():
                shutil.copy2(src_lbl, lbl_out / src_lbl.name)

    manifest = build_train_manifest_with_oversampling(
        kept_train,
        minority_class=1,
        repeat_factor=int(split_cfg.get("d10_repeat_factor", 3)),
    )
    manifest_path = out_dataset_dir / "train_oversampled.txt"
    manifest_lines = [f"images/train/{stem}.jpg" for stem in manifest]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return {
        "train_raw": len(splits.train),
        "train_kept": len(kept_train),
        "train_dropped_empty": dropped_empty,
        "val": len(splits.val),
        "test": len(splits.test),
        "train_oversampled_manifest": len(manifest),
    }
