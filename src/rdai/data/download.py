"""Locate or download the RDD2022 India dataset (and optional Japan/Czech subsets)."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = ROOT_DIR / "data" / "rdd2022"

# Official RDD2022 Figshare / sekilab dataset archive URLs
RDD2022_URLS: dict[str, str] = {
    "India": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_specific_data_CRDDC2022/RDD2022_India.zip",
    "Japan": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_specific_data_CRDDC2022/RDD2022_Japan.zip",
    "Czech": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_specific_data_CRDDC2022/RDD2022_Czech.zip",
}

# Published RDD2022 India reference counts from PRD Section 7 (Arya et al., 2022)
RDD2022_INDIA_REFERENCE_COUNTS: dict[str, Any] = {
    "train_images": 7706,
    "test_images": 1959,
    "class_counts": {
        "D00": 1555,
        "D10": 68,
        "D20": 2021,
        "D40": 3187,
    },
    "total_boxes": 6831,
}


def verify_rdd2022_layout(country_dir: Path) -> dict[str, Any]:
    """Verify that `country_dir` contains `train/images` and `train/annotations/xmls`."""
    train_img_dir = country_dir / "train" / "images"
    train_xml_dir = country_dir / "train" / "annotations" / "xmls"
    test_img_dir = country_dir / "test" / "images"

    n_train_imgs = len(list(train_img_dir.glob("*.jpg"))) if train_img_dir.is_dir() else 0
    n_train_xmls = len(list(train_xml_dir.glob("*.xml"))) if train_xml_dir.is_dir() else 0
    n_test_imgs = len(list(test_img_dir.glob("*.jpg"))) if test_img_dir.is_dir() else 0

    return {
        "valid": n_train_imgs > 0 and n_train_xmls > 0,
        "country_dir": str(country_dir),
        "train_images": n_train_imgs,
        "train_xmls": n_train_xmls,
        "test_images": n_test_imgs,
    }


def download_and_extract_country(
    country: str = "India",
    dest_root: Path = DEFAULT_RAW_DIR,
    timeout: float = 60.0,
) -> Path:
    """Download and extract an RDD2022 country archive if not already present."""
    if country not in RDD2022_URLS:
        raise ValueError(f"Unsupported RDD2022 country '{country}'. Choose from {list(RDD2022_URLS)}")

    country_dir = dest_root / country
    status = verify_rdd2022_layout(country_dir)
    if status["valid"]:
        logger.info("RDD2022 %s already extracted at %s (%s)", country, country_dir, status)
        return country_dir

    dest_root.mkdir(parents=True, exist_ok=True)
    zip_path = dest_root / f"RDD2022_{country}.zip"
    url = RDD2022_URLS[country]

    if not zip_path.is_file():
        logger.info("Downloading RDD2022 %s from %s -> %s", country, url, zip_path)
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with zip_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)

    logger.info("Extracting %s into %s", zip_path, dest_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_root)

    return country_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    layout = verify_rdd2022_layout(DEFAULT_RAW_DIR / "India")
    if layout["valid"]:
        print(f"Found local RDD2022 India dataset: {layout}")
    else:
        print(
            "RDD2022 India dataset not yet downloaded locally in data/rdd2022/India. "
            "Run download_and_extract_country('India') on Colab or locally to fetch."
        )
