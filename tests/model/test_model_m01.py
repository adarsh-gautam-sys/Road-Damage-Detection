"""Model test M01: Trained weights load and class names are D00, D10, D20, D40 in exact order."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdai.train.train import load_app_model_config, verify_checkpoint_classes


@pytest.mark.slow
def test_m01_weights_load_and_class_names_match(project_root: Path) -> None:
    """M01: The weights file loads and class names are D00, D10, D20, D40 in that order."""
    model_cfg = load_app_model_config(project_root / "configs" / "app.yaml")
    weights_path = project_root / str(model_cfg["weights_path"])
    if not weights_path.is_file():
        pytest.skip(
            f"Trained checkpoint not found at {weights_path}. "
            "Run Experiment E1 in notebooks/02_train_colab.ipynb on Google AI Pro Colab "
            "and place best.pt in runs/e1_yolo11s_640/weights/best.pt."
        )

    names = verify_checkpoint_classes(weights_path)
    assert names == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
