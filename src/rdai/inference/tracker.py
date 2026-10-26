"""ByteTrack multi-object video tracking and per-track deduplication (`FR-A2`, `I02`)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rdai.inference.detector import CLASS_CODE_TO_NAME
from rdai.severity.rules import grade_detection_box, load_severity_config
from rdai.train.train import EXPECTED_CLASS_MAP


def deduplicate_tracked_frames(
    frame_detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate per-frame tracked detections so each unique `track_id` is counted once.

    For each `track_id` (or `(track_id, class_code)`), retains the frame detection with the
    highest `confidence * extent` product. Detections without a `track_id` are kept as-is.
    """
    best_by_track: dict[tuple[int, str], dict[str, Any]] = {}
    untracked: list[dict[str, Any]] = []

    for det in frame_detections:
        track_id = det.get("track_id")
        if track_id is None:
            untracked.append(det)
            continue

        key = (int(track_id), str(det["class_code"]))
        score_cur = float(det.get("confidence", 0.0)) * max(
            float(det.get("severity", {}).get("extent", 0.0)), 1e-4
        )
        existing = best_by_track.get(key)
        if existing is None:
            best_by_track[key] = det
        else:
            score_old = float(existing.get("confidence", 0.0)) * max(
                float(existing.get("severity", {}).get("extent", 0.0)), 1e-4
            )
            if score_cur > score_old:
                best_by_track[key] = det

    deduped = list(best_by_track.values()) + untracked
    deduped.sort(key=lambda d: (int(d.get("frame_idx", 0)), int(d.get("track_id") or 0)))
    return deduped


def track_video_file(
    video_path: Path,
    model: Any,
    conf: float = 0.25,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Run ByteTrack across frames of `video_path` and return deduplicated detections."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0.0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    severity_cfg = load_severity_config()

    raw_frame_dets: list[dict[str, Any]] = []
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h_px, w_px = frame.shape[:2]

            if hasattr(model, "track"):
                results = model.track(
                    frame,
                    conf=conf,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                results = model.predict(frame, conf=conf, verbose=False)

            if results:
                pred = results[0]
                boxes = getattr(pred, "boxes", None)
                if boxes is not None and len(boxes) > 0:
                    xyxy = np.asarray(
                        boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy, dtype=float
                    )
                    confs = np.asarray(
                        boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf, dtype=float
                    )
                    clss = np.asarray(
                        boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls, dtype=int
                    )
                    raw_ids = getattr(boxes, "id", None)
                    track_ids = (
                        np.asarray(
                            raw_ids.cpu() if hasattr(raw_ids, "cpu") else raw_ids, dtype=int
                        ).tolist()
                        if raw_ids is not None
                        else [1] * len(confs)
                    )

                    for box_coords, conf_val, cls_id, tid in zip(
                        xyxy, confs, clss, track_ids, strict=True
                    ):
                        if float(conf_val) < conf:
                            continue
                        cls_code = EXPECTED_CLASS_MAP.get(int(cls_id))
                        if cls_code is None:
                            continue
                        clamped = [
                            round(float(np.clip(box_coords[0], 0.0, float(w_px))), 2),
                            round(float(np.clip(box_coords[1], 0.0, float(h_px))), 2),
                            round(float(np.clip(box_coords[2], 0.0, float(w_px))), 2),
                            round(float(np.clip(box_coords[3], 0.0, float(h_px))), 2),
                        ]
                        sev = grade_detection_box(
                            cls_code, clamped, img_w=w_px, img_h=h_px, config=severity_cfg
                        )
                        raw_frame_dets.append(
                            {
                                "frame_idx": frame_idx,
                                "timestamp_s": round(frame_idx / fps, 3),
                                "track_id": int(tid),
                                "class_id": int(cls_id),
                                "class_code": cls_code,
                                "class_name": CLASS_CODE_TO_NAME[cls_code],
                                "confidence": round(float(conf_val), 4),
                                "box": clamped,
                                "severity": sev,
                            }
                        )

            frame_idx += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(min(0.99, frame_idx / total_frames))
    finally:
        cap.release()

    if frame_idx == 0:
        raise ValueError(f"Video file contains zero readable frames: {video_path}")

    deduped = deduplicate_tracked_frames(raw_frame_dets)
    if progress_callback is not None:
        progress_callback(1.0)

    return {
        "frames_processed": frame_idx,
        "fps": fps,
        "raw_detections": len(raw_frame_dets),
        "unique_detections": deduped,
    }
