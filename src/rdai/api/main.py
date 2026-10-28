"""FastAPI application factory and `/api/v1` endpoints (`PRD Section 11`, `M7`)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, File, Form, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from rdai.api.errors import APIError, register_exception_handlers
from rdai.api.schemas import (
    DetectionItemSchema,
    GeoInfoSchema,
    HealthResponse,
    ImageDetectResponse,
    JobStatusResponse,
    ModelInfoSchema,
    SegmentStatusPatchRequest,
    SeverityInfoSchema,
    VideoJobAcceptedResponse,
)
from rdai.api.validation import (
    sanitize_filename,
    save_annotated_and_thumbnail,
    validate_and_decode_image,
    validate_video_upload,
)
from rdai.config import AppConfig, load_app_config
from rdai.db.models import (
    ALLOWED_STATUS_TRANSITIONS,
    VALID_SEGMENT_STATUSES,
    DetectionRecord,
    ImageRecord,
    JobRecord,
    SegmentRecord,
)
from rdai.db.session import create_db_engine_and_session
from rdai.geo.exif import extract_exif_gps
from rdai.geo.osm import lookup_road_importance
from rdai.geo.track import interpolate_track_position, parse_track_auto
from rdai.priority.score import recluster_and_rank_segments

VALID_CLASS_CODES = {"D00", "D10", "D20", "D40"}
VALID_SEVERITY_LABELS = {"Low", "Medium", "High"}


class _LazyDefaultDetector:
    """Lazy wrapper around `RoadDamageDetector` or fallback stub when weights are not yet downloaded."""

    def __init__(self, config: AppConfig) -> None:
        self.model_name = config.model_name
        self.run_name = config.model_run
        self.default_conf = config.default_conf
        self._weights_path = Path(config.weights_path)
        self._detector: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._detector is not None or self._weights_path.exists()

    def _ensure_detector(self) -> Any | None:
        if self._detector is None and self._weights_path.exists():
            from rdai.inference.detector import RoadDamageDetector

            self._detector = RoadDamageDetector(weights_path=self._weights_path)
        return self._detector

    def detect_image(self, image: Any, conf_threshold: float | None = None) -> list[dict[str, Any]]:
        det = self._ensure_detector()
        if det is not None:
            try:
                return list(det.detect_image(image, conf=conf_threshold))
            except ModuleNotFoundError:
                return []
        return []

    def track_video(
        self,
        video_path: Path,
        conf_threshold: float | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        from rdai.inference.tracker import track_video_file

        if progress_callback:
            progress_callback(0.2)
        res = track_video_file(
            video_path=video_path,
            weights_path=self._weights_path,
            conf_threshold=conf_threshold,
        )
        if progress_callback:
            progress_callback(1.0)
        return res


def _parse_bbox_param(bbox: str | None) -> tuple[float, float, float, float] | None:
    """Parse `min_lng,min_lat,max_lng,max_lat` query parameter or raise 422 (`A05`, `S03`)."""
    if bbox is None or not bbox.strip():
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise APIError(
            422,
            "INVALID_BBOX",
            "bbox must have 4 comma-separated floats: min_lng,min_lat,max_lng,max_lat.",
        )
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise APIError(422, "INVALID_BBOX", f"bbox values must be numeric floats: {bbox}") from exc
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        raise APIError(422, "INVALID_BBOX", "bbox values must be finite numbers.")
    return vals[0], vals[1], vals[2], vals[3]


def _parse_since_param(since: str | None) -> datetime | None:
    """Parse ISO-8601 `since` parameter or raise 422 (`A05`, `S03`)."""
    if since is None or not since.strip():
        return None
    normalized = since.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise APIError(422, "INVALID_SINCE", f"Invalid ISO timestamp for 'since': {since}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _resolve_photo_coordinates(
    raw_bytes: bytes,
    req_lat: float | None,
    req_lng: float | None,
    india_bbox: dict[str, float],
) -> tuple[float | None, float | None, str, list[str]]:
    """Resolve photo coordinates from request params or EXIF and check India bounding box (`A04`)."""
    if (req_lat is None) ^ (req_lng is None):
        raise APIError(
            422, "INVALID_COORDINATES", "Both 'lat' and 'lng' must be provided together."
        )

    lat: float | None = None
    lng: float | None = None
    source = "none"

    if req_lat is not None and req_lng is not None:
        if (
            math.isnan(req_lat)
            or math.isnan(req_lng)
            or not (-90.0 <= req_lat <= 90.0)
            or not (-180.0 <= req_lng <= 180.0)
        ):
            raise APIError(
                422,
                "INVALID_COORDINATES",
                f"Coordinates out of valid range: lat={req_lat}, lng={req_lng}.",
            )
        lat = float(req_lat)
        lng = float(req_lng)
        source = "request"
    else:
        exif_coords = extract_exif_gps(raw_bytes)
        if exif_coords is not None:
            lat, lng = float(exif_coords[0]), float(exif_coords[1])
            source = "exif"

    warnings: list[str] = []
    if lat is not None and lng is not None:
        in_india = (
            india_bbox["lat_min"] <= lat <= india_bbox["lat_max"]
            and india_bbox["lng_min"] <= lng <= india_bbox["lng_max"]
        )
        if not in_india:
            warnings.append("outside_india")

    return lat, lng, source, warnings


def _recluster_db_segments(
    session: Session,
    osm_lookup_fn: Callable[[float, float], dict[str, Any]],
) -> int:
    """Recluster all located detections in the database and update `segments` and `detections` rows."""
    all_dets = list(session.scalars(select(DetectionRecord).order_by(DetectionRecord.id)).all())
    all_segs = list(session.scalars(select(SegmentRecord).order_by(SegmentRecord.id)).all())

    det_dicts = [
        {
            "id": d.id,
            "lat": d.lat,
            "lng": d.lng,
            "severity_score": d.severity_score,
            "confidence": d.confidence,
            "segment_id": d.segment_id,
        }
        for d in all_dets
    ]
    seg_dicts = [
        {
            "id": s.id,
            "centroid_lat": s.centroid_lat,
            "centroid_lng": s.centroid_lng,
            "road_class": s.road_class,
            "status": s.status,
        }
        for s in all_segs
    ]

    reclustered = recluster_and_rank_segments(
        detections=det_dicts,
        existing_segments=seg_dicts,
        osm_lookup_fn=osm_lookup_fn,
    )

    new_seg_list = reclustered["segments"]
    updated_det_list = reclustered["detections"]

    det_seg_by_id = {int(d["id"]): d.get("segment_id") for d in updated_det_list}
    for d_row in all_dets:
        d_row.segment_id = det_seg_by_id.get(d_row.id)

    existing_seg_map = {s.id: s for s in all_segs}
    active_seg_ids: set[int] = set()
    now_utc = datetime.now(UTC)

    for s_data in new_seg_list:
        sid = int(s_data["id"])
        active_seg_ids.add(sid)
        if sid in existing_seg_map:
            row = existing_seg_map[sid]
            row.centroid_lat = float(s_data["centroid_lat"])
            row.centroid_lng = float(s_data["centroid_lng"])
            row.n_detections = int(s_data["n_detections"])
            row.max_severity = float(s_data["max_severity"])
            row.mean_confidence = float(s_data["mean_confidence"])
            row.road_class = str(s_data["road_class"])
            row.priority_score = float(s_data["priority_score"])
            row.priority_label = str(s_data["priority_label"])
            row.terms_json = json.dumps(s_data["terms"])
            row.weighted_terms_json = json.dumps(s_data["weighted_terms"])
            row.status = str(s_data["status"])
            row.updated_at = now_utc
        else:
            session.add(
                SegmentRecord(
                    id=sid,
                    centroid_lat=float(s_data["centroid_lat"]),
                    centroid_lng=float(s_data["centroid_lng"]),
                    n_detections=int(s_data["n_detections"]),
                    max_severity=float(s_data["max_severity"]),
                    mean_confidence=float(s_data["mean_confidence"]),
                    road_class=str(s_data["road_class"]),
                    priority_score=float(s_data["priority_score"]),
                    priority_label=str(s_data["priority_label"]),
                    terms_json=json.dumps(s_data["terms"]),
                    weighted_terms_json=json.dumps(s_data["weighted_terms"]),
                    status=str(s_data["status"]),
                    updated_at=now_utc,
                )
            )

    for old_id, old_row in existing_seg_map.items():
        if old_id not in active_seg_ids:
            session.delete(old_row)

    return len(new_seg_list)


def _serialize_image_response(
    img_row: ImageRecord,
    det_rows: list[DetectionRecord],
    cfg: AppConfig,
    conf_used: float,
) -> ImageDetectResponse:
    warnings = json.loads(img_row.geo_warnings_json or "[]")
    return ImageDetectResponse(
        image_id=img_row.id,
        width=img_row.width,
        height=img_row.height,
        geo=GeoInfoSchema(
            lat=img_row.lat,
            lng=img_row.lng,
            source=img_row.geo_source,
            warnings=warnings,
        ),
        annotated_url=img_row.annotated_url,
        thumb_url=img_row.thumb_url,
        detections=[
            DetectionItemSchema(
                id=d.id,
                class_code=d.class_code,
                class_name=d.class_name,
                confidence=round(float(d.confidence), 4),
                box=[round(d.x1, 2), round(d.y1, 2), round(d.x2, 2), round(d.y2, 2)],
                severity=SeverityInfoSchema(
                    score=round(float(d.severity_score), 2),
                    label=d.severity_label,
                    extent=round(float(d.extent), 4),
                    measure=d.measure,
                ),
                segment_id=d.segment_id,
            )
            for d in det_rows
        ],
        model=ModelInfoSchema(
            name=cfg.model_name,
            run=cfg.model_run,
            conf=round(float(conf_used), 4),
        ),
    )


def create_app(
    config: AppConfig | None = None,
    detector: Any | None = None,
    osm_lookup_fn: Callable[[float, float], dict[str, Any]] | None = None,
) -> FastAPI:
    """Create and configure the Road Damage AI FastAPI application (`PRD Section 11`)."""
    cfg = config if config is not None else load_app_config()
    media_root = Path(cfg.media_dir).resolve()
    media_root.mkdir(parents=True, exist_ok=True)

    _engine, session_factory = create_db_engine_and_session(cfg.database_url)
    write_lock = threading.Lock()

    active_detector = detector if detector is not None else _LazyDefaultDetector(cfg)
    active_osm_fn = osm_lookup_fn if osm_lookup_fn is not None else lookup_road_importance

    app = FastAPI(title=cfg.title, version=cfg.version)
    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = cfg.api_prefix.rstrip("/")

    @app.get(f"{prefix}/health", response_model=HealthResponse)
    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    def get_health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=bool(getattr(active_detector, "is_loaded", True)),
            version=cfg.version,
            model_name=cfg.model_name,
            run=cfg.model_run,
        )

    @app.post(f"{prefix}/detect/image", response_model=ImageDetectResponse)
    async def detect_image_endpoint(
        file: Annotated[UploadFile, File(...)],
        lat: Annotated[float | None, Form()] = None,
        lng: Annotated[float | None, Form()] = None,
        conf: Annotated[float | None, Form()] = None,
    ) -> ImageDetectResponse:
        raw_bytes = await file.read()
        safe_name = sanitize_filename(file.filename)
        pil_img, _fmt = validate_and_decode_image(
            raw_bytes=raw_bytes,
            filename=safe_name,
            max_photo_mb=cfg.max_photo_mb,
            max_megapixels=cfg.max_megapixels,
            allowed_formats=cfg.allowed_image_formats,
        )

        res_lat, res_lng, geo_source, geo_warnings = _resolve_photo_coordinates(
            raw_bytes=raw_bytes,
            req_lat=lat,
            req_lng=lng,
            india_bbox=cfg.india_bbox,
        )

        conf_used = float(conf) if conf is not None else cfg.default_conf
        if not (0.0 < conf_used <= 1.0):
            raise APIError(422, "INVALID_CONF", "Confidence threshold must be in (0.0, 1.0].")

        sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

        with write_lock, session_factory() as session:
            existing_img = session.scalar(
                select(ImageRecord).where(ImageRecord.sha256 == sha256_hex)
            )
            if existing_img is not None:
                existing_dets = list(
                    session.scalars(
                        select(DetectionRecord)
                        .where(DetectionRecord.image_id == existing_img.id)
                        .order_by(DetectionRecord.id)
                    ).all()
                )
                return _serialize_image_response(existing_img, existing_dets, cfg, conf_used)

            raw_detections = active_detector.detect_image(pil_img, conf_threshold=conf_used)

            img_row = ImageRecord(
                sha256=sha256_hex,
                source_filename=safe_name,
                width=pil_img.width,
                height=pil_img.height,
                lat=res_lat,
                lng=res_lng,
                geo_source=geo_source,
                geo_warnings_json=json.dumps(geo_warnings),
                annotated_url="",
                thumb_url="",
            )
            session.add(img_row)
            session.flush()

            img_row.annotated_url = f"/media/annotated/{img_row.id}.jpg"
            img_row.thumb_url = f"/media/thumbs/{img_row.id}.jpg"

            save_annotated_and_thumbnail(
                image=pil_img,
                detections=raw_detections,
                image_id=img_row.id,
                media_dir=media_root,
                annotated_max_edge_px=cfg.annotated_max_edge_px,
                thumb_max_edge_px=cfg.thumb_max_edge_px,
            )

            created_dets: list[DetectionRecord] = []
            for d in raw_detections:
                box = d["box"]
                sev = d["severity"]
                det_row = DetectionRecord(
                    image_id=img_row.id,
                    class_code=str(d["class_code"]),
                    class_name=str(d["class_name"]),
                    confidence=float(d["confidence"]),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    extent=float(sev["extent"]),
                    measure=str(sev["measure"]),
                    severity_score=float(sev["score"]),
                    severity_label=str(sev["label"]),
                    lat=res_lat,
                    lng=res_lng,
                    segment_id=None,
                )
                session.add(det_row)
                created_dets.append(det_row)

            session.flush()
            if res_lat is not None and res_lng is not None and created_dets:
                _recluster_db_segments(session, active_osm_fn)

            session.commit()
            for det_row in created_dets:
                session.refresh(det_row)
            return _serialize_image_response(img_row, created_dets, cfg, conf_used)

    def _run_video_background_job(
        job_id: int,
        video_bytes: bytes,
        safe_video_name: str,
        track_content: str | None,
        track_name: str | None,
        offset_s: float,
        conf_used: float,
    ) -> None:
        tmp_dir = media_root / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_video_path = tmp_dir / f"job_{job_id}_{safe_video_name}"

        def _update_progress(p: float) -> None:
            with write_lock, session_factory() as s:
                j = s.get(JobRecord, job_id)
                if j is not None:
                    j.status = "running"
                    j.progress = round(min(0.99, max(0.0, float(p))), 2)
                    s.commit()

        try:
            tmp_video_path.write_bytes(video_bytes)
            with write_lock, session_factory() as s:
                j = s.get(JobRecord, job_id)
                if j is not None:
                    j.status = "running"
                    j.progress = 0.1
                    s.commit()

            track_points: list[dict[str, float]] = []
            if track_content:
                track_points = parse_track_auto(track_content, filename=track_name or "track.csv")

            deduped = active_detector.track_video(
                video_path=tmp_video_path,
                conf_threshold=conf_used,
                progress_callback=_update_progress,
            )

            with write_lock, session_factory() as s:
                for item in deduped:
                    det_lat: float | None = None
                    det_lng: float | None = None
                    if track_points:
                        interp = interpolate_track_position(
                            track_points,
                            timestamp_s=float(item.get("timestamp_s", 0.0)),
                            offset_s=float(offset_s),
                            tolerance_s=cfg.track_tolerance_seconds,
                        )
                        if interp is not None:
                            det_lat, det_lng = interp[0], interp[1]

                    box = item["box"]
                    sev = item["severity"]
                    s.add(
                        DetectionRecord(
                            image_id=None,
                            job_id=job_id,
                            frame_idx=int(item.get("frame_idx", 0)),
                            track_id=int(item.get("track_id", 0)),
                            class_code=str(item["class_code"]),
                            class_name=str(item["class_name"]),
                            confidence=float(item["confidence"]),
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                            extent=float(sev["extent"]),
                            measure=str(sev["measure"]),
                            severity_score=float(sev["score"]),
                            severity_label=str(sev["label"]),
                            lat=det_lat,
                            lng=det_lng,
                            segment_id=None,
                        )
                    )
                s.flush()
                n_segs = _recluster_db_segments(s, active_osm_fn)
                j = s.get(JobRecord, job_id)
                if j is not None:
                    j.status = "done"
                    j.progress = 1.0
                    j.summary_json = json.dumps(
                        {
                            "unique_detections": len(deduped),
                            "segments_total": n_segs,
                        }
                    )
                    j.finished_at = datetime.now(UTC)
                s.commit()
        except Exception as exc:
            with write_lock, session_factory() as s:
                j = s.get(JobRecord, job_id)
                if j is not None:
                    j.status = "failed"
                    j.error = str(exc)
                    j.finished_at = datetime.now(UTC)
                    s.commit()
        finally:
            if tmp_video_path.exists():
                tmp_video_path.unlink(missing_ok=True)

    @app.post(f"{prefix}/detect/video", status_code=202, response_model=VideoJobAcceptedResponse)
    async def detect_video_endpoint(
        background_tasks: BackgroundTasks,
        file: Annotated[UploadFile, File(...)],
        track: Annotated[UploadFile | None, File()] = None,
        offset_s: Annotated[float, Form()] = 0.0,
        conf: Annotated[float | None, Form()] = None,
    ) -> VideoJobAcceptedResponse:
        video_bytes = await file.read()
        safe_video_name = validate_video_upload(
            raw_bytes=video_bytes,
            filename=file.filename,
            max_video_mb=cfg.max_video_mb,
        )

        track_content: str | None = None
        track_name: str | None = None
        if track is not None:
            raw_track = await track.read()
            track_content = raw_track.decode("utf-8", errors="replace")
            track_name = sanitize_filename(track.filename)

        conf_used = float(conf) if conf is not None else cfg.default_conf

        with write_lock, session_factory() as session:
            job = JobRecord(kind="video", status="queued", progress=0.0)
            session.add(job)
            session.commit()
            session.refresh(job)
            job_id = job.id

        background_tasks.add_task(
            _run_video_background_job,
            job_id,
            video_bytes,
            safe_video_name,
            track_content,
            track_name,
            float(offset_s),
            conf_used,
        )
        return VideoJobAcceptedResponse(job_id=job_id, status="queued")

    @app.get(f"{prefix}/jobs/{{job_id}}", response_model=JobStatusResponse)
    def get_job_status(job_id: int) -> JobStatusResponse:
        with session_factory() as session:
            job = session.get(JobRecord, job_id)
            if job is None:
                raise APIError(404, "JOB_NOT_FOUND", f"Job {job_id} not found.")
            return JobStatusResponse(
                id=job.id,
                kind=job.kind,
                status=job.status,
                progress=round(float(job.progress), 2),
                error=job.error,
                summary=json.loads(job.summary_json or "{}"),
            )

    @app.get(f"{prefix}/detections")
    def list_detections(
        class_filter: str | None = Query(default=None, alias="class"),
        class_code: str | None = Query(default=None),
        severity: str | None = Query(default=None),
        status: str | None = Query(default=None),
        bbox: str | None = Query(default=None),
        since: str | None = Query(default=None),
        limit: int = Query(default=20),
        offset: int = Query(default=0),
        sort: str = Query(default="severity_desc"),
    ) -> dict[str, Any]:
        if limit < 1 or limit > cfg.max_limit:
            raise APIError(
                422,
                "INVALID_LIMIT",
                f"limit must be between 1 and {cfg.max_limit}.",
            )
        if offset < 0:
            raise APIError(422, "INVALID_OFFSET", "offset must be >= 0.")

        cls_val = class_filter if class_filter is not None else class_code
        if cls_val is not None and cls_val not in VALID_CLASS_CODES:
            raise APIError(422, "INVALID_CLASS", f"Unsupported class code: {cls_val}")
        if severity is not None and severity not in VALID_SEVERITY_LABELS:
            raise APIError(422, "INVALID_SEVERITY", f"Unsupported severity label: {severity}")
        if status is not None and status not in VALID_SEGMENT_STATUSES:
            raise APIError(422, "INVALID_STATUS", f"Unsupported status filter: {status}")

        bbox_tuple = _parse_bbox_param(bbox)
        since_dt = _parse_since_param(since)

        with session_factory() as session:
            stmt = select(DetectionRecord)
            if cls_val is not None:
                stmt = stmt.where(DetectionRecord.class_code == cls_val)
            if severity is not None:
                stmt = stmt.where(DetectionRecord.severity_label == severity)
            if status is not None:
                seg_ids_stmt = select(SegmentRecord.id).where(SegmentRecord.status == status)
                stmt = stmt.where(DetectionRecord.segment_id.in_(seg_ids_stmt))
            if bbox_tuple is not None:
                min_lng, min_lat, max_lng, max_lat = bbox_tuple
                stmt = stmt.where(
                    DetectionRecord.lat.is_not(None),
                    DetectionRecord.lng.is_not(None),
                    DetectionRecord.lat >= min_lat,
                    DetectionRecord.lat <= max_lat,
                    DetectionRecord.lng >= min_lng,
                    DetectionRecord.lng <= max_lng,
                )
            if since_dt is not None:
                stmt = stmt.where(DetectionRecord.created_at >= since_dt)

            if sort == "severity_asc":
                stmt = stmt.order_by(DetectionRecord.severity_score.asc(), DetectionRecord.id.asc())
            elif sort == "confidence_desc":
                stmt = stmt.order_by(DetectionRecord.confidence.desc(), DetectionRecord.id.asc())
            elif sort == "created_desc":
                stmt = stmt.order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc())
            elif sort == "id_asc":
                stmt = stmt.order_by(DetectionRecord.id.asc())
            else:
                stmt = stmt.order_by(
                    DetectionRecord.severity_score.desc(), DetectionRecord.id.asc()
                )

            all_matching = list(session.scalars(stmt).all())
            sliced = all_matching[offset : offset + limit]

            return {
                "total": len(all_matching),
                "limit": limit,
                "offset": offset,
                "items": [
                    {
                        "id": d.id,
                        "image_id": d.image_id,
                        "class_code": d.class_code,
                        "class_name": d.class_name,
                        "confidence": round(float(d.confidence), 4),
                        "box": [round(d.x1, 2), round(d.y1, 2), round(d.x2, 2), round(d.y2, 2)],
                        "severity": {
                            "score": round(float(d.severity_score), 2),
                            "label": d.severity_label,
                            "extent": round(float(d.extent), 4),
                            "measure": d.measure,
                        },
                        "lat": d.lat,
                        "lng": d.lng,
                        "segment_id": d.segment_id,
                    }
                    for d in sliced
                ],
            }

    @app.get(f"{prefix}/detections/{{detection_id}}")
    def get_detection(detection_id: int) -> dict[str, Any]:
        with session_factory() as session:
            d = session.get(DetectionRecord, detection_id)
            if d is None:
                raise APIError(404, "DETECTION_NOT_FOUND", f"Detection {detection_id} not found.")
            img = session.get(ImageRecord, d.image_id) if d.image_id is not None else None
            return {
                "id": d.id,
                "image_id": d.image_id,
                "annotated_url": img.annotated_url if img else None,
                "thumb_url": img.thumb_url if img else None,
                "class_code": d.class_code,
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "box": [round(d.x1, 2), round(d.y1, 2), round(d.x2, 2), round(d.y2, 2)],
                "severity": {
                    "score": round(float(d.severity_score), 2),
                    "label": d.severity_label,
                    "extent": round(float(d.extent), 4),
                    "measure": d.measure,
                },
                "lat": d.lat,
                "lng": d.lng,
                "segment_id": d.segment_id,
            }

    @app.get(f"{prefix}/segments")
    def list_segments(
        label: str | None = Query(default=None),
        status: str | None = Query(default=None),
        bbox: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if label is not None and label not in VALID_SEVERITY_LABELS:
            raise APIError(422, "INVALID_PRIORITY_LABEL", f"Unsupported priority label: {label}")
        if status is not None and status not in VALID_SEGMENT_STATUSES:
            raise APIError(422, "INVALID_STATUS", f"Unsupported segment status: {status}")

        bbox_tuple = _parse_bbox_param(bbox)

        with session_factory() as session:
            stmt = select(SegmentRecord).order_by(
                SegmentRecord.priority_score.desc(),
                SegmentRecord.id.asc(),
            )
            if label is not None:
                stmt = stmt.where(SegmentRecord.priority_label == label)
            if status is not None:
                stmt = stmt.where(SegmentRecord.status == status)
            if bbox_tuple is not None:
                min_lng, min_lat, max_lng, max_lat = bbox_tuple
                stmt = stmt.where(
                    SegmentRecord.centroid_lat >= min_lat,
                    SegmentRecord.centroid_lat <= max_lat,
                    SegmentRecord.centroid_lng >= min_lng,
                    SegmentRecord.centroid_lng <= max_lng,
                )

            rows = list(session.scalars(stmt).all())
            return {
                "total": len(rows),
                "items": [
                    {
                        "id": s.id,
                        "centroid_lat": s.centroid_lat,
                        "centroid_lng": s.centroid_lng,
                        "n_detections": s.n_detections,
                        "max_severity": s.max_severity,
                        "mean_confidence": s.mean_confidence,
                        "road_class": s.road_class,
                        "priority_score": s.priority_score,
                        "priority_label": s.priority_label,
                        "terms": json.loads(s.terms_json or "{}"),
                        "weighted_terms": json.loads(s.weighted_terms_json or "{}"),
                        "status": s.status,
                    }
                    for s in rows
                ],
            }

    @app.get(f"{prefix}/segments/{{segment_id}}")
    def get_segment(segment_id: int) -> dict[str, Any]:
        with session_factory() as session:
            s = session.get(SegmentRecord, segment_id)
            if s is None:
                raise APIError(404, "SEGMENT_NOT_FOUND", f"Segment {segment_id} not found.")
            members = list(
                session.scalars(
                    select(DetectionRecord)
                    .where(DetectionRecord.segment_id == segment_id)
                    .order_by(DetectionRecord.severity_score.desc())
                ).all()
            )
            return {
                "id": s.id,
                "centroid_lat": s.centroid_lat,
                "centroid_lng": s.centroid_lng,
                "n_detections": s.n_detections,
                "max_severity": s.max_severity,
                "mean_confidence": s.mean_confidence,
                "road_class": s.road_class,
                "priority_score": s.priority_score,
                "priority_label": s.priority_label,
                "terms": json.loads(s.terms_json or "{}"),
                "weighted_terms": json.loads(s.weighted_terms_json or "{}"),
                "status": s.status,
                "detections": [
                    {
                        "id": m.id,
                        "image_id": m.image_id,
                        "class_code": m.class_code,
                        "class_name": m.class_name,
                        "confidence": round(float(m.confidence), 4),
                        "severity_score": round(float(m.severity_score), 2),
                        "severity_label": m.severity_label,
                        "lat": m.lat,
                        "lng": m.lng,
                    }
                    for m in members
                ],
            }

    @app.patch(f"{prefix}/segments/{{segment_id}}/status")
    def patch_segment_status(
        segment_id: int,
        payload: SegmentStatusPatchRequest,
    ) -> dict[str, Any]:
        target_status = payload.status.strip().lower()
        if target_status not in VALID_SEGMENT_STATUSES:
            raise APIError(
                422,
                "INVALID_STATUS",
                f"Unknown segment status '{payload.status}'. Must be one of {sorted(VALID_SEGMENT_STATUSES)}.",
            )

        with write_lock, session_factory() as session:
            seg = session.get(SegmentRecord, segment_id)
            if seg is None:
                raise APIError(404, "SEGMENT_NOT_FOUND", f"Segment {segment_id} not found.")

            current = seg.status
            allowed_next = ALLOWED_STATUS_TRANSITIONS.get(current, set())
            if target_status not in allowed_next:
                raise APIError(
                    409,
                    "ILLEGAL_STATUS_TRANSITION",
                    f"Cannot transition segment {segment_id} from '{current}' to '{target_status}'.",
                )

            seg.status = target_status
            seg.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(seg)
            return {
                "id": seg.id,
                "status": seg.status,
                "priority_score": seg.priority_score,
                "priority_label": seg.priority_label,
            }

    @app.get(f"{prefix}/map/geojson")
    def get_map_geojson(
        layer: str = Query(default="detections"),
        bbox: str | None = Query(default=None),
    ) -> dict[str, Any]:
        if layer not in {"detections", "segments"}:
            raise APIError(422, "INVALID_LAYER", "layer must be 'detections' or 'segments'.")

        bbox_tuple = _parse_bbox_param(bbox)

        with session_factory() as session:
            features: list[dict[str, Any]] = []
            if layer == "detections":
                stmt = select(DetectionRecord).where(
                    DetectionRecord.lat.is_not(None),
                    DetectionRecord.lng.is_not(None),
                )
                if bbox_tuple is not None:
                    min_lng, min_lat, max_lng, max_lat = bbox_tuple
                    stmt = stmt.where(
                        DetectionRecord.lat >= min_lat,
                        DetectionRecord.lat <= max_lat,
                        DetectionRecord.lng >= min_lng,
                        DetectionRecord.lng <= max_lng,
                    )
                for d in session.scalars(stmt).all():
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [float(d.lng), float(d.lat)],
                            },
                            "properties": {
                                "id": d.id,
                                "image_id": d.image_id,
                                "class_code": d.class_code,
                                "class_name": d.class_name,
                                "confidence": round(float(d.confidence), 4),
                                "severity_score": round(float(d.severity_score), 2),
                                "severity_label": d.severity_label,
                                "segment_id": d.segment_id,
                            },
                        }
                    )
            else:
                stmt_s = select(SegmentRecord).order_by(SegmentRecord.priority_score.desc())
                if bbox_tuple is not None:
                    min_lng, min_lat, max_lng, max_lat = bbox_tuple
                    stmt_s = stmt_s.where(
                        SegmentRecord.centroid_lat >= min_lat,
                        SegmentRecord.centroid_lat <= max_lat,
                        SegmentRecord.centroid_lng >= min_lng,
                        SegmentRecord.centroid_lng <= max_lng,
                    )
                for s in session.scalars(stmt_s).all():
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [float(s.centroid_lng), float(s.centroid_lat)],
                            },
                            "properties": {
                                "id": s.id,
                                "n_detections": s.n_detections,
                                "max_severity": s.max_severity,
                                "road_class": s.road_class,
                                "priority_score": s.priority_score,
                                "priority_label": s.priority_label,
                                "status": s.status,
                            },
                        }
                    )

            return {
                "type": "FeatureCollection",
                "features": features,
            }

    @app.get(f"{prefix}/stats")
    def get_stats() -> dict[str, Any]:
        with session_factory() as session:
            imgs = list(session.scalars(select(ImageRecord)).all())
            dets = list(session.scalars(select(DetectionRecord)).all())
            segs = list(session.scalars(select(SegmentRecord)).all())

            by_class = {c: 0 for c in sorted(VALID_CLASS_CODES)}
            by_severity = {lbl: 0 for lbl in ("Low", "Medium", "High")}
            by_status = {st: 0 for st in ("new", "verified", "scheduled", "fixed", "rejected")}

            for d in dets:
                if d.class_code in by_class:
                    by_class[d.class_code] += 1
                if d.severity_label in by_severity:
                    by_severity[d.severity_label] += 1
            for s in segs:
                if s.status in by_status:
                    by_status[s.status] += 1

            return {
                "total_images": len(imgs),
                "total_detections": len(dets),
                "total_segments": len(segs),
                "by_class": by_class,
                "by_severity": by_severity,
                "by_status": by_status,
            }

    @app.get(f"{prefix}/export/csv")
    def export_detections_csv() -> Response:
        with session_factory() as session:
            dets = list(session.scalars(select(DetectionRecord).order_by(DetectionRecord.id)).all())
            imgs_by_id = {img.id: img for img in session.scalars(select(ImageRecord)).all()}

            buf = io.StringIO()
            writer = csv.writer(buf, lineterminator="\n")
            writer.writerow(
                [
                    "id",
                    "image_id",
                    "class_code",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "extent",
                    "measure",
                    "severity_score",
                    "severity_label",
                    "lat",
                    "lng",
                    "segment_id",
                    "source_filename",
                ]
            )
            for d in dets:
                src_img = imgs_by_id.get(d.image_id) if d.image_id is not None else None
                writer.writerow(
                    [
                        d.id,
                        d.image_id if d.image_id is not None else "",
                        d.class_code,
                        d.class_name,
                        round(float(d.confidence), 4),
                        round(float(d.x1), 2),
                        round(float(d.y1), 2),
                        round(float(d.x2), 2),
                        round(float(d.y2), 2),
                        round(float(d.extent), 4),
                        d.measure,
                        round(float(d.severity_score), 2),
                        d.severity_label,
                        d.lat if d.lat is not None else "",
                        d.lng if d.lng is not None else "",
                        d.segment_id if d.segment_id is not None else "",
                        src_img.source_filename if src_img is not None else "",
                    ]
                )

            return Response(
                content=buf.getvalue(),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": 'attachment; filename="detections.csv"'},
            )

    @app.get("/media/{subpath:path}")
    @app.get(f"{prefix}/media/{{subpath:path}}", include_in_schema=False)
    def serve_media_file(subpath: str) -> FileResponse:
        if ".." in subpath or subpath.startswith(("/", "\\")):
            raise APIError(400, "INVALID_MEDIA_PATH", "Path traversal is not allowed.")
        candidate = (media_root / subpath).resolve()
        if media_root not in candidate.parents or not candidate.is_file():
            raise APIError(404, "MEDIA_NOT_FOUND", "Requested media file was not found.")
        return FileResponse(candidate, media_type="image/jpeg")

    return app


app = create_app()
