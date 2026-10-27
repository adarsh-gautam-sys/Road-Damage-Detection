"""SQLAlchemy 2.0 ORM models for `images`, `detections`, `segments`, and `jobs` (`PRD Section 11`)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for Road Damage AI SQLite persistence."""


VALID_SEGMENT_STATUSES = {"new", "verified", "scheduled", "fixed", "rejected"}

ALLOWED_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "new": {"verified", "rejected"},
    "verified": {"scheduled", "rejected"},
    "scheduled": {"fixed"},
    "fixed": set(),
    "rejected": set(),
}


class ImageRecord(Base):
    """Uploaded photo metadata and resolved geographic location (`images` table)."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    source_filename: Mapped[str] = mapped_column(String(255), default="upload.jpg", nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    captured_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    geo_source: Mapped[str] = mapped_column(String(32), default="none", nullable=False)
    geo_warnings_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)
    annotated_url: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    thumb_url: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )


class DetectionRecord(Base):
    """Single detected road damage bounding box, severity score, and segment link (`detections` table)."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    job_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    frame_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_code: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    extent: Mapped[float] = mapped_column(Float, nullable=False)
    measure: Mapped[str] = mapped_column(String(32), nullable=False)
    severity_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity_label: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )


class SegmentRecord(Base):
    """Clustered road segment with 4-term maintenance priority score and repair status (`segments` table)."""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    centroid_lat: Mapped[float] = mapped_column(Float, nullable=False)
    centroid_lng: Mapped[float] = mapped_column(Float, nullable=False)
    n_detections: Mapped[int] = mapped_column(Integer, nullable=False)
    max_severity: Mapped[float] = mapped_column(Float, nullable=False)
    mean_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    road_class: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    priority_label: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    terms_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    weighted_terms_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="new", index=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )


class JobRecord(Base):
    """Background video inference job tracking progress and summary (`jobs` table)."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), default="video", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
