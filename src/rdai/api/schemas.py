"""Pydantic v2 request and response schemas for `/api/v1` endpoints (`PRD Section 11`, `A02`, `A10`)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str
    model_name: str
    run: str


class GeoInfoSchema(BaseModel):
    lat: float | None = None
    lng: float | None = None
    source: str
    warnings: list[str] = Field(default_factory=list)


class SeverityInfoSchema(BaseModel):
    score: float
    label: str
    extent: float
    measure: str


class DetectionItemSchema(BaseModel):
    id: int
    class_code: str
    class_name: str
    confidence: float
    box: list[float]
    severity: SeverityInfoSchema
    segment_id: int | None = None


class ModelInfoSchema(BaseModel):
    name: str
    run: str
    conf: float


class ImageDetectResponse(BaseModel):
    image_id: int
    width: int
    height: int
    geo: GeoInfoSchema
    annotated_url: str
    thumb_url: str
    detections: list[DetectionItemSchema]
    model: ModelInfoSchema


class VideoJobAcceptedResponse(BaseModel):
    job_id: int
    status: str


class JobStatusResponse(BaseModel):
    id: int
    kind: str
    status: str
    progress: float
    error: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class SegmentStatusPatchRequest(BaseModel):
    status: str
