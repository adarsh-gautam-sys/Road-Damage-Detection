# REST API Reference — Road Damage AI (`/api/v1`)

The `rdai` backend is powered by **FastAPI**. By default, it runs on `http://localhost:8000` with the base route prefix `/api/v1`. Interactive OpenAPI documentation (Swagger UI) is available at `http://localhost:8000/docs`.

---

## 1. System & Health Endpoints

### 1.1. Health Check
Checks model readiness, loaded weights, hardware accelerator device, and active confidence gate.

* **Method:** `GET`
* **Path:** `/api/v1/health` (also aliased at `/health`)
* **Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "yolo11s",
  "run_name": "e1_yolo11s_640",
  "device": "cuda:0",
  "optimal_conf": 0.193,
  "db_connected": true,
  "timestamp": "2026-10-30T00:58:30Z"
}
```

### 1.2. Runtime Configuration
Exposes active thresholds, clustering parameters, and prioritization weights from `configs/*.yaml`.

* **Method:** `GET`
* **Path:** `/api/v1/config`
* **Response (200 OK):**
```json
{
  "api_prefix": "/api/v1",
  "optimal_conf": 0.193,
  "clustering": {
    "eps_meters": 25.0,
    "min_samples": 1
  },
  "priority_weights": {
    "s_max": 0.45,
    "density": 0.25,
    "road_importance": 0.20,
    "confidence": 0.10
  },
  "india_bbox": {
    "lat_min": 6.0,
    "lat_max": 38.0,
    "lng_min": 68.0,
    "lng_max": 98.0
  }
}
```

### 1.3. Evaluation Metrics
Returns verified quantitative benchmark results directly from `reports/metrics.json`.

* **Method:** `GET`
* **Path:** `/api/v1/metrics`
* **Response (200 OK):**
```json
{
  "status": "populated",
  "optimal_conf": 0.193,
  "experiments": {
    "E1": {
      "overall": {
        "precision": 0.4487,
        "recall": 0.3467,
        "map50": 0.3655,
        "map50_95": 0.1728
      },
      "per_class": {
        "D00": { "precision": 0.3223, "recall": 0.2295, "ap50": 0.2409 },
        "D10": { "precision": 0.4316, "recall": 0.1538, "ap50": 0.2221 },
        "D20": { "precision": 0.5443, "recall": 0.6073, "ap50": 0.5885 },
        "D40": { "precision": 0.4964, "recall": 0.3963, "ap50": 0.4105 }
      }
    }
  }
}
```

---

## 2. Ingestion & Detection Endpoints

### 2.1. Photo Detection (`/api/v1/detect/image`)
Submits a single road photograph for full YOLO11s detection, geometric severity grading, and 25m segment clustering.

* **Method:** `POST`
* **Path:** `/api/v1/detect/image`
* **Content-Type:** `multipart/form-data`
* **Form Fields:**
  - `file` *(binary, required)*: Image file (JPEG, PNG, WebP; max 15MB, max 40MP).
  - `lat` *(float, optional)*: Explicit latitude (overrides EXIF if provided).
  - `lng` *(float, optional)*: Explicit longitude (overrides EXIF if provided).
  - `conf` *(float, optional)*: Custom confidence threshold (defaults to `0.193`).

* **Example cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@road_sample.jpg" \
  -F "lat=28.4595" \
  -F "lng=77.0266" \
  -F "conf=0.193"
```

* **Response (200 OK):**
```json
{
  "image_id": 108,
  "sha256": "4a7b92...",
  "width": 1280,
  "height": 720,
  "lat": 28.4595,
  "lng": 77.0266,
  "geo_source": "explicit",
  "annotated_url": "/media/annotated/4a7b92.jpg",
  "thumb_url": "/media/thumbs/4a7b92.jpg",
  "detections_count": 2,
  "detections": [
    {
      "id": 341,
      "class_code": "D40",
      "class_name": "pothole",
      "confidence": 0.842,
      "box": { "x1": 420.0, "y1": 310.5, "x2": 580.0, "y2": 450.0 },
      "extent": 0.0242,
      "measure": "area_ratio",
      "severity_score": 42.1,
      "severity_label": "Medium",
      "segment_id": 12
    },
    {
      "id": 342,
      "class_code": "D20",
      "class_name": "alligator_crack",
      "confidence": 0.612,
      "box": { "x1": 150.0, "y1": 500.0, "x2": 390.0, "y2": 680.0 },
      "extent": 0.0468,
      "measure": "area_ratio",
      "severity_score": 41.3,
      "severity_label": "Medium",
      "segment_id": 12
    }
  ],
  "segment": {
    "id": 12,
    "priority_score": 68.4,
    "priority_label": "Medium",
    "centroid_lat": 28.45948,
    "centroid_lng": 77.02661,
    "n_detections": 5
  }
}
```

---

### 2.2. Video Detection Job (`/api/v1/detect/video`)
Uploads vehicle dashcam video for background ByteTrack deduplicated processing.

* **Method:** `POST`
* **Path:** `/api/v1/detect/video`
* **Content-Type:** `multipart/form-data`
* **Form Fields:**
  - `video` *(binary, required)*: Video file (`.mp4`, `.mov`, `.avi`; max 200MB).
  - `track` *(binary, optional)*: Synchronized GPX or CSV file with coordinates and timestamps.
  - `conf` *(float, optional)*: Confidence gate (defaults to `0.193`).

* **Response (202 Accepted):**
```json
{
  "job_id": 17,
  "kind": "video",
  "status": "queued",
  "progress": 0.0,
  "message": "Video accepted for background processing"
}
```

---

### 2.3. Video Job Status Polling (`/api/v1/jobs/{job_id}`)
Monitors background processing progress and retrieves final deduplicated detection count.

* **Method:** `GET`
* **Path:** `/api/v1/jobs/{job_id}`
* **Response (200 OK):**
```json
{
  "id": 17,
  "status": "finished",
  "progress": 1.0,
  "error": null,
  "summary": {
    "total_frames_processed": 450,
    "unique_defect_tracks": 14,
    "segments_created": 3
  },
  "created_at": "2026-10-30T00:50:00Z",
  "finished_at": "2026-10-30T00:50:42Z"
}
```

---

## 3. Maintenance Segments & Priority Queue

### 3.1. List Prioritized Road Segments (`/api/v1/segments`)
Returns 25-meter road segments ranked in descending order of Maintenance Priority Score ($P$).

* **Method:** `GET`
* **Path:** `/api/v1/segments`
* **Query Parameters:**
  - `limit` *(int, default: 20, max: 100)*: Items per page.
  - `offset` *(int, default: 0)*: Pagination offset.
  - `status` *(string, optional)*: Filter by `new`, `verified`, `scheduled`, `fixed`, `rejected`.
  - `priority_label` *(string, optional)*: Filter by `High`, `Medium`, `Low`.
  - `min_priority` *(float, optional)*: Minimum priority score threshold.

* **Response (200 OK):**
```json
{
  "total": 48,
  "limit": 20,
  "offset": 0,
  "items": [
    {
      "id": 12,
      "centroid_lat": 28.45948,
      "centroid_lng": 77.02661,
      "n_detections": 5,
      "max_severity": 84.5,
      "mean_confidence": 0.74,
      "road_class": "primary",
      "priority_score": 75.8,
      "priority_label": "High",
      "status": "new",
      "terms": {
        "s_max": 84.5,
        "density": 50.0,
        "road_importance": 85.0,
        "confidence": 74.0
      },
      "weighted_terms": {
        "s_max_contrib": 38.02,
        "density_contrib": 12.50,
        "road_contrib": 17.00,
        "conf_contrib": 7.40
      }
    }
  ]
}
```

---

### 3.2. Get Segment Detail (`/api/v1/segments/{segment_id}`)
Returns comprehensive details for a single 25-meter road segment, including all associated detections and the full priority ledger.

* **Method:** `GET`
* **Path:** `/api/v1/segments/{segment_id}`

---

### 3.3. Update Segment Work-Order Status (`/api/v1/segments/{segment_id}/status`)
Updates the maintenance repair status following the civil engineering state transition lifecycle:
$$\text{new} \rightarrow \{\text{verified}, \text{rejected}\}, \quad \text{verified} \rightarrow \{\text{scheduled}, \text{rejected}\}, \quad \text{scheduled} \rightarrow \{\text{fixed}\}$$

* **Method:** `PATCH`
* **Path:** `/api/v1/segments/{segment_id}/status`
* **Request Body:**
```json
{
  "status": "scheduled"
}
```
* **Response (200 OK):**
```json
{
  "id": 12,
  "status": "scheduled",
  "updated_at": "2026-10-30T01:00:00Z"
}
```

---

## 4. Geospatial & Export Endpoints

### 4.1. RFC 7946 GeoJSON Feed (`/api/v1/map/geojson`)
Returns an RFC 7946 compliant GeoJSON `FeatureCollection` for interactive map layers. Coordinates are strictly in `[longitude, latitude]` order per RFC 7946.

* **Method:** `GET`
* **Path:** `/api/v1/map/geojson`
* **Query Parameters:**
  - `status` *(string, optional)*: Filter by repair status.
  - `priority_label` *(string, optional)*: Filter by priority tier.

* **Response (200 OK):**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [77.02661, 28.45948]
      },
      "properties": {
        "segment_id": 12,
        "priority_score": 75.8,
        "priority_label": "High",
        "max_severity": 84.5,
        "n_detections": 5,
        "road_class": "primary",
        "status": "new"
      }
    }
  ]
}
```

---

### 4.2. CSV Maintenance Schedule Export (`/api/v1/export/csv`)
Exports the prioritized road repair ledger as a downloadable CSV table ready for import into municipal GIS software or spreadsheet workflows.

* **Method:** `GET`
* **Path:** `/api/v1/export/csv`
* **Response:** Direct `text/csv` attachment download (`Content-Disposition: attachment; filename=rdai_maintenance_schedule.csv`).

---

### 4.3. Aggregated Municipal Stats (`/api/v1/stats`)
Returns high-level statistics across all inspections in the system.

* **Method:** `GET`
* **Path:** `/api/v1/stats`
* **Response (200 OK):**
```json
{
  "total_images": 215,
  "total_detections": 482,
  "total_segments": 64,
  "by_class": {
    "D00": 74,
    "D10": 12,
    "D20": 182,
    "D40": 214
  },
  "by_severity": {
    "High": 142,
    "Medium": 268,
    "Low": 72
  },
  "by_priority": {
    "High": 18,
    "Medium": 34,
    "Low": 12
  },
  "by_status": {
    "new": 42,
    "verified": 10,
    "scheduled": 8,
    "fixed": 4,
    "rejected": 0
  }
}
```
