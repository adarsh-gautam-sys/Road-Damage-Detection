/** Types mirroring the FastAPI `/api/v1` contract (src/rdai/api). */

export type ClassCode = "D00" | "D10" | "D20" | "D40";
export type SeverityLabel = "Low" | "Medium" | "High";
export type SegmentStatus = "new" | "verified" | "scheduled" | "fixed" | "rejected";
export type SeverityMeasure = "area_ratio" | "diag_ratio";

export interface Severity {
  score: number;
  label: SeverityLabel;
  extent: number;
  measure: SeverityMeasure;
}

/** One detection as returned by `/detect/preview` (not persisted). */
export interface PreviewDetection {
  class_id: number;
  class_code: ClassCode;
  class_name: string;
  confidence: number;
  /** [x1, y1, x2, y2] in source-image pixels. */
  box: [number, number, number, number];
  severity: Severity;
}

export interface PreviewResponse {
  width: number;
  height: number;
  conf_used: number;
  inference_ms: number;
  detections: PreviewDetection[];
}

export interface DetectionItem {
  id: number;
  class_code: ClassCode;
  class_name: string;
  confidence: number;
  box: [number, number, number, number];
  severity: Severity;
  segment_id: number | null;
}

export interface ImageDetectResponse {
  image_id: number;
  width: number;
  height: number;
  geo: { lat: number | null; lng: number | null; source: string; warnings: string[] };
  annotated_url: string;
  thumb_url: string;
  detections: DetectionItem[];
  model: { name: string; run: string; conf: number };
}

export interface Health {
  status: string;
  model_loaded: boolean;
  version: string;
  model_name: string;
  run: string;
}

export interface PriorityTerms {
  s_max: number;
  density: number;
  road_importance: number;
  confidence: number;
}

export interface Segment {
  id: number;
  centroid_lat: number;
  centroid_lng: number;
  n_detections: number;
  max_severity: number;
  mean_confidence: number;
  road_class: string;
  priority_score: number;
  priority_label: SeverityLabel;
  terms: Partial<PriorityTerms>;
  weighted_terms: Partial<PriorityTerms>;
  status: SegmentStatus;
}

export interface SegmentMember {
  id: number;
  image_id: number | null;
  thumb_url: string | null;
  annotated_url: string | null;
  class_code: ClassCode;
  class_name: string;
  confidence: number;
  box: [number, number, number, number];
  extent: number;
  measure: SeverityMeasure;
  severity_score: number;
  severity_label: SeverityLabel;
  lat: number | null;
  lng: number | null;
}

export interface SegmentDetail extends Segment {
  detections: SegmentMember[];
}

export interface Stats {
  total_images: number;
  total_detections: number;
  total_segments: number;
  by_class: Record<ClassCode, number>;
  by_severity: Record<SeverityLabel, number>;
  by_status: Record<SegmentStatus, number>;
}

export interface JobStatus {
  id: number;
  kind: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  error: string | null;
  summary: Record<string, unknown>;
}

export interface ExperimentMetrics {
  experiment_id: string;
  split: string;
  params_m: number;
  gpu_ms: number;
  cpu_ms: number | null;
  overall: { precision: number; recall: number; map50: number; map50_95: number };
  per_class: Record<ClassCode, { precision: number; recall: number; ap50: number; ap50_95: number }>;
  confusion_matrix: number[][];
  optimal_val_conf: number;
  optimal_val_f1: number;
  metric_gate: { passed: boolean; map50: number; min_map50: number; d40_ap50: number; min_pothole_ap50: number };
}

export interface MetricsDoc {
  optimal_conf: number;
  experiments: Record<string, ExperimentMetrics>;
}

export interface GeoJsonFeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: { type: "Point"; coordinates: [number, number] };
    properties: Record<string, unknown>;
  }>;
}

export interface ApiErrorBody {
  error?: { code?: string; message?: string };
  detail?: unknown;
}
