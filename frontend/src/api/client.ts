import type {
  ExperimentMetrics,
  GeoJsonFeatureCollection,
  Health,
  ImageDetectResponse,
  JobStatus,
  MetricsDoc,
  PreviewResponse,
  Segment,
  SegmentDetail,
  SegmentStatus,
  Stats,
} from "./types";

const BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
const API = `${BASE}/api/v1`;

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** Absolute URL for a backend-served asset path such as `/media/thumbs/1.jpg`. */
export function assetUrl(path: string | null | undefined): string {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  return `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "NETWORK", "Can't reach the Road Damage AI server.");
  }
  if (!res.ok) {
    let code = `HTTP_${res.status}`;
    let message = `Request failed (${res.status}).`;
    try {
      const body = (await res.json()) as { error?: { code?: string; message?: string } };
      if (body.error?.code) code = body.error.code;
      if (body.error?.message) message = body.error.message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message);
  }
  return (await res.json()) as T;
}

export interface ClientConfig {
  default_conf: number;
  model: { name: string; run: string };
  limits: { max_photo_mb: number; max_video_mb: number };
  client: {
    preview_fps: number;
    preview_max_edge_px: number;
    preview_jpeg_quality: number;
    upload_max_edge_px: number;
    upload_jpeg_quality: number;
    auto_patrol_min_distance_m: number;
    auto_patrol_min_interval_s: number;
    gps_good_accuracy_m: number;
    job_poll_interval_ms: number;
  };
  priority: {
    weights: { s_max: number; density: number; road_importance: number; confidence: number };
    bands: { low_max: number; medium_max: number };
    eps_meters: number;
    density_saturation_count: number;
  };
  severity: {
    tier_bounds: { low_upper: number; medium_upper: number; max_score: number };
    classes: Record<string, { name: string; measure: string; e_low: number; e_high: number; min_label: string }>;
  };
}

export interface MetricsPayload extends MetricsDoc {
  weight_sensitivity?: unknown;
}

export type { ExperimentMetrics };

function form(fields: Record<string, string | Blob | undefined | null>, filenames: Record<string, string> = {}): FormData {
  const fd = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null) continue;
    if (typeof v === "string") fd.append(k, v);
    else fd.append(k, v, filenames[k] ?? "upload.bin");
  }
  return fd;
}

export const api = {
  health: () => request<Health>("/health"),
  config: () => request<ClientConfig>("/config"),
  stats: () => request<Stats>("/stats"),
  metrics: () => request<MetricsPayload>("/metrics"),

  segments: (params: { label?: string; status?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.label) q.set("label", params.label);
    if (params.status) q.set("status", params.status);
    const qs = q.toString();
    return request<{ total: number; items: Segment[] }>(`/segments${qs ? `?${qs}` : ""}`);
  },
  segment: (id: number) => request<SegmentDetail>(`/segments/${id}`),
  setSegmentStatus: (id: number, status: SegmentStatus) =>
    request<{ id: number; status: SegmentStatus }>(`/segments/${id}/status`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    }),
  geojson: (layer: "detections" | "segments") =>
    request<GeoJsonFeatureCollection>(`/map/geojson?layer=${layer}`),

  /** Stateless live-camera frame. Never writes to the database. */
  preview: (frame: Blob, conf: number, signal?: AbortSignal) =>
    request<PreviewResponse>("/detect/preview", {
      method: "POST",
      body: form({ file: frame, conf: String(conf) }, { file: "frame.jpg" }),
      signal,
    }),

  /** Persisting photo detection. Coordinates are sent explicitly because resizing strips EXIF. */
  detectImage: (
    image: Blob,
    opts: { lat: number | null; lng: number | null; conf: number; filename?: string },
  ) =>
    request<ImageDetectResponse>("/detect/image", {
      method: "POST",
      body: form(
        {
          file: image,
          lat: opts.lat === null ? undefined : String(opts.lat),
          lng: opts.lng === null ? undefined : String(opts.lng),
          conf: String(opts.conf),
        },
        { file: opts.filename ?? "capture.jpg" },
      ),
    }),

  detectVideo: (video: File, track: File | null, conf: number) =>
    request<{ job_id: number; status: string }>("/detect/video", {
      method: "POST",
      body: form(
        { file: video, track: track ?? undefined, conf: String(conf) },
        { file: video.name, track: track?.name ?? "track.gpx" },
      ),
    }),
  job: (id: number) => request<JobStatus>(`/jobs/${id}`),

  exportDetectionsCsvUrl: () => `${API}/export/csv`,
  exportSegmentsGeojsonUrl: () => `${API}/map/geojson?layer=segments`,
  figureUrl: (name: string) => `${BASE}/reports/figures/${name}`,
};
