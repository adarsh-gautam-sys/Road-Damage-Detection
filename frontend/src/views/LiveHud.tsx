import { Camera, CameraOff, Crosshair, ImageUp } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { ImageDetectResponse, PreviewDetection } from "../api/types";
import { LcdReading } from "../components/LcdReading";
import { GoldButton, PlainButton } from "../components/States";
import { Switch } from "../components/Switch";
import { hitTest, mapBoxToViewport, resolveSelected, type SelectionAnchor } from "../lib/boxes";
import { shouldAutoLog, type PatrolStamp } from "../lib/geo";
import { toJpegBlob } from "../lib/image";
import { CLASS_COLOR_VAR } from "../lib/severity";
import { useConfig } from "../state/ConfigContext";
import type { GeoFix, GeoStatus } from "../hooks/useGeolocation";

type CamState = "idle" | "requesting" | "live" | "denied" | "unsupported" | "error";

interface Props {
  conf: number;
  onConfChange: (v: number) => void;
  geo: { status: GeoStatus; fix: GeoFix | null };
  onOpenSegment: (id: number) => void;
  onSwitchToUpload: () => void;
  onCommitted: () => void;
}

function cssColor(varName: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || "#fff";
}

export function LiveHud({ conf, onConfChange, geo, onOpenSegment, onSwitchToUpload, onCommitted }: Props) {
  const cfg = useConfig();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [cam, setCam] = useState<CamState>("idle");
  const [dets, setDets] = useState<PreviewDetection[]>([]);
  const [frame, setFrame] = useState<{ w: number; h: number } | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [anchor, setAnchor] = useState<SelectionAnchor | null>(null);
  const [auto, setAuto] = useState(false);
  const [committing, setCommitting] = useState(false);
  const [last, setLast] = useState<ImageDetectResponse | null>(null);
  const [commitError, setCommitError] = useState<string | null>(null);
  const [sessionCount, setSessionCount] = useState(0);
  const [flashKey, setFlashKey] = useState(0);

  // Latest values for the long-lived loop (so changing them never restarts the camera)
  const confRef = useRef(conf);
  const autoRef = useRef(auto);
  const geoRef = useRef(geo.fix);
  const committingRef = useRef(false);
  const lastLogRef = useRef<PatrolStamp | null>(null);
  confRef.current = conf;
  autoRef.current = auto;
  geoRef.current = geo.fix;

  const visible = useMemo(() => dets.filter((d) => d.confidence >= conf), [dets, conf]);
  const selectedIdx = useMemo(
    () => (frame ? resolveSelected(visible, anchor, frame) : null),
    [visible, anchor, frame],
  );
  const focusDet: PreviewDetection | null = useMemo(() => {
    if (selectedIdx !== null) return visible[selectedIdx] ?? null;
    return visible.reduce<PreviewDetection | null>(
      (best, d) => (best === null || d.severity.score > best.severity.score ? d : best),
      null,
    );
  }, [visible, selectedIdx]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCam("unsupported");
      return;
    }
    setCam("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play().catch(() => undefined);
      }
      setCam("live");
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      setCam(name === "NotAllowedError" || name === "SecurityError" ? "denied" : "error");
    }
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const commit = useCallback(async () => {
    const v = videoRef.current;
    if (!v || committingRef.current) return;
    committingRef.current = true;
    setCommitting(true);
    setCommitError(null);
    try {
      const { blob } = await toJpegBlob(v, cfg.client.upload_max_edge_px, cfg.client.upload_jpeg_quality);
      const fix = geoRef.current;
      const res = await api.detectImage(blob, {
        lat: fix?.lat ?? null,
        lng: fix?.lng ?? null,
        conf: confRef.current,
        filename: `patrol_${Date.now()}.jpg`,
      });
      lastLogRef.current = { lat: fix?.lat ?? null, lng: fix?.lng ?? null, t: Date.now() };
      setLast(res);
      setSessionCount((n) => n + 1);
      setFlashKey((k) => k + 1);
      onCommitted();
    } catch (err) {
      setCommitError(err instanceof Error ? err.message : "Could not log this frame.");
    } finally {
      committingRef.current = false;
      setCommitting(false);
    }
  }, [cfg.client.upload_jpeg_quality, cfg.client.upload_max_edge_px, onCommitted]);

  // Preview loop: stateless frames at ~preview_fps, never persisted.
  useEffect(() => {
    if (cam !== "live") return;
    let stopped = false;
    let timer: number | undefined;
    const ac = new AbortController();
    const period = 1000 / cfg.client.preview_fps;
    let prevTick = performance.now();

    const tick = async () => {
      if (stopped) return;
      const t0 = performance.now();
      const v = videoRef.current;
      if (v && v.readyState >= 2 && !document.hidden) {
        try {
          const { blob } = await toJpegBlob(v, cfg.client.preview_max_edge_px, cfg.client.preview_jpeg_quality);
          const res = await api.preview(blob, confRef.current, ac.signal);
          if (stopped) return;
          setPreviewError(null);
          setFrame({ w: res.width, h: res.height });
          setDets(res.detections);
          setLatency(res.inference_ms);
          const now = performance.now();
          setFps(1000 / Math.max(1, now - prevTick));
          prevTick = now;

          if (autoRef.current && !committingRef.current) {
            const fix = geoRef.current;
            const go = shouldAutoLog({
              hasDetection: res.detections.some((d) => d.confidence >= confRef.current),
              now: { lat: fix?.lat ?? null, lng: fix?.lng ?? null, t: Date.now() },
              last: lastLogRef.current,
              cfg: cfg.client,
            });
            if (go) void commit();
          }
        } catch (err) {
          if (stopped || (err instanceof DOMException && err.name === "AbortError")) return;
          setPreviewError(err instanceof Error ? err.message : "Preview failed.");
        }
      }
      timer = window.setTimeout(() => void tick(), Math.max(0, period - (performance.now() - t0)));
    };
    void tick();
    return () => {
      stopped = true;
      ac.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [cam, cfg.client, commit]);

  // Overlay: boxes, fiducials, dimming of non-selected boxes.
  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const { clientWidth: w, clientHeight: h } = wrap;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!frame) return;
      visible.forEach((d, i) => {
        const r = mapBoxToViewport(d.box, frame, { w, h });
        const color = cssColor(CLASS_COLOR_VAR[d.class_code]);
        const dim = selectedIdx !== null && selectedIdx !== i;
        ctx.globalAlpha = dim ? 0.3 : 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = selectedIdx === i ? 3 : 2;
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        // corner fiducials
        ctx.lineWidth = 1.5;
        const f = 5;
        for (const [cx, cy] of [
          [r.x, r.y],
          [r.x + r.w, r.y],
          [r.x, r.y + r.h],
          [r.x + r.w, r.y + r.h],
        ] as const) {
          ctx.beginPath();
          ctx.moveTo(cx - f, cy);
          ctx.lineTo(cx + f, cy);
          ctx.moveTo(cx, cy - f);
          ctx.lineTo(cx, cy + f);
          ctx.stroke();
        }
        // tag
        const tag = `${d.class_code} ${d.severity.score.toFixed(0)}`;
        ctx.font = '600 12px "JetBrains Mono", monospace';
        const tw = ctx.measureText(tag).width + 10;
        const ty = r.y - 20 < 0 ? r.y + 2 : r.y - 20;
        ctx.fillStyle = color;
        ctx.fillRect(r.x, ty, tw, 18);
        ctx.fillStyle = d.class_code === "D20" ? "#141412" : "#ffffff";
        ctx.fillText(tag, r.x + 5, ty + 13);
        ctx.globalAlpha = 1;
      });
    };
    draw();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(draw);
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [visible, frame, selectedIdx]);

  const onPointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const wrap = wrapRef.current;
    if (!wrap || !frame) return;
    const bounds = wrap.getBoundingClientRect();
    const px = e.clientX - bounds.left;
    const py = e.clientY - bounds.top;
    const rects = visible.map((d) => mapBoxToViewport(d.box, frame, { w: bounds.width, h: bounds.height }));
    const hit = hitTest(rects, px, py);
    if (hit === null || hit === selectedIdx) {
      setAnchor(null);
      return;
    }
    const d = visible[hit]!;
    setAnchor({ class_code: d.class_code, cx: (d.box[0] + d.box[2]) / 2, cy: (d.box[1] + d.box[3]) / 2 });
  };

  const live = cam === "live";
  const gpsLine =
    geo.status === "live" && geo.fix
      ? `GPS ±${geo.fix.accuracy.toFixed(0)} m`
      : geo.status === "acquiring"
        ? "GPS acquiring…"
        : "No GPS fix";
  const noFix = live && !geo.fix;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div
        ref={wrapRef}
        className="relative min-h-[260px] flex-1 overflow-hidden rounded-lg border border-etch-strong bg-black"
      >
        <video ref={videoRef} playsInline muted autoPlay className="absolute inset-0 h-full w-full object-cover" />
        <canvas
          ref={canvasRef}
          onPointerDown={onPointer}
          className="absolute inset-0 h-full w-full touch-manipulation"
          aria-label="Live detections. Tap a box to focus it."
        />

        {/* Rangefinder brackets + centre ticks */}
        <div key={flashKey} className={`bracket-frame pointer-events-none absolute inset-0 ${flashKey ? "snap" : ""}`} aria-hidden="true">
          {(["top-2 left-2 border-t-2 border-l-2", "top-2 right-2 border-t-2 border-r-2", "bottom-2 left-2 border-b-2 border-l-2", "bottom-2 right-2 border-b-2 border-r-2"] as const).map((c) => (
            <span key={c} className={`absolute h-6 w-6 border-ink/85 ${c}`} />
          ))}
          <span className="absolute top-1/2 left-1/2 h-px w-8 -translate-x-1/2 bg-ink/45" />
          <span className="absolute top-1/2 left-1/2 h-8 w-px -translate-y-1/2 bg-ink/45" />
        </div>
        {flashKey > 0 && <div key={`f${flashKey}`} className="shutter-flash absolute inset-0" aria-hidden="true" />}

        {live && (
          <div className="pointer-events-none absolute bottom-2 left-3 right-3 flex items-end justify-between font-mono text-[11px] text-ink">
            <span className="rounded bg-black/65 px-1.5 py-0.5">
              {fps ? `${fps.toFixed(1)} fps` : "…"} · {latency !== null ? `${latency.toFixed(0)} ms` : "…"}
            </span>
            <span className="rounded bg-black/65 px-1.5 py-0.5">{gpsLine}</span>
          </div>
        )}

        {cam !== "live" && (
          <div className="absolute inset-0 grid place-items-center bg-chassis/92 p-6 text-center">
            <div className="flex max-w-xs flex-col items-center gap-3">
              {cam === "denied" || cam === "unsupported" || cam === "error" ? (
                <>
                  <CameraOff size={34} className="text-ink-2" aria-hidden="true" />
                  <h2 className="font-label text-lg font-semibold">
                    {cam === "denied" ? "Camera permission denied" : cam === "unsupported" ? "Live camera needs HTTPS" : "Camera unavailable"}
                  </h2>
                  <p className="text-ink-2">
                    {cam === "denied"
                      ? "Allow camera access in the browser's site settings, then try again."
                      : cam === "unsupported"
                        ? "Browsers only open the camera on https:// or localhost. Upload a photo or video instead."
                        : "Another app may be using the camera. Close it, or upload a photo instead."}
                  </p>
                  <div className="flex flex-wrap justify-center gap-2">
                    {cam !== "unsupported" && <PlainButton onClick={() => void startCamera()}>Try again</PlainButton>}
                    <PlainButton onClick={onSwitchToUpload}>
                      <ImageUp size={16} aria-hidden="true" /> Upload instead
                    </PlainButton>
                  </div>
                </>
              ) : (
                <>
                  <Crosshair size={34} className="text-gold" aria-hidden="true" />
                  <h2 className="font-label text-lg font-semibold">Point at the road surface</h2>
                  <p className="text-ink-2">Live boxes are previews only. Nothing is saved until you log a reading.</p>
                  <GoldButton onClick={() => void startCamera()} disabled={cam === "requesting"}>
                    <span className="inline-flex items-center gap-2">
                      <Camera size={18} aria-hidden="true" /> {cam === "requesting" ? "Opening…" : "Start camera"}
                    </span>
                  </GoldButton>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {previewError && live && (
        <p role="status" className="rounded-md border border-sev-med/50 px-3 py-2 text-sm text-sev-med-text">
          {previewError} Retrying…
        </p>
      )}

      {/* Transflective LCD register */}
      <div className="lcd px-3.5 pt-2.5 pb-1">
        <LcdReading
          det={live ? focusDet : null}
          empty={live ? `NO DEFECT ABOVE CONF ${conf.toFixed(3)}` : "CAMERA OFF · READING UNAVAILABLE"}
        />
        <div className="flex items-center gap-3">
          <label htmlFor="conf-live" className="font-label text-[12px] font-bold uppercase tracking-wider">
            Conf
          </label>
          <input
            id="conf-live"
            type="range"
            min={0.05}
            max={0.9}
            step={0.001}
            value={conf}
            onChange={(e) => onConfChange(Number(e.target.value))}
            aria-valuetext={`confidence threshold ${conf.toFixed(3)}`}
          />
          <output htmlFor="conf-live" className="tnum w-12 text-right text-[13px] font-bold">
            {conf.toFixed(3)}
          </output>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <GoldButton className="flex-1" onClick={() => void commit()} disabled={!live || committing}>
          {committing ? "Logging…" : "Log reading"}
        </GoldButton>
        <Switch checked={auto} onChange={setAuto} label="Auto-patrol" disabled={!live} />
      </div>
      {live && auto && (
        <p className="-mt-1 text-[13px] text-ink-2">
          Logs a defect after {cfg.client.auto_patrol_min_distance_m} m of movement or {cfg.client.auto_patrol_min_interval_s} s.
          {noFix && " No GPS yet: only the timer applies."}
        </p>
      )}

      {commitError && (
        <p role="alert" className="rounded-md border border-sev-high/60 px-3 py-2 text-sm text-sev-high-text">
          {commitError}
        </p>
      )}
      {last && !commitError && (
        <div className="plate flex items-center justify-between gap-3 px-3 py-2 text-sm" role="status">
          <p className="min-w-0 text-ink-2">
            <span className="tnum text-ink">#{sessionCount}</span> logged ·{" "}
            {last.detections.length === 0
              ? "no defects at this threshold"
              : `${last.detections.length} defect${last.detections.length > 1 ? "s" : ""}`}
            {last.geo.lat === null && " · no location, so not on the map"}
          </p>
          {last.detections[0]?.segment_id != null && (
            <PlainButton onClick={() => onOpenSegment(last.detections[0]!.segment_id!)} className="shrink-0">
              Segment {last.detections[0]!.segment_id}
            </PlainButton>
          )}
        </div>
      )}
    </div>
  );
}
