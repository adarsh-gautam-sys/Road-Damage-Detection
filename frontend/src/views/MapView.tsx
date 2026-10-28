import L from "leaflet";
import { Crosshair, MapPinOff } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { MapContainer, Marker, TileLayer, useMap } from "react-leaflet";
import { api } from "../api/client";
import type { ClassCode, SeverityLabel } from "../api/types";
import { EmptyState, ErrorNote, PlainButton } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import type { GeoFix } from "../hooks/useGeolocation";
import { pinIcon, severityIcon } from "../lib/leafletIcons";
import { CLASS_NAMES } from "../lib/severity";

type Layer = "segments" | "detections";
const SEVERITIES: SeverityLabel[] = ["High", "Medium", "Low"];
const CLASSES: ClassCode[] = ["D40", "D20", "D00", "D10"];
const INDIA: [number, number] = [22.5, 79];

function FitOnce({ points, signal }: { points: Array<[number, number]>; signal: string }) {
  const map = useMap();
  useEffect(() => {
    if (points.length === 0) return;
    if (points.length === 1) map.setView(points[0]!, 17);
    else map.fitBounds(L.latLngBounds(points), { padding: [48, 48], maxZoom: 18 });
    // refit only when the layer/filters change, never on a background refresh
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signal]);
  return null;
}

function FlyTo({ target }: { target: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (target) map.flyTo(target, Math.max(map.getZoom(), 17), { duration: 0.8 });
  }, [target, map]);
  return null;
}

interface Props {
  refreshKey: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
  onGoCapture: () => void;
  fix: GeoFix | null;
}

export function MapView({ refreshKey, selectedId, onSelect, onGoCapture, fix }: Props) {
  const [layer, setLayer] = useState<Layer>("segments");
  const [sev, setSev] = useState<Set<SeverityLabel>>(new Set(SEVERITIES));
  const [cls, setCls] = useState<Set<ClassCode>>(new Set(CLASSES));
  const [flyTarget, setFlyTarget] = useState<[number, number] | null>(null);

  const segs = useAsync(() => api.segments(), [refreshKey]);
  const dets = useAsync(() => api.geojson("detections"), [refreshKey]);

  const toggle = <T,>(set: Set<T>, v: T, setter: (s: Set<T>) => void) => {
    const n = new Set(set);
    if (n.has(v)) n.delete(v);
    else n.add(v);
    setter(n);
  };

  const segmentMarkers = useMemo(
    () =>
      (segs.data?.items ?? [])
        .filter((s) => sev.has(s.priority_label))
        .map((s) => ({
          id: s.id,
          pos: [s.centroid_lat, s.centroid_lng] as [number, number],
          icon: severityIcon(s.priority_label, String(Math.round(s.priority_score)), 34 + Math.round(s.priority_score / 8), s.id === selectedId),
          title: `Segment ${s.id}, ${s.priority_label} priority ${s.priority_score.toFixed(1)}`,
        })),
    [segs.data, sev, selectedId],
  );

  const detectionMarkers = useMemo(
    () =>
      (dets.data?.features ?? [])
        .filter((f) => sev.has(f.properties.severity_label as SeverityLabel) && cls.has(f.properties.class_code as ClassCode))
        .map((f) => {
          const p = f.properties as { id: number; class_code: ClassCode; severity_label: SeverityLabel; severity_score: number; segment_id: number | null };
          return {
            id: p.id,
            segmentId: p.segment_id,
            pos: [f.geometry.coordinates[1], f.geometry.coordinates[0]] as [number, number],
            icon: severityIcon(p.severity_label, String(Math.round(p.severity_score)), 30, p.segment_id === selectedId && selectedId !== null),
            title: `${p.class_code} ${CLASS_NAMES[p.class_code]}, ${p.severity_label} ${p.severity_score.toFixed(1)}`,
          };
        }),
    [dets.data, sev, cls, selectedId],
  );

  const points = (layer === "segments" ? segmentMarkers : detectionMarkers).map((m) => m.pos);
  const loading = layer === "segments" ? segs.loading && !segs.data : dets.loading && !dets.data;
  const error = layer === "segments" ? segs.error : dets.error;
  const empty = !loading && !error && (layer === "segments" ? (segs.data?.items.length ?? 0) === 0 : (dets.data?.features.length ?? 0) === 0);
  const youIcon = useMemo(() => pinIcon(), []);

  return (
    <div className="relative flex h-full min-h-0 flex-col">
      <div className="z-[500] flex shrink-0 items-center gap-2 overflow-x-auto border-b border-etch bg-chassis px-3 py-2">
        <div role="group" aria-label="Map layer" className="flex shrink-0 gap-1">
          {(["segments", "detections"] as const).map((l) => (
            <button
              key={l}
              type="button"
              aria-pressed={layer === l}
              onClick={() => setLayer(l)}
              className={`min-h-11 rounded-md border px-3 font-label text-sm font-bold uppercase tracking-wider ${
                layer === l ? "border-gold bg-raised text-gold" : "border-etch text-ink-2 hover:text-ink"
              }`}
            >
              {l}
            </button>
          ))}
        </div>
        <span className="mx-1 h-6 w-px shrink-0 bg-etch" aria-hidden="true" />
        <div role="group" aria-label="Filter by severity" className="flex shrink-0 gap-1">
          {SEVERITIES.map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={sev.has(s)}
              onClick={() => toggle(sev, s, setSev)}
              className={`min-h-11 rounded-md border px-3 font-label text-sm font-bold uppercase tracking-wider ${
                sev.has(s) ? "border-etch-strong bg-raised text-ink" : "border-etch text-ink-3 line-through"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        {layer === "detections" && (
          <>
            <span className="mx-1 h-6 w-px shrink-0 bg-etch" aria-hidden="true" />
            <div role="group" aria-label="Filter by class" className="flex shrink-0 gap-1">
              {CLASSES.map((c) => (
                <button
                  key={c}
                  type="button"
                  aria-pressed={cls.has(c)}
                  title={CLASS_NAMES[c]}
                  onClick={() => toggle(cls, c, setCls)}
                  className={`min-h-11 rounded-md border px-3 font-mono text-sm font-bold ${
                    cls.has(c) ? "border-etch-strong bg-raised text-ink" : "border-etch text-ink-3 line-through"
                  }`}
                >
                  {c}
                </button>
              ))}
            </div>
          </>
        )}
        <PlainButton
          className="ml-auto shrink-0"
          disabled={!fix}
          onClick={() => fix && setFlyTarget([fix.lat, fix.lng])}
          aria-label="Centre map on my location"
        >
          <Crosshair size={16} aria-hidden="true" /> Me
        </PlainButton>
      </div>

      <div className="relative min-h-0 flex-1">
        <MapContainer center={INDIA} zoom={5} className="h-full w-full" zoomControl>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />
          <FitOnce points={points} signal={`${layer}|${[...sev].join()}|${[...cls].join()}|${points.length > 0}`} />
          <FlyTo target={flyTarget} />
          {layer === "segments" &&
            segmentMarkers.map((m) => (
              <Marker key={m.id} position={m.pos} icon={m.icon} title={m.title} eventHandlers={{ click: () => onSelect(m.id) }} />
            ))}
          {layer === "detections" &&
            detectionMarkers.map((m) => (
              <Marker
                key={m.id}
                position={m.pos}
                icon={m.icon}
                title={m.title}
                eventHandlers={{ click: () => m.segmentId !== null && onSelect(m.segmentId) }}
              />
            ))}
          {fix && <Marker position={[fix.lat, fix.lng]} icon={youIcon} title="You are here" interactive={false} />}
        </MapContainer>

        {error && (
          <div className="absolute inset-x-3 top-3 z-[600]">
            <ErrorNote message={error.message} onRetry={layer === "segments" ? segs.reload : dets.reload} />
          </div>
        )}
        {empty && (
          <div className="pointer-events-none absolute inset-0 z-[600] grid place-items-center bg-chassis/80">
            <div className="pointer-events-auto">
              <EmptyState
                icon={<MapPinOff size={26} aria-hidden="true" />}
                title={layer === "segments" ? "No segments on the map yet" : "No located detections yet"}
                body="Log a reading with a GPS fix or a placed pin and it appears here, grouped into road segments."
                action={<PlainButton onClick={onGoCapture}>Go to capture</PlainButton>}
              />
            </div>
          </div>
        )}
        {loading && (
          <p className="absolute top-3 left-3 z-[600] rounded bg-chassis/90 px-2 py-1 font-mono text-[12px] text-ink-2" role="status">
            Loading…
          </p>
        )}
      </div>

      <p className="z-[500] shrink-0 border-t border-etch bg-chassis px-3 py-1.5 text-[12px] text-ink-3">
        Marker shape: ▲ high, ◆ medium, ● low. The number is the {layer === "segments" ? "priority score" : "severity score"}.
      </p>
    </div>
  );
}
