import { Camera, Images, MapPin } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, assetUrl } from "../api/client";
import type { ImageDetectResponse } from "../api/types";
import { LocationPicker, type LatLng } from "../components/LocationPicker";
import { SeverityChip } from "../components/Severity";
import { ErrorNote, GoldButton, PlainButton } from "../components/States";
import type { GeoFix, GeoStatus } from "../hooks/useGeolocation";
import { fileToBitmap, toJpegBlob } from "../lib/image";
import { CLASS_NAMES, formatCoord } from "../lib/severity";
import { useConfig } from "../state/ConfigContext";

interface Props {
  conf: number;
  geo: { status: GeoStatus; fix: GeoFix | null };
  onOpenSegment: (id: number) => void;
  onCommitted: () => void;
}

interface Prepared {
  blob: Blob;
  url: string;
  width: number;
  height: number;
  name: string;
}

export function PhotoFlow({ conf, geo, onOpenSegment, onCommitted }: Props) {
  const cfg = useConfig();
  const camInput = useRef<HTMLInputElement>(null);
  const galleryInput = useRef<HTMLInputElement>(null);
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [pin, setPin] = useState<LatLng | null>(null);
  const [adjusting, setAdjusting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImageDetectResponse | null>(null);

  useEffect(() => () => {
    if (prepared) URL.revokeObjectURL(prepared.url);
  }, [prepared]);

  const effective: LatLng | null = pin ?? (geo.fix ? { lat: geo.fix.lat, lng: geo.fix.lng } : null);
  const source = pin ? "Pinned" : geo.fix ? `GPS ±${geo.fix.accuracy.toFixed(0)} m` : "No fix";

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(null);
    try {
      // Resize before upload to save mobile data. This drops EXIF, so coordinates are sent separately.
      const bmp = await fileToBitmap(file);
      const out = await toJpegBlob(bmp, cfg.client.upload_max_edge_px, cfg.client.upload_jpeg_quality);
      bmp.close();
      setPrepared({ blob: out.blob, url: URL.createObjectURL(out.blob), width: out.width, height: out.height, name: file.name });
    } catch {
      setError("That file couldn't be read as an image. Try a JPEG, PNG or WebP photo.");
    }
  };

  const submit = async () => {
    if (!prepared) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.detectImage(prepared.blob, {
        lat: effective?.lat ?? null,
        lng: effective?.lng ?? null,
        conf,
        filename: prepared.name.replace(/\.[^.]+$/, "") + ".jpg",
      });
      setResult(res);
      onCommitted();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed.");
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setPrepared(null);
    setResult(null);
    setError(null);
    setAdjusting(false);
  };

  if (result) {
    return (
      <div className="flex flex-col gap-4">
        <img
          src={assetUrl(result.annotated_url)}
          alt="Photo with detected defects outlined"
          className="max-h-[52dvh] w-full rounded-lg border border-etch-strong object-contain bg-black"
        />
        {result.geo.lat === null && (
          <p className="rounded-md border border-sev-med/50 px-3 py-2 text-sm text-sev-med-text">
            Saved without a location, so it will not appear on the map. Add a pin next time.
          </p>
        )}
        {result.detections.length === 0 ? (
          <p className="plate p-4 text-ink-2">
            No defects found at confidence {result.model.conf.toFixed(3)}. Lower the threshold to see weaker detections.
          </p>
        ) : (
          <ul className="flex flex-col divide-y divide-etch rounded-lg border border-etch">
            {result.detections.map((d) => (
              <li key={d.id} className="flex flex-wrap items-center gap-x-3 gap-y-2 p-3">
                <div className="min-w-0 flex-1">
                  <p className="font-label text-base font-semibold">
                    <span className="font-mono">{d.class_code}</span> {CLASS_NAMES[d.class_code]}
                  </p>
                  <p className="tnum font-mono text-[13px] text-ink-2">
                    {d.severity.measure} {d.severity.extent.toFixed(3)} · conf {d.confidence.toFixed(2)}
                  </p>
                </div>
                <SeverityChip label={d.severity.label} score={d.severity.score} />
                {d.segment_id !== null && (
                  <PlainButton onClick={() => onOpenSegment(d.segment_id!)}>Segment {d.segment_id}</PlainButton>
                )}
              </li>
            ))}
          </ul>
        )}
        <PlainButton onClick={reset}>Analyse another photo</PlainButton>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <input
        ref={camInput}
        type="file"
        accept="image/*"
        capture="environment"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => void pick(e.target.files?.[0])}
      />
      <input
        ref={galleryInput}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => void pick(e.target.files?.[0])}
      />

      {!prepared ? (
        <div className="plate flex flex-col gap-4 p-5">
          <p className="text-ink-2">
            Take a photo of the road surface or pick one you already have. It is resized to {cfg.client.upload_max_edge_px}px before
            upload.
          </p>
          <div className="flex flex-wrap gap-3">
            <GoldButton onClick={() => camInput.current?.click()}>
              <span className="inline-flex items-center gap-2">
                <Camera size={18} aria-hidden="true" /> Take photo
              </span>
            </GoldButton>
            <PlainButton onClick={() => galleryInput.current?.click()} className="min-h-12">
              <Images size={18} aria-hidden="true" /> Choose from gallery
            </PlainButton>
          </div>
        </div>
      ) : (
        <>
          <img src={prepared.url} alt="Photo ready to analyse" className="max-h-[44dvh] w-full rounded-lg border border-etch-strong object-contain bg-black" />

          <div className="plate flex flex-col gap-3 p-3">
            <div className="flex items-center justify-between gap-3">
              <p className="flex min-w-0 items-center gap-2">
                <MapPin size={18} className="shrink-0 text-gold" aria-hidden="true" />
                <span className="tnum truncate font-mono text-[13px]">{formatCoord(effective?.lat, effective?.lng)}</span>
                <span className="label shrink-0">{source}</span>
              </p>
              <PlainButton onClick={() => setAdjusting((a) => !a)} aria-expanded={adjusting}>
                {adjusting ? "Done" : "Adjust pin"}
              </PlainButton>
            </div>
            {adjusting && <LocationPicker value={effective} onChange={setPin} />}
            {!effective && !adjusting && (
              <p className="text-sm text-sev-med-text">No location yet. Without one this photo can't join a road segment.</p>
            )}
          </div>

          <div className="flex gap-3">
            <GoldButton className="flex-1" onClick={() => void submit()} disabled={busy}>
              {busy ? "Analysing…" : "Analyse and log"}
            </GoldButton>
            <PlainButton onClick={reset} disabled={busy}>
              Discard
            </PlainButton>
          </div>
        </>
      )}
      {error && <ErrorNote message={error} onRetry={prepared ? () => void submit() : undefined} />}
    </div>
  );
}
