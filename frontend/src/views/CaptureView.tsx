import { useState } from "react";
import type { GeoFix, GeoStatus } from "../hooks/useGeolocation";
import { LiveHud } from "./LiveHud";
import { PhotoFlow } from "./PhotoFlow";
import { VideoFlow } from "./VideoFlow";

type Mode = "live" | "photo" | "video";

const MODES: Array<{ id: Mode; label: string }> = [
  { id: "live", label: "Live" },
  { id: "photo", label: "Photo" },
  { id: "video", label: "Video" },
];

interface Props {
  conf: number;
  onConfChange: (v: number) => void;
  geo: { status: GeoStatus; fix: GeoFix | null };
  onOpenSegment: (id: number) => void;
  onViewMap: () => void;
  onCommitted: () => void;
}

export function CaptureView({ conf, onConfChange, geo, onOpenSegment, onViewMap, onCommitted }: Props) {
  const [mode, setMode] = useState<Mode>("live");

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-3 p-3 lg:p-6">
      <div role="tablist" aria-label="Capture mode" className="grid shrink-0 grid-cols-3 gap-1 rounded-lg border border-etch bg-panel p-1">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="tab"
            type="button"
            aria-selected={mode === m.id}
            onClick={() => setMode(m.id)}
            className={`min-h-11 rounded-md font-label text-sm font-bold uppercase tracking-wider transition-colors ${
              mode === m.id ? "bg-raised text-gold ring-1 ring-etch-strong" : "text-ink-2 hover:text-ink"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto lg:overflow-visible">
        {mode === "live" && (
          <div className="flex h-full min-h-[560px] flex-col">
            <LiveHud
              conf={conf}
              onConfChange={onConfChange}
              geo={geo}
              onOpenSegment={onOpenSegment}
              onSwitchToUpload={() => setMode("photo")}
              onCommitted={onCommitted}
            />
          </div>
        )}
        {mode === "photo" && <PhotoFlow conf={conf} geo={geo} onOpenSegment={onOpenSegment} onCommitted={onCommitted} />}
        {mode === "video" && <VideoFlow conf={conf} onDone={onCommitted} onViewMap={onViewMap} />}
      </div>
    </div>
  );
}
