import { CLASS_NAMES } from "../lib/severity";
import type { PreviewDetection } from "../api/types";
import { SeverityChip } from "./Severity";

/** Recessed transflective register that shows the measured numbers for one defect. */
export function LcdReading({ det, empty }: { det: PreviewDetection | null; empty: string }) {
  if (!det) {
    return (
      <div className="flex min-h-[52px] items-center">
        <p className="font-mono text-[13px] tracking-wide text-lcd-ink/70">{empty}</p>
      </div>
    );
  }
  const { score, label, extent, measure } = det.severity;
  return (
    <div className="grid gap-1" aria-live="polite">
      <div className="flex items-center justify-between gap-3">
        <p className="min-w-0 truncate font-label text-[15px] font-bold uppercase tracking-wide">
          <span className="font-mono">{det.class_code}</span> {CLASS_NAMES[det.class_code]}
        </p>
        <SeverityChip label={label} />
      </div>
      <p className="tnum font-mono text-[13px] leading-snug">
        {measure} = <b>{extent.toFixed(3)}</b> · S = <b>{score.toFixed(1)}</b>/100 · conf <b>{det.confidence.toFixed(2)}</b>
      </p>
    </div>
  );
}
