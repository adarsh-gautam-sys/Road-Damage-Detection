import type { SeverityLabel } from "../api/types";
import { severityMeta } from "../lib/severity";

/** Severity shape: triangle = High, diamond = Medium, circle = Low. Paired with a word, never alone. */
export function SeverityGlyph({ label, size = 12, color }: { label: SeverityLabel; size?: number; color?: string }) {
  const { glyph, cssVar } = severityMeta(label);
  const fill = color ?? `var(${cssVar})`;
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" focusable="false">
      {glyph === "triangle" && <path d="M6 1 11.2 10.6H0.8Z" fill={fill} />}
      {glyph === "diamond" && <path d="M6 0.8 11.2 6 6 11.2 0.8 6Z" fill={fill} />}
      {glyph === "circle" && <circle cx="6" cy="6" r="5" fill={fill} />}
    </svg>
  );
}

const CHIP_STYLE: Record<SeverityLabel, string> = {
  High: "bg-sev-high text-white",
  Medium: "bg-sev-med text-gold-ink",
  Low: "bg-sev-low text-white",
};

export function SeverityChip({ label, score, className = "" }: { label: SeverityLabel; score?: number; className?: string }) {
  const { word } = severityMeta(label);
  const style = CHIP_STYLE[label] ?? "bg-surface-2 text-ink";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 font-label text-[12px] font-bold tracking-wider ${style} ${className}`}
      title={`${label ?? "Unknown"} severity`}
    >
      <SeverityGlyph label={label} size={10} color="currentColor" />
      <span>{word}</span>
      {score !== undefined && <span className="tnum font-mono font-medium">{score.toFixed(1)}</span>}
    </span>
  );
}
