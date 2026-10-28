import type { ClassCode, PriorityTerms, SeverityLabel } from "../api/types";

export interface SeverityMeta {
  word: string;
  /** Shape drawn by <SeverityGlyph>; severity is never conveyed by colour alone. */
  glyph: "triangle" | "diamond" | "circle";
  cssVar: string;
}

const META: Record<SeverityLabel, SeverityMeta> = {
  High: { word: "HIGH", glyph: "triangle", cssVar: "--sev-high" },
  Medium: { word: "MED", glyph: "diamond", cssVar: "--sev-med" },
  Low: { word: "LOW", glyph: "circle", cssVar: "--sev-low" },
};

const FALLBACK_META: SeverityMeta = { word: "UNKNOWN", glyph: "circle", cssVar: "--ink-3" };

export function severityMeta(label: SeverityLabel | string | null | undefined): SeverityMeta {
  if (!label) return FALLBACK_META;
  return META[label as SeverityLabel] ?? FALLBACK_META;
}

export const CLASS_NAMES: Record<ClassCode, string> = {
  D00: "Longitudinal crack",
  D10: "Transverse crack",
  D20: "Alligator crack",
  D40: "Pothole",
};

export const CLASS_COLOR_VAR: Record<ClassCode, string> = {
  D40: "--sev-high",
  D20: "--sev-med",
  D00: "--cls-linear",
  D10: "--cls-linear",
};

export const PRIORITY_TERM_ROWS: Array<{ key: keyof PriorityTerms; symbol: string; label: string; weightKey: string }> = [
  { key: "s_max", symbol: "S", label: "Worst severity", weightKey: "s_max" },
  { key: "density", symbol: "D", label: "Defect density", weightKey: "density" },
  { key: "road_importance", symbol: "R", label: "Road importance", weightKey: "road_importance" },
  { key: "confidence", symbol: "C", label: "Model confidence", weightKey: "confidence" },
];

/** Rows + total for the right-aligned priority ledger. Values are the already-weighted terms. */
export function formatLedger(weighted: Partial<PriorityTerms>): {
  rows: Array<{ key: keyof PriorityTerms; value: number }>;
  total: number;
} {
  const rows = PRIORITY_TERM_ROWS.map((r) => ({ key: r.key, value: Number(weighted[r.key] ?? 0) }));
  const total = rows.reduce((acc, r) => acc + r.value, 0);
  return { rows, total };
}

export function formatCoord(lat: number | null | undefined, lng: number | null | undefined): string {
  if (lat == null || lng == null) return "no fix";
  const ns = lat >= 0 ? "N" : "S";
  const ew = lng >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(5)}°${ns} ${Math.abs(lng).toFixed(5)}°${ew}`;
}
