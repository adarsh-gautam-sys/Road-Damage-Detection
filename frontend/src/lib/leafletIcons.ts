import L from "leaflet";
import type { SeverityLabel } from "../api/types";
import { severityMeta } from "./severity";

const FILL: Record<SeverityLabel, string> = { High: "#e02828", Medium: "#e59400", Low: "#23805f" };
const INK: Record<SeverityLabel, string> = { High: "#ffffff", Medium: "#141412", Low: "#ffffff" };

function shapePath(glyph: "triangle" | "diamond" | "circle"): string {
  if (glyph === "triangle") return "M20 3 38 35H2Z";
  if (glyph === "diamond") return "M20 2 38 20 20 38 2 20Z";
  return "M20 20m-17 0a17 17 0 1 0 34 0a17 17 0 1 0-34 0";
}

/**
 * Map marker: severity shape + colour + the number inside. Shape and number mean it never
 * relies on colour alone. `size` grows with importance.
 */
export function severityIcon(label: SeverityLabel, text: string, size = 38, selected = false): L.DivIcon {
  const { glyph } = severityMeta(label);
  const stroke = selected ? "#f2c200" : "#181b1f";
  const sw = selected ? 3.5 : 2;
  const ty = glyph === "triangle" ? 29 : 25;
  const html = `<svg width="${size}" height="${size}" viewBox="0 0 40 40" role="img" aria-label="${label} ${text}">
    <path d="${shapePath(glyph)}" fill="${FILL[label]}" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"/>
    <text x="20" y="${ty}" text-anchor="middle" font-family="JetBrains Mono, monospace" font-weight="700" font-size="12" fill="${INK[label]}">${text}</text>
  </svg>`;
  return L.divIcon({ html, className: "rd-marker", iconSize: [size, size], iconAnchor: [size / 2, size / 2] });
}

export function pinIcon(): L.DivIcon {
  const html = `<svg width="36" height="36" viewBox="0 0 36 36" aria-hidden="true">
    <circle cx="18" cy="18" r="11" fill="rgba(24,27,31,.55)" stroke="#f2c200" stroke-width="2.5"/>
    <path d="M18 4v9M18 23v9M4 18h9M23 18h9" stroke="#f3f4f6" stroke-width="2" stroke-linecap="square"/>
  </svg>`;
  return L.divIcon({ html, className: "rd-marker", iconSize: [36, 36], iconAnchor: [18, 18] });
}
