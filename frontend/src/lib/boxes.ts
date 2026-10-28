export interface Size {
  w: number;
  h: number;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Map a box in source-image pixels onto a viewport that displays the source with
 * `object-fit: cover` (centre-cropped), so overlays line up with what the user sees.
 */
export function mapBoxToViewport(
  box: [number, number, number, number],
  source: Size,
  viewport: Size,
): Rect {
  const scale = Math.max(viewport.w / source.w, viewport.h / source.h);
  const offsetX = (source.w * scale - viewport.w) / 2;
  const offsetY = (source.h * scale - viewport.h) / 2;
  const [x1, y1, x2, y2] = box;
  return {
    x: x1 * scale - offsetX,
    y: y1 * scale - offsetY,
    w: (x2 - x1) * scale,
    h: (y2 - y1) * scale,
  };
}

export interface SelectionAnchor {
  class_code: string;
  /** centre in source-image pixels */
  cx: number;
  cy: number;
}

/**
 * Live boxes shift every frame, so a selection is remembered as (class, centre) and
 * re-resolved to the nearest same-class box. Returns its index, or null if it left the frame.
 */
export function resolveSelected(
  dets: Array<{ class_code: string; box: [number, number, number, number] }>,
  anchor: SelectionAnchor | null,
  source: Size,
  maxJumpFraction = 0.25,
): number | null {
  if (!anchor || dets.length === 0) return null;
  const limit = Math.hypot(source.w, source.h) * maxJumpFraction;
  let best: number | null = null;
  let bestDist = Infinity;
  dets.forEach((d, i) => {
    if (d.class_code !== anchor.class_code) return;
    const cx = (d.box[0] + d.box[2]) / 2;
    const cy = (d.box[1] + d.box[3]) / 2;
    const dist = Math.hypot(cx - anchor.cx, cy - anchor.cy);
    if (dist <= limit && dist < bestDist) {
      best = i;
      bestDist = dist;
    }
  });
  return best;
}

/** Index of the smallest rect containing (px, py), or null. */
export function hitTest(rects: Rect[], px: number, py: number): number | null {
  let best: number | null = null;
  let bestArea = Infinity;
  rects.forEach((r, i) => {
    if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) {
      const area = r.w * r.h;
      if (area < bestArea) {
        best = i;
        bestArea = area;
      }
    }
  });
  return best;
}
