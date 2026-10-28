import { describe, expect, it } from "vitest";
import { shouldAutoLog, haversineMeters } from "./geo";
import { mapBoxToViewport, resolveSelected, hitTest } from "./boxes";
import { computeResizeDims } from "./image";
import { severityMeta, formatLedger } from "./severity";

const cfg = { auto_patrol_min_distance_m: 10, auto_patrol_min_interval_s: 5 };

describe("haversineMeters", () => {
  it("is zero for identical points and ~111 km per degree of latitude", () => {
    expect(haversineMeters(28.45, 77.02, 28.45, 77.02)).toBe(0);
    const d = haversineMeters(28.0, 77.0, 29.0, 77.0);
    expect(d).toBeGreaterThan(110_000);
    expect(d).toBeLessThan(112_500);
  });
});

describe("shouldAutoLog (auto-patrol gating)", () => {
  const base = { lat: 28.4595, lng: 77.0266, t: 100_000 };

  it("never logs when there is nothing detected", () => {
    expect(shouldAutoLog({ hasDetection: false, now: base, last: null, cfg })).toBe(false);
  });

  it("logs the first detection immediately", () => {
    expect(shouldAutoLog({ hasDetection: true, now: base, last: null, cfg })).toBe(true);
  });

  it("blocks when neither far enough nor long enough", () => {
    const now = { ...base, t: base.t + 2_000, lat: base.lat + 0.00002 }; // ~2 m, 2 s
    expect(shouldAutoLog({ hasDetection: true, now, last: base, cfg })).toBe(false);
  });

  it("allows once moved >= min distance", () => {
    const now = { ...base, t: base.t + 1_000, lat: base.lat + 0.00015 }; // ~16 m, 1 s
    expect(shouldAutoLog({ hasDetection: true, now, last: base, cfg })).toBe(true);
  });

  it("allows once the interval has elapsed even when stationary", () => {
    const now = { ...base, t: base.t + 5_000 };
    expect(shouldAutoLog({ hasDetection: true, now, last: base, cfg })).toBe(true);
  });

  it("without GPS it falls back to the time rule only", () => {
    const last = { lat: null, lng: null, t: 0 };
    expect(
      shouldAutoLog({ hasDetection: true, now: { lat: null, lng: null, t: 3_000 }, last, cfg }),
    ).toBe(false);
    expect(
      shouldAutoLog({ hasDetection: true, now: { lat: null, lng: null, t: 6_000 }, last, cfg }),
    ).toBe(true);
  });
});

describe("mapBoxToViewport (object-fit: cover)", () => {
  it("is a straight scale when aspect ratios match", () => {
    const r = mapBoxToViewport([100, 50, 300, 150], { w: 640, h: 480 }, { w: 320, h: 240 });
    expect(r).toEqual({ x: 50, y: 25, w: 100, h: 50 });
  });

  it("crops left/right when the viewport is narrower than the source (cover)", () => {
    // source 640x480, viewport 240x240 => scale 0.5 (height), x offset -(320-240)/2 = -40
    const r = mapBoxToViewport([0, 0, 640, 480], { w: 640, h: 480 }, { w: 240, h: 240 });
    expect(r.x).toBeCloseTo(-40);
    expect(r.w).toBeCloseTo(320);
    expect(r.h).toBeCloseTo(240);
  });
});

describe("resolveSelected (selection survives frame jitter)", () => {
  const mk = (code: string, box: [number, number, number, number]) => ({ class_code: code, box });
  const anchor = { class_code: "D40", cx: 200, cy: 200 };

  it("re-finds the same-class box nearest the anchor after the box moves slightly", () => {
    const dets = [mk("D20", [190, 190, 210, 210]), mk("D40", [215, 205, 255, 245]), mk("D40", [500, 500, 540, 540])];
    expect(resolveSelected(dets, anchor, { w: 640, h: 480 })).toBe(1);
  });

  it("drops the selection when no same-class box is near", () => {
    const dets = [mk("D40", [600, 400, 640, 440])];
    expect(resolveSelected(dets, anchor, { w: 640, h: 480 })).toBeNull();
  });

  it("returns null with no anchor or no detections", () => {
    expect(resolveSelected([], anchor, { w: 640, h: 480 })).toBeNull();
    expect(resolveSelected([mk("D40", [0, 0, 10, 10])], null, { w: 640, h: 480 })).toBeNull();
  });
});

describe("hitTest", () => {
  it("prefers the smallest box containing the point", () => {
    const rects = [
      { x: 0, y: 0, w: 300, h: 300 },
      { x: 100, y: 100, w: 50, h: 50 },
    ];
    expect(hitTest(rects, 120, 120)).toBe(1);
    expect(hitTest(rects, 250, 250)).toBe(0);
    expect(hitTest(rects, 400, 400)).toBeNull();
  });
});

describe("computeResizeDims", () => {
  it("never upscales and keeps aspect ratio", () => {
    expect(computeResizeDims(640, 480, 1280)).toEqual({ w: 640, h: 480 });
    expect(computeResizeDims(4000, 3000, 1280)).toEqual({ w: 1280, h: 960 });
    expect(computeResizeDims(3000, 4000, 1280)).toEqual({ w: 960, h: 1280 });
  });
});

describe("severity presentation", () => {
  it("always pairs a word and a glyph with the colour (never colour alone)", () => {
    for (const label of ["Low", "Medium", "High"] as const) {
      const m = severityMeta(label);
      expect(m.word.length).toBeGreaterThan(2);
      expect(m.glyph).toMatch(/^(triangle|diamond|circle)$/);
      expect(m.cssVar).toMatch(/^--sev-/);
    }
  });

  it("formats the four weighted terms so they sum to the shown priority", () => {
    const ledger = formatLedger({ s_max: 35.3, density: 10, road_importance: 20, confidence: 8.9 });
    expect(ledger.total).toBeCloseTo(74.2, 1);
    expect(ledger.rows.map((r) => r.key)).toEqual(["s_max", "density", "road_importance", "confidence"]);
  });
});
