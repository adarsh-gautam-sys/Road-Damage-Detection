const EARTH_RADIUS_M = 6_371_000;

const toRad = (deg: number): number => (deg * Math.PI) / 180;

/** Great-circle distance in metres. */
export function haversineMeters(lat1: number, lng1: number, lat2: number, lng2: number): number {
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(a)));
}

export interface PatrolStamp {
  lat: number | null;
  lng: number | null;
  /** epoch milliseconds */
  t: number;
}

export interface PatrolGate {
  auto_patrol_min_distance_m: number;
  auto_patrol_min_interval_s: number;
}

/**
 * Auto-patrol commit rule: a detection is committed only if it is the first one, or the
 * device moved >= min distance, or >= min interval elapsed since the last commit.
 * Thresholds come from `configs/app.yaml` via `/api/v1/config`.
 */
export function shouldAutoLog(args: {
  hasDetection: boolean;
  now: PatrolStamp;
  last: PatrolStamp | null;
  cfg: PatrolGate;
}): boolean {
  const { hasDetection, now, last, cfg } = args;
  if (!hasDetection) return false;
  if (last === null) return true;

  const elapsedS = (now.t - last.t) / 1000;
  if (elapsedS >= cfg.auto_patrol_min_interval_s) return true;

  if (now.lat !== null && now.lng !== null && last.lat !== null && last.lng !== null) {
    return haversineMeters(last.lat, last.lng, now.lat, now.lng) >= cfg.auto_patrol_min_distance_m;
  }
  return false;
}
