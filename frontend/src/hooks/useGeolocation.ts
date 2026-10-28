import { useEffect, useRef, useState } from "react";

export interface GeoFix {
  lat: number;
  lng: number;
  accuracy: number;
  /** epoch ms */
  t: number;
}

export type GeoStatus = "idle" | "acquiring" | "live" | "denied" | "unavailable";

/** Continuous GPS while `enabled`. Needs HTTPS (or localhost) in real browsers. */
export function useGeolocation(enabled: boolean): { status: GeoStatus; fix: GeoFix | null; error: string | null } {
  const [status, setStatus] = useState<GeoStatus>("idle");
  const [fix, setFix] = useState<GeoFix | null>(null);
  const [error, setError] = useState<string | null>(null);
  const watchId = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) {
      setStatus("idle");
      return;
    }
    if (!("geolocation" in navigator)) {
      setStatus("unavailable");
      setError("This browser has no location support.");
      return;
    }
    setStatus("acquiring");
    watchId.current = navigator.geolocation.watchPosition(
      (pos) => {
        setFix({
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
          t: Date.now(),
        });
        setStatus("live");
        setError(null);
      },
      (err) => {
        if (err.code === err.PERMISSION_DENIED) {
          setStatus("denied");
          setError("Location permission was denied.");
        } else {
          setStatus("unavailable");
          setError(err.code === err.TIMEOUT ? "Timed out waiting for a GPS fix." : "No GPS fix available.");
        }
      },
      { enableHighAccuracy: true, maximumAge: 2000, timeout: 20000 },
    );
    return () => {
      if (watchId.current !== null) navigator.geolocation.clearWatch(watchId.current);
      watchId.current = null;
    };
  }, [enabled]);

  return { status, fix, error };
}
