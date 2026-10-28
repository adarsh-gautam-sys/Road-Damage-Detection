import { Camera, Map as MapIcon, ListOrdered, ScrollText, ChevronLeft, WifiOff } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { Sheet } from "./components/Sheet";
import { PlainButton, Skeleton } from "./components/States";
import { useGeolocation } from "./hooks/useGeolocation";
import { useMediaQuery } from "./hooks/useMediaQuery";
import { ConfigProvider, useConfig, useConfigState } from "./state/ConfigContext";
import { AuditView } from "./views/AuditView";
import { CaptureView } from "./views/CaptureView";
import { MapView } from "./views/MapView";
import { QueueView } from "./views/QueueView";
import { SegmentDetailView } from "./views/SegmentDetailView";

type View = "capture" | "map" | "queue" | "audit";

const NAV: Array<{ id: View; label: string; Icon: typeof Camera }> = [
  { id: "capture", label: "Capture", Icon: Camera },
  { id: "map", label: "Map", Icon: MapIcon },
  { id: "queue", label: "Priority", Icon: ListOrdered },
  { id: "audit", label: "Model", Icon: ScrollText },
];

function Header({ gps, online, conf }: { gps: string; online: boolean; conf: number }) {
  const cfg = useConfig();
  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-etch bg-chassis px-3 pt-[env(safe-area-inset-top)] box-content">
      <svg width="24" height="24" viewBox="0 0 64 64" aria-hidden="true" className="shrink-0">
        <path d="M10 22v-12h12M54 22v-12h-12M10 42v12h12M54 42v12h-12" fill="none" stroke="#f2c200" strokeWidth="5" strokeLinecap="square" />
        <path d="M32 24v16M24 32h16" stroke="#f3f4f6" strokeWidth="4" strokeLinecap="square" />
      </svg>
      <span className="hidden font-label text-[15px] font-bold tracking-[0.12em] uppercase sm:inline">Road Damage AI</span>
      <div className="ml-auto flex items-center gap-1.5 font-mono text-[12px]">
        <span className="rounded border border-etch-strong px-2 py-1 text-ink-2">{gps}</span>
        <span className="hidden rounded border border-etch-strong px-2 py-1 text-ink-2 sm:inline">
          {cfg.model.name} · conf {conf.toFixed(3)}
        </span>
        <span className={`inline-flex items-center gap-1 rounded border px-2 py-1 ${online ? "border-etch-strong text-ink-2" : "border-sev-high/60 text-sev-high-text"}`}>
          <span className={`inline-block h-2 w-2 rounded-full ${online ? "bg-sev-low-text" : "bg-sev-high-text"}`} aria-hidden="true" />
          {online ? "Online" : "Offline"}
        </span>
      </div>
    </header>
  );
}

function Shell() {
  const cfg = useConfig();
  const isWide = useMediaQuery("(min-width: 1024px)");
  const [view, setView] = useState<View>("capture");
  const [conf, setConf] = useState(cfg.default_conf);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [online, setOnline] = useState(true);
  const geo = useGeolocation(true);

  const bump = useCallback(() => setRefreshKey((k) => k + 1), []);

  useEffect(() => {
    let live = true;
    const ping = () =>
      api
        .health()
        .then(() => live && setOnline(true))
        .catch(() => live && setOnline(false));
    void ping();
    const t = window.setInterval(ping, 15000);
    const on = () => void ping();
    window.addEventListener("online", on);
    window.addEventListener("offline", () => setOnline(false));
    return () => {
      live = false;
      window.clearInterval(t);
      window.removeEventListener("online", on);
    };
  }, []);

  const openSegment = useCallback((id: number) => {
    setSelectedId(id);
    setView((v) => (v === "queue" ? "queue" : "map"));
  }, []);

  const gpsLabel =
    geo.status === "live" && geo.fix
      ? `GPS ±${geo.fix.accuracy.toFixed(0)} m`
      : geo.status === "acquiring"
        ? "GPS acquiring"
        : geo.status === "denied"
          ? "GPS denied"
          : geo.status === "unavailable"
            ? "No GPS"
            : "GPS off";

  const split = isWide && (view === "map" || view === "queue");
  const closeDetail = useCallback(() => setSelectedId(null), []);

  const goCapture = () => setView("capture");

  const detail = selectedId !== null && (
    <SegmentDetailView id={selectedId} onChanged={bump} />
  );

  return (
    <div className="flex h-dvh flex-col lg:flex-row">
      {/* Desktop rail */}
      <nav aria-label="Main" className="hidden w-[84px] shrink-0 flex-col items-stretch gap-1 border-r border-etch bg-chassis p-2 lg:flex">
        {NAV.filter((n) => n.id !== "queue").map(({ id, label, Icon }) => {
          const active = id === view || (id === "map" && view === "queue");
          return (
            <button
              key={id}
              type="button"
              aria-current={active ? "page" : undefined}
              onClick={() => setView(id)}
              className={`flex min-h-16 flex-col items-center justify-center gap-1 rounded-md font-label text-[12px] font-bold uppercase tracking-wider ${
                active ? "bg-raised text-gold ring-1 ring-etch-strong" : "text-ink-2 hover:text-ink"
              }`}
            >
              <Icon size={22} aria-hidden="true" />
              {id === "map" ? "Command" : label}
            </button>
          );
        })}
      </nav>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Header gps={gpsLabel} online={online} conf={conf} />
        {!online && (
          <p role="status" className="flex shrink-0 items-center gap-2 bg-sev-high/15 px-3 py-2 text-sm text-sev-high-text">
            <WifiOff size={16} aria-hidden="true" /> You're offline. Readings can't be saved until the connection returns.
          </p>
        )}

        <main className="min-h-0 flex-1 overflow-hidden">
          {view === "capture" && (
            <div className="h-full overflow-y-auto">
              <CaptureView
                conf={conf}
                onConfChange={setConf}
                geo={geo}
                onOpenSegment={openSegment}
                onViewMap={() => setView("map")}
                onCommitted={bump}
              />
            </div>
          )}

          {view === "audit" && (
            <div className="h-full overflow-y-auto">
              <AuditView />
            </div>
          )}

          {split ? (
            <div className="flex h-full">
              <div className="min-w-0 flex-1">
                <MapView refreshKey={refreshKey} selectedId={selectedId} onSelect={setSelectedId} onGoCapture={goCapture} fix={geo.fix} />
              </div>
              <aside aria-label="Segments" className="flex w-[420px] shrink-0 flex-col border-l border-etch bg-chassis">
                {selectedId !== null ? (
                  <>
                    <div className="flex shrink-0 items-center gap-2 border-b border-etch px-2 py-2">
                      <PlainButton onClick={closeDetail}>
                        <ChevronLeft size={16} aria-hidden="true" /> Queue
                      </PlainButton>
                      <h2 className="font-label text-lg font-semibold">Segment {selectedId}</h2>
                    </div>
                    <div className="min-h-0 flex-1 overflow-y-auto p-4">{detail}</div>
                  </>
                ) : (
                  <QueueView refreshKey={refreshKey} selectedId={selectedId} onSelect={setSelectedId} onGoCapture={goCapture} />
                )}
              </aside>
            </div>
          ) : (
            <>
              {view === "map" && (
                <MapView refreshKey={refreshKey} selectedId={selectedId} onSelect={setSelectedId} onGoCapture={goCapture} fix={geo.fix} />
              )}
              {view === "queue" && (
                <QueueView refreshKey={refreshKey} selectedId={selectedId} onSelect={setSelectedId} onGoCapture={goCapture} />
              )}
            </>
          )}
        </main>

        {/* Phone tab bar */}
        <nav aria-label="Main" className="grid shrink-0 grid-cols-4 border-t border-etch bg-chassis pb-[env(safe-area-inset-bottom)] lg:hidden">
          {NAV.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              aria-current={view === id ? "page" : undefined}
              onClick={() => setView(id)}
              className={`flex min-h-14 flex-col items-center justify-center gap-0.5 font-label text-[12px] font-bold uppercase tracking-wider ${
                view === id ? "text-gold" : "text-ink-2"
              }`}
            >
              <Icon size={22} aria-hidden="true" />
              {label}
            </button>
          ))}
        </nav>
      </div>

      {!isWide && selectedId !== null && (view === "map" || view === "queue") && (
        <Sheet title={`Segment ${selectedId}`} onClose={closeDetail}>
          {detail}
        </Sheet>
      )}
    </div>
  );
}

function Boot() {
  const { config, error, reload } = useConfigState();
  if (config) return <Shell />;
  return (
    <div className="mx-auto flex h-dvh max-w-md flex-col justify-center gap-4 p-6">
      {error ? (
        <>
          <h1 className="font-label text-2xl font-bold">Can't reach the server</h1>
          <p className="text-ink-2">
            {error} Start the backend with <span className="font-mono text-[13px] text-ink">make run</span> and check this device can reach it.
          </p>
          <PlainButton onClick={reload}>Try again</PlainButton>
        </>
      ) : (
        <div aria-busy="true" aria-label="Connecting" className="flex flex-col gap-3">
          <Skeleton className="h-10 w-2/3" />
          <Skeleton className="h-48" />
          <Skeleton className="h-14" />
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <ConfigProvider>
      <Boot />
    </ConfigProvider>
  );
}
