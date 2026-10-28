import { useState } from "react";
import { api, assetUrl } from "../api/client";
import type { SegmentStatus } from "../api/types";
import { LedgerTable } from "../components/Ledger";
import { SeverityChip } from "../components/Severity";
import { ErrorNote, PlainButton, Skeleton } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { CLASS_NAMES, formatCoord } from "../lib/severity";
import { NEXT_STATUS, STATUS_ACTION, STATUS_LABEL } from "../lib/status";
import { useConfig } from "../state/ConfigContext";

export function SegmentDetailView({ id, onChanged }: { id: number; onChanged: () => void }) {
  const cfg = useConfig();
  const { data, error, loading, reload } = useAsync(() => api.segment(id), [id]);
  const [busy, setBusy] = useState<SegmentStatus | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const change = async (to: SegmentStatus) => {
    setBusy(to);
    setActionError(null);
    try {
      await api.setSegmentStatus(id, to);
      reload();
      onChanged();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Couldn't change the status.");
    } finally {
      setBusy(null);
    }
  };

  if (loading && !data) {
    return (
      <div className="flex flex-col gap-3" aria-busy="true">
        <Skeleton className="h-16" />
        <Skeleton className="h-40" />
        <Skeleton className="h-32" />
      </div>
    );
  }
  if (error || !data) return <ErrorNote message={error?.message ?? "Segment not found."} onRetry={reload} />;

  const next = NEXT_STATUS[data.status];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <SeverityChip label={data.priority_label} score={data.priority_score} className="text-[14px]" />
        <p className="text-ink-2">
          {data.road_class} road · {data.n_detections} defect{data.n_detections === 1 ? "" : "s"}
        </p>
        <p className="tnum w-full font-mono text-[13px] text-ink-3">{formatCoord(data.centroid_lat, data.centroid_lng)}</p>
      </div>

      <section aria-labelledby="ledger-h">
        <h3 id="ledger-h" className="mb-1 font-label text-base font-semibold">
          Why this rank
        </h3>
        <LedgerTable
          terms={data.terms}
          weighted={data.weighted_terms}
          weights={cfg.priority.weights}
          total={data.priority_score}
        />
        <p className="mt-2 text-[13px] text-ink-3">
          Segments group defects within {cfg.priority.eps_meters} m. Bands: Low up to {cfg.priority.bands.low_max}, Medium up to{" "}
          {cfg.priority.bands.medium_max}, High above.
        </p>
      </section>

      <section aria-labelledby="status-h">
        <h3 id="status-h" className="mb-2 font-label text-base font-semibold">
          Work order: <span className="text-gold">{STATUS_LABEL[data.status]}</span>
        </h3>
        {next.length === 0 ? (
          <p className="text-ink-2">This segment is closed. No further changes.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {next.map((s) => (
              <PlainButton key={s} onClick={() => void change(s)} disabled={busy !== null}>
                {busy === s ? "Saving…" : STATUS_ACTION[s]}
              </PlainButton>
            ))}
          </div>
        )}
        {actionError && (
          <div className="mt-2">
            <ErrorNote message={actionError} />
          </div>
        )}
      </section>

      <section aria-labelledby="det-h">
        <h3 id="det-h" className="mb-2 font-label text-base font-semibold">
          Defects in this segment
        </h3>
        <ul className="flex flex-col divide-y divide-etch rounded-lg border border-etch">
          {data.detections.map((d) => (
            <li key={d.id} className="flex items-center gap-3 p-2.5">
              {d.thumb_url ? (
                <a href={assetUrl(d.annotated_url ?? d.thumb_url)} target="_blank" rel="noreferrer" className="shrink-0">
                  <img
                    src={assetUrl(d.thumb_url)}
                    alt={`${CLASS_NAMES[d.class_code]} photo`}
                    loading="lazy"
                    width={72}
                    height={72}
                    className="h-[72px] w-[72px] rounded border border-etch-strong object-cover"
                  />
                </a>
              ) : (
                <div className="grid h-[72px] w-[72px] shrink-0 place-items-center rounded border border-etch text-[11px] text-ink-3">video</div>
              )}
              <div className="min-w-0 flex-1">
                <p className="font-label text-[15px] font-semibold">
                  <span className="font-mono">{d.class_code}</span> {CLASS_NAMES[d.class_code]}
                </p>
                <p className="tnum font-mono text-[12px] text-ink-2">
                  {d.measure} {d.extent.toFixed(3)} · conf {d.confidence.toFixed(2)}
                </p>
              </div>
              <SeverityChip label={d.severity_label} score={d.severity_score} />
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
