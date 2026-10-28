import { Download, ListOrdered } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import type { SegmentStatus, SeverityLabel } from "../api/types";
import { LedgerLine } from "../components/Ledger";
import { SeverityChip } from "../components/Severity";
import { EmptyState, ErrorNote, PlainButton, Skeleton } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { ALL_STATUSES, STATUS_LABEL } from "../lib/status";

interface Props {
  refreshKey: number;
  selectedId: number | null;
  onSelect: (id: number) => void;
  onGoCapture: () => void;
}

const LABELS: SeverityLabel[] = ["High", "Medium", "Low"];

export function QueueView({ refreshKey, selectedId, onSelect, onGoCapture }: Props) {
  const [label, setLabel] = useState<SeverityLabel | "">("");
  const [status, setStatus] = useState<SegmentStatus | "">("");
  const { data, error, loading, reload } = useAsync(
    () => api.segments({ label: label || undefined, status: status || undefined }),
    [label, status, refreshKey],
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-etch px-3 py-2">
        <div role="group" aria-label="Filter by priority" className="flex gap-1">
          {(["", ...LABELS] as const).map((l) => (
            <button
              key={l || "all"}
              type="button"
              aria-pressed={label === l}
              onClick={() => setLabel(l)}
              className={`min-h-11 rounded-md border px-3 font-label text-sm font-bold uppercase tracking-wider ${
                label === l ? "border-gold bg-raised text-gold" : "border-etch text-ink-2 hover:text-ink"
              }`}
            >
              {l || "All"}
            </button>
          ))}
        </div>
        <label className="ml-auto flex items-center gap-2">
          <span className="label">Status</span>
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value as SegmentStatus | "")}
            className="min-h-11 rounded-md border border-etch-strong bg-panel px-2 text-ink"
          >
            <option value="">Any</option>
            {ALL_STATUSES.map((s) => (
              <option key={s} value={s}>
                {STATUS_LABEL[s]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {loading && !data && (
          <div className="flex flex-col gap-2 p-3" aria-busy="true" aria-label="Loading segments">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-[76px]" />
            ))}
          </div>
        )}
        {error && (
          <div className="p-3">
            <ErrorNote message={error.message} onRetry={reload} />
          </div>
        )}
        {data && data.items.length === 0 && (
          <EmptyState
            icon={<ListOrdered size={26} aria-hidden="true" />}
            title={label || status ? "No segments match" : "No segments yet"}
            body={
              label || status
                ? "Clear a filter to see the rest of the queue."
                : "Log a located reading and nearby defects are grouped into road segments, ranked here."
            }
            action={!label && !status ? <PlainButton onClick={onGoCapture}>Go to capture</PlainButton> : undefined}
          />
        )}
        {data && data.items.length > 0 && (
          <ol className="divide-y divide-etch">
            {data.items.map((s, i) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => onSelect(s.id)}
                  aria-current={selectedId === s.id}
                  className={`flex w-full flex-col gap-1.5 px-3 py-3 text-left hover:bg-panel ${selectedId === s.id ? "bg-panel ring-1 ring-gold ring-inset" : ""}`}
                >
                  <span className="flex items-center gap-3">
                    <span className="tnum w-6 shrink-0 font-mono text-[13px] text-ink-3">{i + 1}</span>
                    <span className="min-w-0 flex-1">
                      <span className="font-label text-base font-semibold">Segment {s.id}</span>
                      <span className="ml-2 text-[13px] text-ink-2">
                        {s.road_class} · {s.n_detections} defect{s.n_detections === 1 ? "" : "s"} · {STATUS_LABEL[s.status]}
                      </span>
                    </span>
                    <SeverityChip label={s.priority_label} score={s.priority_score} />
                  </span>
                  <LedgerLine weighted={s.weighted_terms} total={s.priority_score} />
                </button>
              </li>
            ))}
          </ol>
        )}
      </div>

      <div className="flex shrink-0 flex-wrap gap-2 border-t border-etch px-3 py-2">
        <a
          href={api.exportDetectionsCsvUrl()}
          download
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-etch-strong px-3 font-label text-sm font-semibold uppercase tracking-wider hover:bg-raised"
        >
          <Download size={16} aria-hidden="true" /> Detections CSV
        </a>
        <a
          href={api.exportSegmentsGeojsonUrl()}
          download="segments.geojson"
          className="inline-flex min-h-11 items-center gap-2 rounded-md border border-etch-strong px-3 font-label text-sm font-semibold uppercase tracking-wider hover:bg-raised"
        >
          <Download size={16} aria-hidden="true" /> Segments GeoJSON
        </a>
      </div>
    </div>
  );
}
