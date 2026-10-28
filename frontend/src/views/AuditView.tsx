import { api } from "../api/client";
import type { ClassCode, ExperimentMetrics } from "../api/types";
import { ErrorNote, Skeleton } from "../components/States";
import { useAsync } from "../hooks/useAsync";
import { CLASS_NAMES } from "../lib/severity";
import { useConfig } from "../state/ConfigContext";

const CLASSES: ClassCode[] = ["D00", "D10", "D20", "D40"];
const f3 = (n: number | null | undefined) => (typeof n === "number" ? n.toFixed(3) : "—");

/** Rows of the confusion matrix are ground truth; the last column is "missed" (background). */
function supportFor(e1: ExperimentMetrics, idx: number): number {
  return (e1.confusion_matrix[idx] ?? []).reduce((a, b) => a + b, 0);
}

export function AuditView() {
  const cfg = useConfig();
  const { data, error, loading, reload } = useAsync(() => api.metrics(), []);
  const e1 = data?.experiments?.E1;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 p-4 pb-10 lg:p-8">
      <header>
        <h1 className="font-label text-2xl font-bold tracking-wide">Model and method</h1>
        <p className="mt-1 max-w-prose text-ink-2">
          Every number on this page is read from the evaluation file <span className="font-mono text-[13px]">reports/metrics.json</span>{" "}
          or from the configuration files. None is typed into the interface.
        </p>
      </header>

      {loading && !data && (
        <div className="flex flex-col gap-3" aria-busy="true">
          <Skeleton className="h-40" />
          <Skeleton className="h-56" />
        </div>
      )}
      {error && <ErrorNote message={error.message} onRetry={reload} />}
      {data && !e1 && <p className="plate p-4 text-ink-2">No evaluation has been recorded yet. Run training and evaluation, then reload.</p>}

      {e1 && (
        <>
          <section aria-labelledby="res-h">
            <h2 id="res-h" className="mb-1 font-label text-lg font-semibold">
              Held-out test results, {cfg.model.name} ({e1.experiment_id})
            </h2>
            <p className="mb-3 text-sm text-ink-3">
              Split: {e1.split}. Operating threshold {f3(e1.optimal_val_conf)} (validation F1 {f3(e1.optimal_val_f1)}). Size {e1.params_m} M parameters.
            </p>
            <div className="overflow-x-auto">
              <table className="tnum w-full min-w-[460px] font-mono text-[13px]">
                <caption className="sr-only">Per-class precision, recall and average precision on the test split</caption>
                <thead>
                  <tr className="border-b border-etch-strong text-ink-3">
                    {["Class", "Test boxes", "Precision", "Recall", "AP@50", "AP@50:95"].map((h, i) => (
                      <th key={h} scope="col" className={`py-2 font-label text-[12px] font-semibold uppercase tracking-wider ${i === 0 ? "text-left" : "text-right"}`}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {CLASSES.map((c, i) => {
                    const m = e1.per_class[c];
                    return (
                      <tr key={c} className="border-b border-etch">
                        <th scope="row" className="py-2 text-left font-sans text-[14px] font-normal">
                          <span className="mr-2 font-mono">{c}</span>
                          {CLASS_NAMES[c]}
                        </th>
                        <td className="py-2 text-right text-ink-2">{supportFor(e1, i)}</td>
                        <td className="py-2 text-right">{f3(m.precision)}</td>
                        <td className="py-2 text-right">{f3(m.recall)}</td>
                        <td className="py-2 text-right text-ink">{f3(m.ap50)}</td>
                        <td className="py-2 text-right">{f3(m.ap50_95)}</td>
                      </tr>
                    );
                  })}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="row" className="py-2 text-left font-label text-[12px] font-semibold uppercase tracking-wider text-ink-2">All classes</th>
                    <td className="py-2 text-right text-ink-2">{CLASSES.reduce((a, _c, i) => a + supportFor(e1, i), 0)}</td>
                    <td className="py-2 text-right">{f3(e1.overall.precision)}</td>
                    <td className="py-2 text-right">{f3(e1.overall.recall)}</td>
                    <td className="py-2 text-right font-bold text-gold">{f3(e1.overall.map50)}</td>
                    <td className="py-2 text-right">{f3(e1.overall.map50_95)}</td>
                  </tr>
                </tfoot>
              </table>
            </div>

            {e1.metric_gate && (
              <div className="lcd mt-4 px-4 py-3 text-[13px]" role="note">
                <p>
                  <b>{e1.metric_gate.passed ? "Target met." : "Target not met."}</b> The goal was mAP@50 of at least{" "}
                  {e1.metric_gate.min_map50.toFixed(2)} and pothole AP@50 of at least {e1.metric_gate.min_pothole_ap50.toFixed(2)}.
                  The model scored {f3(e1.metric_gate.map50)} and {f3(e1.metric_gate.d40_ap50)}.
                </p>
                <p className="mt-1">
                  Test images come from road stretches the model never saw in training, so these numbers are stricter than a random image split.
                  Transverse cracks have only {supportFor(e1, 1)} test boxes, so their score is unstable.
                </p>
              </div>
            )}

            <p className="tnum mt-3 font-mono text-[13px] text-ink-2">
              Speed per image: {e1.gpu_ms} ms on the training GPU{e1.cpu_ms !== null ? `, ${e1.cpu_ms} ms on a laptop CPU` : ""}.
            </p>
          </section>

          <section aria-labelledby="fig-h">
            <h2 id="fig-h" className="mb-3 font-label text-lg font-semibold">
              Where it goes wrong
            </h2>
            <div className="grid gap-4 sm:grid-cols-2">
              <figure>
                <img
                  src={api.figureUrl("confusion_matrix_normalized.png")}
                  alt="Normalised confusion matrix of the four damage classes plus background"
                  loading="lazy"
                  className="w-full rounded-lg border border-etch bg-white"
                />
                <figcaption className="mt-1.5 text-[13px] text-ink-2">
                  Defects are rarely confused with each other. Most errors are damage missed as background, or background flagged as damage.
                </figcaption>
              </figure>
              <figure>
                <img
                  src={api.figureUrl("PR_curve.png")}
                  alt="Precision-recall curves per class"
                  loading="lazy"
                  className="w-full rounded-lg border border-etch bg-white"
                />
                <figcaption className="mt-1.5 text-[13px] text-ink-2">Precision against recall for each class on the validation set.</figcaption>
              </figure>
            </div>
          </section>
        </>
      )}

      <section aria-labelledby="sev-h">
        <h2 id="sev-h" className="mb-1 font-label text-lg font-semibold">
          How severity is scored
        </h2>
        <p className="mb-3 max-w-prose text-ink-2">
          The dataset has no severity labels, so severity is a geometric rule on the box. It measures how much of the picture the damage fills, not
          its depth, and a close small pothole can look bigger than a far deep one. Scores are 0 to 100: Low up to{" "}
          {cfg.severity.tier_bounds.low_upper}, Medium up to {cfg.severity.tier_bounds.medium_upper}, High above.
        </p>
        <div className="overflow-x-auto">
          <table className="tnum w-full min-w-[420px] font-mono text-[13px]">
            <caption className="sr-only">Severity measure and thresholds per class</caption>
            <thead>
              <tr className="border-b border-etch-strong text-ink-3">
                {["Class", "Measure", "Score 0 at", "Score 100 at"].map((h, i) => (
                  <th key={h} scope="col" className={`py-2 font-label text-[12px] font-semibold uppercase tracking-wider ${i === 0 ? "text-left" : "text-right"}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CLASSES.map((c) => {
                const k = cfg.severity.classes[c];
                return (
                  <tr key={c} className="border-b border-etch">
                    <th scope="row" className="py-2 text-left font-sans text-[14px] font-normal">
                      <span className="mr-2 font-mono">{c}</span>
                      {CLASS_NAMES[c]}
                    </th>
                    <td className="py-2 text-right text-ink-2">{k?.measure}</td>
                    <td className="py-2 text-right">{k?.e_low}</td>
                    <td className="py-2 text-right">{k?.e_high}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="pri-h">
        <h2 id="pri-h" className="mb-1 font-label text-lg font-semibold">
          How segments are ranked
        </h2>
        <p className="max-w-prose text-ink-2">
          Detections within {cfg.priority.eps_meters} m form a segment. Its priority is{" "}
          <span className="tnum font-mono text-[13px] text-ink">
            {cfg.priority.weights.s_max}·S + {cfg.priority.weights.density}·D + {cfg.priority.weights.road_importance}·R +{" "}
            {cfg.priority.weights.confidence}·C
          </span>
          : worst severity, defect count (full marks at {cfg.priority.density_saturation_count}), road importance from OpenStreetMap, and mean model
          confidence. Weights live in <span className="font-mono text-[13px]">configs/priority.yaml</span>.
        </p>
      </section>

      <section aria-labelledby="lim-h">
        <h2 id="lim-h" className="mb-1 font-label text-lg font-semibold">
          Limits
        </h2>
        <ul className="max-w-prose list-disc space-y-1 pl-5 text-ink-2">
          <li>Trained on daytime photos from one phone mount on plains roads around Delhi. Night, rain and other cameras are untested.</li>
          <li>Severity is a 2D proxy and has not been validated against engineers' ratings in this app.</li>
          <li>Road class comes from OpenStreetMap and falls back to a default when the lookup is unavailable.</li>
        </ul>
      </section>

      <section aria-labelledby="cred-h">
        <h2 id="cred-h" className="mb-1 font-label text-lg font-semibold">
          Credits
        </h2>
        <p className="max-w-prose text-ink-2">
          Data: RDD2022 (Arya et al., 2022), India subset. Map data © OpenStreetMap contributors. Detector: Ultralytics YOLO11 (AGPL-3.0). Server{" "}
          model run <span className="font-mono text-[13px]">{cfg.model.run}</span>.
        </p>
      </section>
    </div>
  );
}
