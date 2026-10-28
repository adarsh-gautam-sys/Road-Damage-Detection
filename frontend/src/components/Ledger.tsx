import type { PriorityTerms } from "../api/types";
import { formatLedger, PRIORITY_TERM_ROWS } from "../lib/severity";

/** One line of right-aligned tabular terms that sum to the priority shown beside it. */
export function LedgerLine({ weighted, total }: { weighted: Partial<PriorityTerms>; total: number }) {
  const { rows } = formatLedger(weighted);
  return (
    <p className="tnum flex flex-wrap items-baseline justify-end gap-x-3 font-mono text-[13px] text-ink-2" aria-label="Priority terms">
      {rows.map((r, i) => (
        <span key={r.key}>
          <span className="text-ink-3">{PRIORITY_TERM_ROWS[i]!.symbol}</span> {r.value.toFixed(1)}
        </span>
      ))}
      <span className="text-ink">= {total.toFixed(1)}</span>
    </p>
  );
}

/** Full ledger: weight × raw term → weighted contribution, then the sum. */
export function LedgerTable({
  terms,
  weighted,
  weights,
  total,
}: {
  terms: Partial<PriorityTerms>;
  weighted: Partial<PriorityTerms>;
  weights: PriorityTerms;
  total: number;
}) {
  return (
    <table className="tnum w-full font-mono text-[13px]">
      <caption className="sr-only">How the priority score is built from four weighted terms</caption>
      <thead>
        <tr className="border-b border-etch-strong text-ink-3">
          <th scope="col" className="py-1.5 text-left font-label text-[12px] font-semibold uppercase tracking-wider">Term</th>
          <th scope="col" className="py-1.5 text-right font-label text-[12px] font-semibold uppercase tracking-wider">Weight</th>
          <th scope="col" className="py-1.5 text-right font-label text-[12px] font-semibold uppercase tracking-wider">Value</th>
          <th scope="col" className="py-1.5 text-right font-label text-[12px] font-semibold uppercase tracking-wider">Adds</th>
        </tr>
      </thead>
      <tbody>
        {PRIORITY_TERM_ROWS.map((r) => (
          <tr key={r.key} className="border-b border-etch">
            <th scope="row" className="py-2 text-left font-sans text-[14px] font-normal">
              <span className="mr-2 text-ink-3">{r.symbol}</span>
              {r.label}
            </th>
            <td className="py-2 text-right text-ink-2">{weights[r.key].toFixed(2)}</td>
            <td className="py-2 text-right">{Number(terms[r.key] ?? 0).toFixed(1)}</td>
            <td className="py-2 text-right text-ink">{Number(weighted[r.key] ?? 0).toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
      <tfoot>
        <tr>
          <th scope="row" colSpan={3} className="py-2 text-right font-label text-[12px] font-semibold uppercase tracking-wider text-ink-2">
            Priority
          </th>
          <td className="py-2 text-right text-[15px] font-bold text-gold">{total.toFixed(1)}</td>
        </tr>
      </tfoot>
    </table>
  );
}
