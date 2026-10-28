import { AlertTriangle, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon: ReactNode;
  title: string;
  body: string;
  action?: ReactNode;
}) {
  return (
    <div className="mx-auto flex max-w-sm flex-col items-center gap-3 px-6 py-14 text-center">
      <div className="grid h-14 w-14 place-items-center rounded-full border border-etch-strong text-ink-2">{icon}</div>
      <h2 className="font-label text-xl font-semibold tracking-wide">{title}</h2>
      <p className="text-ink-2">{body}</p>
      {action}
    </div>
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="plate flex items-start gap-3 border-sev-high/60 p-3">
      <AlertTriangle className="mt-0.5 shrink-0 text-sev-high-text" size={20} aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="text-ink">{message}</p>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="flex min-h-11 items-center gap-1.5 rounded-md border border-etch-strong px-3 font-label text-sm font-semibold uppercase tracking-wider hover:bg-raised"
        >
          <RefreshCw size={16} aria-hidden="true" /> Retry
        </button>
      )}
    </div>
  );
}

/** Gold plate: the only filled accent in the product. Reserved for the primary action. */
export function GoldButton({
  children,
  className = "",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
  return (
    <button
      {...rest}
      className={`min-h-12 rounded-md bg-gold px-5 font-label text-base font-bold uppercase tracking-wider text-gold-ink shadow-plate transition-colors hover:bg-gold-press active:translate-y-px disabled:bg-etch-strong disabled:text-ink-3 disabled:shadow-none ${className}`}
    >
      {children}
    </button>
  );
}

export function PlainButton({
  children,
  className = "",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: ReactNode }) {
  return (
    <button
      {...rest}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-md border border-etch-strong bg-panel px-4 font-label text-sm font-semibold uppercase tracking-wider text-ink hover:bg-raised disabled:opacity-45 ${className}`}
    >
      {children}
    </button>
  );
}
