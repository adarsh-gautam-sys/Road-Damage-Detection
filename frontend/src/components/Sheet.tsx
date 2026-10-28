import { X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";

/**
 * Bottom sheet on phones, side panel on wide screens. Escape and backdrop dismiss.
 * Used for segment detail so the map/queue stays in place behind it.
 */
export function Sheet({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[1200] lg:pointer-events-none">
      <button
        type="button"
        aria-label="Close details"
        className="absolute inset-0 bg-black/55 lg:hidden"
        onClick={onClose}
        tabIndex={-1}
      />
      <section
        role="dialog"
        aria-label={title}
        className="sheet-in pointer-events-auto absolute inset-x-0 bottom-0 flex max-h-[88dvh] flex-col rounded-t-xl border border-b-0 border-etch-strong bg-chassis shadow-sheet lg:inset-y-0 lg:right-0 lg:left-auto lg:max-h-none lg:w-[440px] lg:rounded-none lg:border-y-0 lg:border-r-0"
      >
        <header className="flex min-h-14 items-center justify-between gap-3 border-b border-etch px-4">
          <h2 className="font-label text-lg font-semibold tracking-wide">{title}</h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="grid h-11 w-11 place-items-center rounded-md border border-etch-strong hover:bg-raised"
          >
            <X size={20} aria-hidden="true" />
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4 pb-[max(1rem,env(safe-area-inset-bottom))]">
          {children}
        </div>
      </section>
    </div>
  );
}
