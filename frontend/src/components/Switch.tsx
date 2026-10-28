export function Switch({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="group flex min-h-11 min-w-11 items-center gap-2.5 rounded-md px-1 disabled:opacity-45"
    >
      <span
        className={`relative h-7 w-12 shrink-0 rounded-full border transition-colors duration-200 ${
          checked ? "border-gold bg-gold" : "border-etch-strong bg-[#0c0e10]"
        }`}
      >
        <span
          className={`absolute top-0.5 h-5.5 w-5.5 rounded-full transition-all duration-200 ease-[var(--ease-out)] ${
            checked ? "left-[22px] bg-gold-ink" : "left-0.5 bg-ink-2"
          }`}
          style={{ width: 22, height: 22 }}
        />
      </span>
      <span className="label text-ink">{label}</span>
    </button>
  );
}
