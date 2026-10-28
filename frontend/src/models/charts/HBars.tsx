export interface Bar { label: string; value: number; color?: string; note?: string; highlight?: boolean }

export default function HBars({ bars, format, title, max }: { bars: Bar[]; format: (v: number) => string; title: string; max?: number }) {
  const top = max ?? Math.max(...bars.map((b) => b.value), 1e-9)
  return (
    <div className="hbars" role="img" aria-label={title}>
      {bars.map((b) => (
        <div key={b.label} className={`hbar${b.highlight ? ' hl' : ''}`}>
          <span className="hbar-label">{b.label}</span>
          <span className="hbar-track">
            <span className="hbar-fill" style={{ width: `${Math.max(0, (b.value / top) * 100)}%`, background: b.color ?? '#2a78d6' }} />
          </span>
          <span className="hbar-value">{format(b.value)}{b.note ? <span className="muted"> {b.note}</span> : null}</span>
        </div>
      ))}
    </div>
  )
}
