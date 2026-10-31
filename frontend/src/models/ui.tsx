import type { ReactNode } from 'react'

export function Screen({ id, label, headline, caption, drawer, children }: {
  id: string; label?: string; headline: string; caption?: string; drawer?: ReactNode; children?: ReactNode
}) {
  return (
    <section id={id} className="sc-screen" aria-labelledby={`${id}-h`}>
      {label && <div className="sc-label">{label}</div>}
      <h2 id={`${id}-h`} className="sc-headline">{headline}</h2>
      {children}
      {caption && <p className="sc-caption">{caption}</p>}
      {drawer}
    </section>
  )
}

export function Numbers({ items }: { items: { value: string; label: string; note?: string }[] }) {
  return (
    <div className="sc-numbers">
      {items.map((n) => (
        <div className="sc-number" key={n.label}><b>{n.value}</b><span>{n.label}</span>{n.note && <small>{n.note}</small>}</div>
      ))}
    </div>
  )
}

export function Bars({ rows, format, max }: { rows: { label: string; value: number; note?: string; ours?: boolean }[]; format: (v: number) => string; max?: number }) {
  const top = max ?? Math.max(...rows.map((r) => r.value))
  return (
    <div className="sc-bars">
      {rows.map((r) => (
        <div key={r.label} className={`sc-bar${r.ours ? ' ours' : ''}`}>
          <span>{r.label}{r.note && <small>{r.note}</small>}</span>
          <span className="sc-bar-track"><span className="sc-bar-fill" style={{ width: `${(r.value / top) * 100}%` }} /></span>
          <span className="sc-bar-value">{format(r.value)}</span>
        </div>
      ))}
    </div>
  )
}

export function Drawer({ children, title = 'How it works' }: { children: ReactNode; title?: string }) {
  return (
    <details className="sc-drawer">
      <summary aria-label={title} title={title}><span className="sc-drawer-icon" aria-hidden="true" /></summary>
      <div className="sc-drawer-body">{children}</div>
    </details>
  )
}

export function Sparkline({ values, color = '#4338ca', label }: { values: number[]; color?: string; label: string }) {
  const W = 300, H = 46
  const max = Math.max(...values, 1)
  const pts = values.map((v, i) => `${((i / Math.max(values.length - 1, 1)) * W).toFixed(1)},${(H - 2 - (v / max) * (H - 4)).toFixed(1)}`)
  return (
    <svg className="sc-sparkline" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label={label}>
      <path d={`M0,${H} L${pts.join(' L')} L${W},${H} Z`} fill={color} opacity={0.12} />
      <path d={`M${pts.join(' L')}`} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
