import { useState } from 'react'

interface Props {
  x: number[]
  y: number[]
  color: (i: number) => string
  label: (i: number) => string
  xLabel: string
  yLabel: string
  logX?: boolean
  logY?: boolean
  vLines?: number[]
  hLines?: number[]
  title: string
  onPick?: (i: number) => void
}

export default function Scatter({ x, y, color, label, xLabel, yLabel, logX, logY, vLines = [], hLines = [], title, onPick }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const W = 520, H = 340, pad = { l: 52, r: 14, t: 10, b: 38 }
  const tx = (v: number) => (logX ? Math.log10(Math.max(v, 1e-3)) : v)
  const ty = (v: number) => (logY ? Math.log10(Math.max(v, 1e-3)) : v)
  const xs = x.map(tx), ys = y.map(ty)
  const [x0, x1] = [Math.min(...xs, ...vLines.map(tx)), Math.max(...xs, ...vLines.map(tx))]
  const [y0, y1] = [Math.min(...ys, ...hLines.map(ty)), Math.max(...ys, ...hLines.map(ty))]
  const px = (v: number) => pad.l + ((v - x0) / (x1 - x0 || 1)) * (W - pad.l - pad.r)
  const py = (v: number) => H - pad.b - ((v - y0) / (y1 - y0 || 1)) * (H - pad.t - pad.b)
  const ticks = (a: number, b: number, log?: boolean) => {
    if (log) { const out = []; for (let e = Math.ceil(a); e <= Math.floor(b); e++) out.push(e); return out }
    return [0, 0.25, 0.5, 0.75, 1].map((f) => a + f * (b - a))
  }
  const fmt = (v: number, log?: boolean) => (log ? String(+(10 ** v).toPrecision(2)) : v.toFixed(2))
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {ticks(x0, x1, logX).map((t) => (
          <g key={`x${t}`}><line className="gridline" x1={px(t)} x2={px(t)} y1={pad.t} y2={H - pad.b} />
            <text className="axis-text" x={px(t)} y={H - pad.b + 14} textAnchor="middle">{fmt(t, logX)}</text></g>
        ))}
        {ticks(y0, y1, logY).map((t) => (
          <g key={`y${t}`}><line className="gridline" x1={pad.l} x2={W - pad.r} y1={py(t)} y2={py(t)} />
            <text className="axis-text" x={pad.l - 6} y={py(t) + 3} textAnchor="end">{fmt(t, logY)}</text></g>
        ))}
        {vLines.map((v) => <line key={`v${v}`} x1={px(tx(v))} x2={px(tx(v))} y1={pad.t} y2={H - pad.b} stroke="#1e1b2e" strokeDasharray="4 3" />)}
        {hLines.map((v) => <line key={`h${v}`} x1={pad.l} x2={W - pad.r} y1={py(ty(v))} y2={py(ty(v))} stroke="#1e1b2e" strokeDasharray="4 3" />)}
        {xs.map((v, i) => (
          <circle key={i} cx={px(v)} cy={py(ys[i])} r={hover === i ? 6 : 4} fill={color(i)} stroke="#fff" strokeWidth={1.2}
            onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} onClick={() => onPick?.(i)} style={{ cursor: onPick ? 'pointer' : 'default' }} />
        ))}
        <text className="axis-title" x={(W + pad.l) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text className="axis-title" transform={`translate(12,${(H - pad.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
      </svg>
      {hover != null && <div className="tooltip" style={{ left: px(xs[hover]) / W * 100 + '%', top: 8 }}>{label(hover)}</div>}
    </div>
  )
}
