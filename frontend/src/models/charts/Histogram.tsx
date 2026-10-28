import type { Hist } from '../types'

export default function Histogram({ hist, title, color = '#2a78d6', format = (v: number) => v.toPrecision(3), height = 120 }: {
  hist: Hist; title: string; color?: string; format?: (v: number) => string; height?: number
}) {
  const W = 320
  const pad = { l: 8, r: 8, t: 6, b: 20 }
  const max = Math.max(...hist.counts, 1)
  const n = hist.counts.length
  const bw = (W - pad.l - pad.r) / n
  return (
    <figure className="chart mini-chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={title}>
        <line x1={pad.l} x2={W - pad.r} y1={height - pad.b} y2={height - pad.b} stroke="#d6d3e2" />
        {hist.counts.map((c, i) => {
          const h = ((height - pad.t - pad.b) * c) / max
          return (
            <rect key={i} x={pad.l + i * bw + 1} y={height - pad.b - h} width={Math.max(bw - 2, 1)} height={h} rx={2} fill={color}>
              <title>{`${format(hist.edges[i])} to ${format(hist.edges[i + 1])}: ${c} SKUs`}</title>
            </rect>
          )
        })}
        <text className="axis-text" x={pad.l} y={height - 5}>{format(hist.edges[0])}</text>
        <text className="axis-text" x={W - pad.r} y={height - 5} textAnchor="end">{format(hist.edges[hist.edges.length - 1])}</text>
      </svg>
      <figcaption>{title}</figcaption>
    </figure>
  )
}
