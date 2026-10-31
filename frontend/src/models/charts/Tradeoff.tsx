// Every setting as a dot (cost across, match with the ground truth up), with a few marked points.

export interface Marked {
  x: number
  y: number
  label: string
  kind: 'ours' | 'best' | 'baseline'
  anchor?: 'start' | 'end'
  dy?: number
}

interface Props {
  x: number[]
  y: number[]
  marked: Marked[]
  xLabel: string
  yLabel: string
  xTicks: number[]
  yTicks: number[]
  formatX: (v: number) => string
  formatY: (v: number) => string
  title: string
}

export default function Tradeoff({ x, y, marked, xLabel, yLabel, xTicks, yTicks, formatX, formatY, title }: Props) {
  const W = 1040, H = 460, pad = { l: 64, r: 24, t: 16, b: 52 }
  const [x0, x1] = [xTicks[0], xTicks[xTicks.length - 1]]
  const [y0, y1] = [yTicks[0], yTicks[yTicks.length - 1]]
  const px = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r)
  const py = (v: number) => H - pad.b - ((v - y0) / (y1 - y0)) * (H - pad.t - pad.b)
  return (
    <div className="chart sc-tradeoff">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line className="gridline" x1={px(t)} x2={px(t)} y1={pad.t} y2={H - pad.b} />
            <text className="axis-text" x={px(t)} y={H - pad.b + 18} textAnchor="middle">{formatX(t)}</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="gridline" x1={pad.l} x2={W - pad.r} y1={py(t)} y2={py(t)} />
            <text className="axis-text" x={pad.l - 10} y={py(t) + 4} textAnchor="end">{formatY(t)}</text>
          </g>
        ))}
        <text className="axis-title" x={(pad.l + W - pad.r) / 2} y={H - 10} textAnchor="middle">{xLabel}</text>
        <text className="axis-title" transform={`translate(16,${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {x.map((v, i) => <circle key={i} cx={px(v)} cy={py(y[i])} r={2.4} className="sc-tradeoff-dot" />)}
        {marked.map((m) => (
          <g key={m.label} className={`sc-tradeoff-mark ${m.kind}`}>
            {m.kind === 'baseline'
              ? <rect x={px(m.x) - 5} y={py(m.y) - 5} width={10} height={10} />
              : <circle cx={px(m.x)} cy={py(m.y)} r={7} />}
            <text x={px(m.x) + (m.anchor === 'end' ? -12 : 12)} y={py(m.y) + 4 + (m.dy ?? 0)} textAnchor={m.anchor ?? 'start'}>{m.label}</text>
          </g>
        ))}
      </svg>
    </div>
  )
}
