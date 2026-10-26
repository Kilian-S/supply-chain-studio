import { useEffect, useMemo, useRef, useState } from 'react'

export interface Series {
  id: string
  label: string
  color: string
  values: (number | null)[]
}

interface Props {
  title: string
  x: number[] // day numbers
  series: Series[]
  height?: number
  format?: (v: number) => string
  xLabel?: string
}

const PAD = { left: 52, right: 12, top: 10, bottom: 24 }

// Round tick step (1, 2, 2.5 or 5 x 10^k) giving about four intervals up to `v`.
function niceStep(v: number) {
  if (v <= 0) return 1
  const raw = v / 4
  const p = 10 ** Math.floor(Math.log10(raw))
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p
  return 10 * p
}

export default function LineChart({ title, x, series, height = 170, format = (v) => v.toFixed(0), xLabel = 'Day' }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(390)
  const [hover, setHover] = useState<number | null>(null)
  const [showData, setShowData] = useState(false)

  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(240, e.contentRect.width)))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [showData])

  const { yMin, yMax, step, paths } = useMemo(() => {
    let lo = 0
    let hi = 0
    for (const s of series) for (const v of s.values) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
    const step = niceStep(Math.max(hi, -lo))
    const yMax = Math.max(step, Math.ceil(hi / step) * step)
    const yMin = lo < 0 ? Math.floor(lo / step) * step : 0
    const w = width - PAD.left - PAD.right
    const h = height - PAD.top - PAD.bottom
    const n = Math.max(x.length - 1, 1)
    const px = (i: number) => PAD.left + (i / n) * w
    const py = (v: number) => PAD.top + h - ((v - yMin) / (yMax - yMin || 1)) * h
    const paths = series.map((s) => {
      let d = ''
      let pen = false
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return }
        d += `${pen ? 'L' : 'M'}${px(i).toFixed(1)},${py(v).toFixed(1)}`
        pen = true
      })
      return d
    })
    return { yMin, yMax, step, paths }
  }, [series, x, width, height])

  const w = width - PAD.left - PAD.right
  const h = height - PAD.top - PAD.bottom
  const n = Math.max(x.length - 1, 1)
  const px = (i: number) => PAD.left + (i / n) * w
  const py = (v: number) => PAD.top + h - ((v - yMin) / (yMax - yMin || 1)) * h
  const yTicks = Array.from({ length: Math.round((yMax - yMin) / step) + 1 }, (_, i) => yMin + i * step)
  const xTicks = x.length ? [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * (x.length - 1))) : []

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const i = Math.round(((e.clientX - rect.left) / rect.width) * n)
    setHover(Math.max(0, Math.min(x.length - 1, i)))
  }

  return (
    <div className="section">
      <div className="panel-head" style={{ alignItems: 'center' }}>
        <h3>{title}</h3>
        <button className="btn small" onClick={() => setShowData((s) => !s)} aria-pressed={showData}>
          {showData ? 'Chart' : 'Data'}
        </button>
      </div>
      {series.length > 1 && (
        <div className="chart-legend">
          {series.map((s) => (
            <span key={s.id}><i className="swatch" style={{ background: s.color }} />{s.label}</span>
          ))}
        </div>
      )}
      {showData ? (
        <div className="table-wrap" style={{ maxHeight: 260, overflowY: 'auto' }}>
          <table className="grid">
            <thead>
              <tr><th>{xLabel}</th>{series.map((s) => <th key={s.id} className="num">{s.label}</th>)}</tr>
            </thead>
            <tbody>
              {x.map((d, i) => ({ d, i })).reverse().map(({ d, i }) => (
                <tr key={d}>
                  <td className="mono">{d}</td>
                  {series.map((s) => <td key={s.id} className="num">{s.values[i] == null ? '' : format(s.values[i] as number)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="chart" ref={box}>
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
            <g className="axis">
              {yTicks.map((t) => (
                <g key={t}>
                  <line className="gridline" x1={PAD.left} x2={width - PAD.right} y1={py(t)} y2={py(t)} />
                  <text x={PAD.left - 6} y={py(t) + 3} textAnchor="end">{format(t)}</text>
                </g>
              ))}
              {xTicks.map((i) => (
                <text key={i} x={px(i)} y={height - 6} textAnchor="middle">{x[i]}</text>
              ))}
            </g>
            {paths.map((d, k) => (
              <path key={series[k].id} d={d} fill="none" stroke={series[k].color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
            ))}
            {hover != null && (
              <g>
                <line x1={px(hover)} x2={px(hover)} y1={PAD.top} y2={PAD.top + h} stroke="#1e1b2e" strokeDasharray="3 3" />
                {series.map((s) => s.values[hover] != null && (
                  <circle key={s.id} cx={px(hover)} cy={py(s.values[hover] as number)} r={4.5} fill={s.color} stroke="#fff" strokeWidth={2} />
                ))}
              </g>
            )}
            <rect
              x={PAD.left} y={PAD.top} width={w} height={h} fill="transparent"
              onPointerMove={onMove} onPointerLeave={() => setHover(null)}
            />
          </svg>
          {hover != null && (
            <div className="tooltip" style={{ left: Math.min(px(hover) + 10, width - 150), top: 4 }}>
              <div className="row"><span className="muted">{xLabel}</span><b>{x[hover]}</b></div>
              {series.map((s) => (
                <div className="row" key={s.id}>
                  <span><i className="swatch" style={{ background: s.color, marginRight: 5 }} />{s.label}</span>
                  <b>{s.values[hover] == null ? 'n/a' : format(s.values[hover] as number)}</b>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
