import { useEffect, useRef, useState, type ReactNode } from 'react'

export interface ScatterNote { x: number; y: number; text: string; color?: string; anchor?: 'start' | 'middle' | 'end' }

interface Props {
  x: number[]
  y: number[]
  color: (i: number) => string
  label: (i: number) => ReactNode
  xLabel: string
  yLabel: string
  logX?: boolean
  logY?: boolean
  vLines?: number[]
  hLines?: number[]
  title: string
  onPick?: (i: number) => void
  // Optional extras, all off by default.
  width?: number
  height?: number
  radius?: number
  domainX?: [number, number] // data units
  domainY?: [number, number]
  tick?: (v: number) => string // tick label for a value in data units
  background?: { x: number[]; y: number[]; color: string; radius?: number } // many points on a canvas layer underneath, no hover
  diagonal?: string // dashed line where y equals x (both axes log or both linear), with this label
  notes?: ScatterNote[] // direct labels at data positions
}

const PAD = { l: 52, r: 14, t: 10, b: 38 }
const tr = (log?: boolean) => (v: number) => (log ? Math.log10(Math.max(v, 1e-3)) : v)

function extent(lists: number[][]): [number, number] {
  let lo = Infinity, hi = -Infinity
  for (const a of lists) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v }
  return [lo, hi]
}

function scales(x0: number, x1: number, y0: number, y1: number, W: number, H: number) {
  return {
    px: (v: number) => PAD.l + ((v - x0) / (x1 - x0 || 1)) * (W - PAD.l - PAD.r),
    py: (v: number) => H - PAD.b - ((v - y0) / (y1 - y0 || 1)) * (H - PAD.t - PAD.b),
  }
}

export default function Scatter({ x, y, color, label, xLabel, yLabel, logX, logY, vLines = [], hLines = [], title, onPick,
  width = 520, height = 340, radius = 4, domainX, domainY, tick, background, diagonal, notes = [] }: Props) {
  const [hover, setHover] = useState<number | null>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const W = width, H = height
  const tx = tr(logX), ty = tr(logY)
  const xs = x.map(tx), ys = y.map(ty)
  const [x0, x1] = domainX ? [tx(domainX[0]), tx(domainX[1])] : extent([xs, vLines.map(tx), background ? background.x.map(tx) : []])
  const [y0, y1] = domainY ? [ty(domainY[0]), ty(domainY[1])] : extent([ys, hLines.map(ty), background ? background.y.map(ty) : []])
  const { px, py } = scales(x0, x1, y0, y1, W, H)
  const ticks = (a: number, b: number, log?: boolean) => {
    if (log) { const out = []; for (let e = Math.ceil(a); e <= Math.floor(b); e++) out.push(e); return out }
    return [0, 0.25, 0.5, 0.75, 1].map((f) => a + f * (b - a))
  }
  const fmt = (v: number, log?: boolean) => (tick ? tick(log ? 10 ** v : v) : log ? String(+(10 ** v).toPrecision(2)) : v.toFixed(2))

  // The background points, drawn once per size on a canvas under the marks.
  useEffect(() => {
    const el = wrap.current, cv = canvas.current
    if (!background || !el || !cv) return
    const draw = () => {
      const w = el.clientWidth
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      cv.width = Math.round(w * dpr)
      cv.height = Math.round((w * H / W) * dpr)
      const ctx = cv.getContext('2d')
      if (!ctx) return
      const k = (w / W) * dpr
      ctx.setTransform(k, 0, 0, k, 0, 0)
      ctx.clearRect(0, 0, W, H)
      const s = scales(x0, x1, y0, y1, W, H)
      const fx = tr(logX), fy = tr(logY)
      const r = background.radius ?? 1.5
      ctx.fillStyle = background.color
      for (let i = 0; i < background.x.length; i++) {
        ctx.beginPath()
        ctx.arc(s.px(fx(background.x[i])), s.py(fy(background.y[i])), r, 0, Math.PI * 2)
        ctx.fill()
      }
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(el)
    return () => ro.disconnect()
  }, [background, x0, x1, y0, y1, W, H, logX, logY])

  let diag: ReactNode = null
  if (diagonal != null) {
    const a = Math.max(x0, y0), b = Math.min(x1, y1)
    const [ax, ay, bx, by] = [px(a), py(a), px(b), py(b)]
    const angle = (Math.atan2(by - ay, bx - ax) * 180) / Math.PI
    diag = (
      <g>
        <line x1={ax} y1={ay} x2={bx} y2={by} stroke="#1e1b2e" strokeDasharray="5 4" />
        <text className="sc-note" textAnchor="end" fill="#1e1b2e" fontWeight={400}
          transform={`translate(${bx},${by}) rotate(${angle}) translate(-8,-7)`}>{diagonal}</text>
      </g>
    )
  }

  const marks = (
    <>
      {vLines.map((v) => <line key={`v${v}`} x1={px(tx(v))} x2={px(tx(v))} y1={PAD.t} y2={H - PAD.b} stroke="#1e1b2e" strokeDasharray="4 3" />)}
      {hLines.map((v) => <line key={`h${v}`} x1={PAD.l} x2={W - PAD.r} y1={py(ty(v))} y2={py(ty(v))} stroke="#1e1b2e" strokeDasharray="4 3" />)}
      {diag}
      {xs.map((v, i) => (
        <circle key={i} cx={px(v)} cy={py(ys[i])} r={hover === i ? radius + 2 : radius} fill={color(i)} stroke="#fff" strokeWidth={1.2}
          onPointerEnter={() => setHover(i)} onPointerLeave={() => setHover(null)} onClick={() => onPick?.(i)} style={{ cursor: onPick ? 'pointer' : 'default' }} />
      ))}
      {notes.map((n) => (
        <text key={n.text} className="sc-note" x={px(tx(n.x))} y={py(ty(n.y))} textAnchor={n.anchor ?? 'start'} fill={n.color ?? '#1e1b2e'}>{n.text}</text>
      ))}
    </>
  )

  const right = hover != null && px(xs[hover]) > W * 0.65
  return (
    <div className="chart" ref={wrap}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {ticks(x0, x1, logX).map((t) => (
          <g key={`x${t}`}><line className="gridline" x1={px(t)} x2={px(t)} y1={PAD.t} y2={H - PAD.b} />
            <text className="axis-text" x={px(t)} y={H - PAD.b + 14} textAnchor="middle">{fmt(t, logX)}</text></g>
        ))}
        {ticks(y0, y1, logY).map((t) => (
          <g key={`y${t}`}><line className="gridline" x1={PAD.l} x2={W - PAD.r} y1={py(t)} y2={py(t)} />
            <text className="axis-text" x={PAD.l - 6} y={py(t) + 3} textAnchor="end">{fmt(t, logY)}</text></g>
        ))}
        <text className="axis-title" x={(W + PAD.l) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text className="axis-title" transform={`translate(12,${(H - PAD.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {!background && marks}
      </svg>
      {background && <canvas ref={canvas} className="sc-scatter-layer" aria-hidden="true" />}
      {background && <svg className="sc-scatter-layer" viewBox={`0 0 ${W} ${H}`} aria-hidden="true">{marks}</svg>}
      {hover != null && (
        <div className="tooltip" style={background
          ? { left: `${(px(xs[hover]) / W) * 100}%`, top: `${(py(ys[hover]) / H) * 100}%`, transform: right ? 'translate(calc(-100% - 12px), -50%)' : 'translate(12px, -50%)' }
          : { left: px(xs[hover]) / W * 100 + '%', top: 8 }}>{label(hover)}</div>
      )}
    </div>
  )
}
