// A monthly series with an optional forecast fan: the actual line (grey before the forecast origin),
// the P10 to P90 band and the P50 line. Used full width and as small multiples (compact).

export interface FanBand {
  from: number // month of the first forecast value (1-based)
  lo: number[]
  mid: number[]
  hi: number[]
}

interface Props {
  title: string
  actual: (number | null)[] // months 1..n
  band?: FanBand
  reference?: FanBand // a second band in grey (the ground truth), drawn under ours
  height?: number
  width?: number // viewBox width, so text keeps its size in a narrow column
  start?: number // first month shown (1-based), to zoom in on the forecast
  compact?: boolean
  yLabel?: string
  format?: (v: number) => string
}

const W = 1040
const ACCENT = '#4338ca'
const INK = '#1e1b2e'
const GREY = '#a8a4b5'

export default function Fan({ title, actual, band, reference, height = 320, width, start = 1, compact = false, yLabel, format = (v) => String(Math.round(v)) }: Props) {
  const w = width ?? (compact ? 340 : W)
  const pad = compact ? { l: 44, r: 8, t: 8, b: 22 } : { l: 64, r: 16, t: 12, b: 40 }
  const n = Math.max(actual.length, band ? band.from + band.mid.length - 1 : 0)
  const vals = [...actual.slice(start - 1).filter((v): v is number => v != null), ...(band ? band.hi : []), ...(reference ? reference.hi : [])]
  const top = niceMax(Math.max(...vals, 1))
  const px = (m: number) => pad.l + ((m - start) / Math.max(n - start, 1)) * (w - pad.l - pad.r)
  const py = (v: number) => height - pad.b - (v / top) * (height - pad.t - pad.b)
  const line = (ys: (number | null)[], from: number) =>
    ys.map((v, i) => (v == null || from + i < start ? null : `${px(from + i).toFixed(1)},${py(v).toFixed(1)}`)).filter(Boolean).join(' L')
  const origin = band ? band.from - 1 : n
  const years = Math.ceil(n / 12)
  const first = Math.floor((start - 1) / 12)
  const ticks = [0, 0.5, 1].map((f) => f * top)
  return (
    <div className="chart sc-fan">
      <svg viewBox={`0 0 ${w} ${height}`} role="img" aria-label={title}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={pad.l} x2={w - pad.r} y1={py(t)} y2={py(t)} />
            <text className="axis-text" x={pad.l - 6} y={py(t) + 4} textAnchor="end">{format(t)}</text>
          </g>
        ))}
        {Array.from({ length: years }, (_, y) => y).filter((y) => y >= first).map((y) => (
          <g key={y}>
            {y > first && <line className="gridline" x1={px(12 * y + 0.5)} x2={px(12 * y + 0.5)} y1={pad.t} y2={height - pad.b} />}
            {(!compact || y % 2 === 0 || years <= 5) && (
              <text className="axis-text" x={px(12 * y + 6.5)} y={height - pad.b + (compact ? 14 : 18)} textAnchor="middle">{compact ? y + 1 : `Year ${y + 1}`}</text>
            )}
          </g>
        ))}
        {!compact && yLabel && <text className="axis-title" transform={`translate(16,${(pad.t + height - pad.b) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>}
        {band && (
          <>
            <path d={`M${band.hi.map((v, i) => `${px(band.from + i).toFixed(1)},${py(v).toFixed(1)}`).join(' L')} L${band.lo.map((v, i) => `${px(band.from + i).toFixed(1)},${py(v).toFixed(1)}`).reverse().join(' L')} Z`}
              fill={ACCENT} opacity={0.14} />
            <line x1={px(origin + 0.5)} x2={px(origin + 0.5)} y1={pad.t} y2={height - pad.b} stroke={INK} strokeDasharray="3 3" strokeWidth={1} />
          </>
        )}
        {reference && (
          <path d={`M${reference.hi.map((v, i) => `${px(reference.from + i).toFixed(1)},${py(v).toFixed(1)}`).join(' L')} L${reference.lo.map((v, i) => `${px(reference.from + i).toFixed(1)},${py(v).toFixed(1)}`).reverse().join(' L')} Z`}
            fill={INK} opacity={0.38} />
        )}
        <path d={`M${line(actual.slice(0, origin), 1)}`} fill="none" stroke={band ? GREY : INK} strokeWidth={1.4} />
        {band && actual.length > origin && <path d={`M${line(actual.slice(origin - 1), origin)}`} fill="none" stroke={INK} strokeWidth={1.4} />}
        {band && <path d={`M${line(band.mid, band.from)}`} fill="none" stroke={ACCENT} strokeWidth={1.8} />}
      </svg>
    </div>
  )
}

function niceMax(v: number) {
  const e = 10 ** Math.floor(Math.log10(v))
  for (const k of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (k * e >= v) return k * e
  return 10 * e
}
