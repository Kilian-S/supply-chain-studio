import { useState } from 'react'
import { diverging, inkOn, sequential } from './colors'

interface Props {
  rows: string[]
  cols: string[]
  values: number[][]
  mode: 'sequential' | 'diverging'
  format?: (v: number) => string
  max?: number // sequential: value mapped to the darkest step; diverging: |value| mapped to full colour
  cell?: number
  rowWidth?: number
  colHeight?: number
  title: string
  showValues?: boolean
}

export default function Heatmap({ rows, cols, values, mode, format = (v) => v.toFixed(2), max, cell = 34, rowWidth = 150, colHeight = 90, title, showValues = true }: Props) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const top = values.flat().reduce((a, b) => Math.max(a, Math.abs(b)), 0)
  const scale = max ?? (top || 1)
  const width = rowWidth + cols.length * cell
  const height = colHeight + rows.length * cell
  const fill = (v: number) => (mode === 'sequential' ? sequential(v / scale) : diverging(v / scale))
  return (
    <div className="chart heatmap" style={{ maxWidth: width + 8 }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {cols.map((c, j) => (
          c.length <= 3
            ? <text key={c} className="hm-col" x={rowWidth + j * cell + cell / 2} y={colHeight - 8} textAnchor="middle">{c}</text>
            : <text key={c} className="hm-col" transform={`translate(${rowWidth + j * cell + cell / 2},${colHeight - 6}) rotate(-50)`}>{c}</text>
        ))}
        {rows.map((r, i) => (
          <g key={r + i}>
            <text className="hm-row" x={rowWidth - 8} y={colHeight + i * cell + cell / 2 + 4} textAnchor="end">{r}</text>
            {cols.map((c, j) => {
              const v = values[i][j]
              const strength = mode === 'sequential' ? v / scale : v / scale
              return (
                <g key={c} onPointerEnter={() => setHover([i, j])} onPointerLeave={() => setHover(null)}>
                  <rect x={rowWidth + j * cell + 1} y={colHeight + i * cell + 1} width={cell - 2} height={cell - 2} rx={3} fill={fill(v)}
                    stroke={hover && hover[0] === i && hover[1] === j ? '#1e1b2e' : 'none'} strokeWidth={2} />
                  {showValues && cell >= 28 && (
                    <text x={rowWidth + j * cell + cell / 2} y={colHeight + i * cell + cell / 2 + 4} textAnchor="middle" className="hm-val"
                      fill={inkOn(strength)}>{format(v)}</text>
                  )}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
      {hover && (
        <div className="tooltip" style={{ position: 'static', display: 'inline-block', marginTop: 4 }}>
          <b>{rows[hover[0]]}</b> × <b>{cols[hover[1]]}</b>: {format(values[hover[0]][hover[1]])}
        </div>
      )}
    </div>
  )
}
