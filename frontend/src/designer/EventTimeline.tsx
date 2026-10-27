import { useEffect, useRef, useState } from 'react'
import type { ScenarioEvent } from '../api'
import { kindMeta } from './eventKinds'

interface Props {
  events: ScenarioEvent[]
  startDay: number
  endDay: number
  selected: string | null
  onSelect: (id: string) => void
  onChange: (e: ScenarioEvent) => void
}

const ROW = 38
const LEFT = 12
const RIGHT = 12
const AXIS = 26

type Drag = { id: string; mode: 'move' | 'start' | 'end'; x0: number; start: number; end: number | null }

export default function EventTimeline({ events, startDay, endDay, selected, onSelect, onChange }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(700)
  const [drag, setDrag] = useState<Drag | null>(null)

  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, e.contentRect.width)))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [])

  const span = endDay - startDay + 1
  const pxPerDay = (width - LEFT - RIGHT) / span
  const x = (day: number) => LEFT + (day - startDay) * pxPerDay // left edge of a day
  const clamp = (d: number) => Math.max(startDay, Math.min(endDay, d))
  const height = AXIS + Math.max(events.length, 1) * ROW + 8
  const step = span > 600 ? 100 : span > 250 ? 50 : 10
  const ticks: number[] = []
  for (let d = Math.ceil(startDay / step) * step; d <= endDay; d += step) ticks.push(d)

  const begin = (e: ScenarioEvent, mode: Drag['mode']) => (ev: React.PointerEvent) => {
    ev.stopPropagation()
    ;(ev.target as Element).setPointerCapture(ev.pointerId)
    onSelect(e.id)
    setDrag({ id: e.id, mode, x0: ev.clientX, start: e.start_day, end: e.end_day })
  }
  const move = (ev: React.PointerEvent) => {
    if (!drag) return
    const e = events.find((x) => x.id === drag.id)
    if (!e) return
    const dd = Math.round((ev.clientX - drag.x0) / pxPerDay)
    if (drag.mode === 'move') {
      const len = drag.end == null ? null : drag.end - drag.start
      const start = clamp(Math.min(drag.start + dd, len == null ? endDay : endDay - len))
      onChange({ ...e, start_day: start, end_day: len == null ? null : start + len })
    } else if (drag.mode === 'start') {
      const limit = drag.end ?? endDay
      onChange({ ...e, start_day: clamp(Math.min(drag.start + dd, limit)) })
    } else {
      const base = drag.end ?? endDay
      onChange({ ...e, end_day: clamp(Math.max(base + dd, e.start_day)) })
    }
  }
  const keyMove = (e: ScenarioEvent) => (ev: React.KeyboardEvent) => {
    const d = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0
    if (!d) return
    ev.preventDefault()
    const by = ev.shiftKey ? 10 * d : d
    const start = clamp(e.start_day + by)
    const end = e.end_day == null ? null : clamp(e.end_day + (start - e.start_day))
    onChange({ ...e, start_day: start, end_day: end })
  }

  return (
    <div ref={box} className="gantt" onPointerMove={move} onPointerUp={() => setDrag(null)} onPointerCancel={() => setDrag(null)}>
      <svg width={width} height={height} role="group" aria-label="Event timeline">
        <defs>
          <pattern id="surprise" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="8" height="8" fill="transparent" />
            <line x1="0" y1="0" x2="0" y2="8" stroke="rgba(30,27,46,.18)" strokeWidth="2" />
          </pattern>
        </defs>
        {ticks.map((d) => (
          <g key={d}>
            <line x1={x(d)} x2={x(d)} y1={AXIS - 6} y2={height} stroke="#eeecf4" />
            <text x={x(d)} y={14} textAnchor="middle" className="gantt-tick">{d}</text>
          </g>
        ))}
        {events.length === 0 && (
          <text x={width / 2} y={AXIS + 24} textAnchor="middle" className="gantt-empty">No events: the scenario plays as recorded.</text>
        )}
        {events.map((e, i) => {
          const meta = kindMeta(e.kind)
          const y = AXIS + i * ROW + 4
          const x0 = x(e.start_day)
          const x1 = e.end_day == null ? x(endDay + 1) : x(e.end_day + 1)
          const w = Math.max(x1 - x0, 6)
          const ramp = Math.min(e.ramp_days * pxPerDay, w)
          const isSel = selected === e.id
          return (
            <g key={e.id}>
              <g role="button" tabIndex={0} aria-label={`${e.name}, day ${e.start_day}${e.end_day ? ` to ${e.end_day}` : ' onwards'}. Arrow keys move it.`}
                onPointerDown={begin(e, 'move')} onKeyDown={keyMove(e)} onFocus={() => onSelect(e.id)} style={{ cursor: drag ? 'grabbing' : 'grab' }}>
                <rect x={x0} y={y} width={w} height={ROW - 10} rx={6} fill={meta.color} fillOpacity={0.28}
                  stroke={isSel ? '#1e1b2e' : meta.color} strokeWidth={isSel ? 2.5 : 2} />
                {ramp > 0 && <path d={`M${x0 + 2},${y + ROW - 12} L${x0 + ramp},${y + 2} L${x0 + ramp},${y + ROW - 12} Z`} fill={meta.color} fillOpacity={0.35} />}
                {e.visibility === 'surprise' && <rect x={x0} y={y} width={w} height={ROW - 10} rx={6} fill="url(#surprise)" />}
                {e.end_day == null && (
                  <path d={`M${x1 - 10},${y + 5} L${x1 - 3},${y + (ROW - 10) / 2} L${x1 - 10},${y + ROW - 15}`} fill="none" stroke="#1e1b2e" strokeWidth={2} />
                )}
                <text x={Math.min(x0 + 8, width - 140)} y={y + (ROW - 10) / 2 + 4} className="gantt-label"
                  fill="#1e1b2e" dx={w > 90 ? 0 : w + 4}>
                  {e.name}{e.visibility === 'surprise' ? ' (surprise)' : ''}
                </text>
              </g>
              <rect x={x0 - 4} y={y} width={8} height={ROW - 10} fill="transparent" style={{ cursor: 'ew-resize' }}
                onPointerDown={begin(e, 'start')} />
              <rect x={x1 - 4} y={y} width={8} height={ROW - 10} fill="transparent" style={{ cursor: 'ew-resize' }}
                onPointerDown={begin(e, 'end')} />
            </g>
          )
        })}
      </svg>
    </div>
  )
}
