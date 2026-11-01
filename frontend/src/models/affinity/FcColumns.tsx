import { useMemo, useRef, useState, type ReactNode } from 'react'

/* Items placed into bins: one column per FC, one square per SKU. Squares keep a fixed size and each
 * column a fixed grid width at every FC count. Squares of one run (a community) are joined with a
 * 1 px white hairline between them; runs are separated by a 2 px white gap. Squares can be drawn pale
 * (a community split across FCs). Hover or keyboard focus gives a read-out; selecting a square
 * outlines every square of its run in all columns. Arrow keys move between squares, Enter selects. */

interface Props {
  fcs: number
  assign: number[] // FC index (0-based) per SKU
  order: number[] // every SKU in display order; each column keeps this order
  run: (i: number) => string // run key per SKU
  color: (i: number) => string
  heading: (f: number) => string // column heading for FC index f
  describe: (i: number) => ReactNode // read-out for one SKU
  name: (i: number) => string // accessible name for one square
  selected: number | null
  onSelect: (i: number | null) => void
  title: string
  pale?: (i: number) => boolean // drawn at reduced opacity
  minRows?: number // rows to reserve, so the layout does not jump between settings
}

const S = 14 // square
const G = 2 // gap between runs
const PER = 10 // squares per row
const HAIR = 1 // white line between squares of one run
const HEAD = 25 // column heading and its gap

export default function FcColumns({ fcs, assign, order, run, color, heading, describe, name, selected, onSelect, title, pale, minRows = 0 }: Props) {
  const cols = useMemo(() => Array.from({ length: fcs }, (_, f) => order.filter((i) => assign[i] === f)), [fcs, assign, order])
  const flat = useMemo(() => cols.flat(), [cols])
  const [active, setActive] = useState<number | null>(null)
  const [focusSku, setFocusSku] = useState<number | null>(null)
  const rects = useRef(new Map<number, SVGRectElement>())
  const tabStop = focusSku != null && flat.includes(focusSku) ? focusSku : flat[0]
  const selRun = selected != null ? run(selected) : null

  const move = (i: number, by: 'next' | 'prev' | 'down' | 'up' | 'home' | 'end') => {
    const f = assign[i]
    const col = cols[f]
    const p = col.indexOf(i)
    const k = flat.indexOf(i)
    let j: number | undefined
    if (by === 'next') j = flat[Math.min(flat.length - 1, k + 1)]
    else if (by === 'prev') j = flat[Math.max(0, k - 1)]
    else if (by === 'down') j = col[Math.min(col.length - 1, p + PER)]
    else if (by === 'up') j = col[Math.max(0, p - PER)]
    else if (by === 'home') j = flat[0]
    else j = flat[flat.length - 1]
    if (j == null) return
    setFocusSku(j)
    rects.current.get(j)?.focus()
  }
  const onKey = (e: React.KeyboardEvent, i: number) => {
    const keys: Record<string, 'next' | 'prev' | 'down' | 'up' | 'home' | 'end'> = {
      ArrowRight: 'next', ArrowLeft: 'prev', ArrowDown: 'down', ArrowUp: 'up', Home: 'home', End: 'end',
    }
    if (keys[e.key]) { e.preventDefault(); move(i, keys[e.key]) }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(selected === i ? null : i) }
    else if (e.key === 'Escape') onSelect(null)
  }

  const shown = active ?? selected
  return (
    <div className="sc-fc" role="group" aria-label={title}>
      <div className="sc-fc-cols" style={{ minHeight: HEAD + minRows * (S + G) }}>
        {cols.map((col, f) => {
          const rows = Math.max(1, Math.ceil(col.length / PER))
          const w = PER * (S + G) - G, h = rows * (S + G) - G
          return (
            <div key={f} className="sc-fc-col">
              <h3>{heading(f)}</h3>
              <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
                {col.map((i, p) => {
                  const x = (p % PER) * (S + G), y = Math.floor(p / PER) * (S + G)
                  const right = p % PER !== PER - 1 && p + 1 < col.length && run(col[p + 1]) === run(i)
                  const below = p + PER < col.length && run(col[p + PER]) === run(i)
                  return (
                    <rect key={i} ref={(el) => { if (el) rects.current.set(i, el); else rects.current.delete(i) }}
                      x={x} y={y} width={S + (right ? G - HAIR : 0)} height={S + (below ? G - HAIR : 0)} fill={color(i)}
                      fillOpacity={pale?.(i) ? 0.35 : 1}
                      role="button" tabIndex={i === tabStop ? 0 : -1} aria-label={name(i)} aria-pressed={selected === i}
                      onPointerEnter={() => setActive(i)} onPointerLeave={() => setActive(null)}
                      onFocus={() => { setFocusSku(i); setActive(i) }} onBlur={() => setActive(null)}
                      onClick={() => onSelect(selected === i ? null : i)} onKeyDown={(e) => onKey(e, i)} />
                  )
                })}
                {selRun != null && col.map((i, p) => run(i) === selRun && (
                  <rect key={`o${i}`} className="sc-fc-outline" x={(p % PER) * (S + G) + 0.75} y={Math.floor(p / PER) * (S + G) + 0.75} width={S - 1.5} height={S - 1.5} />
                ))}
                {active != null && assign[active] === f && (() => {
                  const p = col.indexOf(active)
                  return <rect className="sc-fc-active" x={(p % PER) * (S + G) - 1.5} y={Math.floor(p / PER) * (S + G) - 1.5} width={S + 3} height={S + 3} />
                })()}
              </svg>
            </div>
          )
        })}
      </div>
      <div className="sc-readout" aria-live="polite">{shown != null ? describe(shown) : null}</div>
    </div>
  )
}
