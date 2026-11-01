import { useEffect, useRef, useState, type ReactNode } from 'react'
import { sequential } from './colors'

/* A symmetric item x item matrix on one canvas (a few hundred items, so tens of thousands of cells),
 * coloured in steps of the blue sequential scale, with a stepped legend drawn in the same canvas.
 * Cells below the first step and the diagonal stay white. Hover gives a read-out under the matrix. */

interface Props {
  order: number[] // items in row and column order
  value: (a: number, b: number) => number // value of the cell for items a and b
  steps: number[] // ascending step boundaries; values from steps[k] up to steps[k + 1] get colour k
  legendTitle: string
  formatStep: (v: number) => string
  readout: (a: number, b: number) => ReactNode
  title: string
  cell?: number // CSS pixels per cell
}

const LEGEND = 150
const SWATCH = 24

const bytes = (css: string) => (css.match(/\d+/g) ?? ['255', '255', '255']).slice(0, 3).map(Number)

export default function PairMatrix({ order, value, steps, legendTitle, formatStep, readout, title, cell = 2 }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const n = order.length
  const side = n * cell
  const cssW = side + LEGEND, cssH = side

  useEffect(() => {
    const cv = ref.current
    if (!cv) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    cv.width = Math.round(cssW * dpr)
    cv.height = Math.round(cssH * dpr)
    const ctx = cv.getContext('2d')
    if (!ctx) return
    const colours = steps.slice(0, -1).map((_, k, a) => bytes(sequential(a.length > 1 ? k / (a.length - 1) : 1)))
    const stepOf = (v: number) => {
      if (!(v >= steps[0])) return -1
      let k = 0
      while (k < colours.length - 1 && v >= steps[k + 1]) k++
      return k
    }
    // One pixel per cell, then scaled up without smoothing.
    const off = document.createElement('canvas')
    off.width = n
    off.height = n
    const octx = off.getContext('2d')
    if (!octx) return
    const img = octx.createImageData(n, n)
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const k = r === c ? -1 : stepOf(value(order[r], order[c]))
        const [R, G, B] = k < 0 ? [255, 255, 255] : colours[k]
        const p = (r * n + c) * 4
        img.data[p] = R; img.data[p + 1] = G; img.data[p + 2] = B; img.data[p + 3] = 255
      }
    }
    octx.putImageData(img, 0, 0)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssW, cssH)
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(off, 0, 0, side, side)
    ctx.strokeStyle = 'rgba(30, 27, 46, 0.14)'
    ctx.lineWidth = 1
    ctx.strokeRect(0.5, 0.5, side - 1, side - 1)

    // Stepped legend: highest step on top, boundary values beside the swatch edges.
    const font = getComputedStyle(cv).fontFamily || 'system-ui'
    const lx = side + 24
    ctx.fillStyle = '#1e1b2e'
    ctx.font = `600 12px ${font}`
    ctx.textBaseline = 'alphabetic'
    ctx.fillText(legendTitle, lx, 12)
    const top = 26
    ctx.font = `12px ${font}`
    ctx.textBaseline = 'middle'
    const m = colours.length
    for (let k = 0; k < m; k++) {
      const y = top + (m - 1 - k) * SWATCH
      const [R, G, B] = colours[k]
      ctx.fillStyle = `rgb(${R},${G},${B})`
      ctx.fillRect(lx, y, 16, SWATCH)
    }
    ctx.fillStyle = '#5e5a72'
    for (let k = 0; k <= m; k++) ctx.fillText(formatStep(steps[k]), lx + 24, top + (m - k) * SWATCH)
  }, [order, value, steps, legendTitle, formatStep, n, side, cssW, cssH])

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const k = r.width / cssW
    const x = (e.clientX - r.left) / k, y = (e.clientY - r.top) / k
    if (x < 0 || y < 0 || x >= side || y >= side) { setHover(null); return }
    const row = Math.floor(y / cell), col = Math.floor(x / cell)
    setHover(row === col ? null : [row, col])
  }

  return (
    <div className="sc-matrix-wrap">
      <div className="sc-matrix" style={{ width: cssW }}>
        <canvas ref={ref} role="img" aria-label={title} style={{ width: cssW }} onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        {hover && (
          <div className="sc-matrix-hover" style={{
            left: `${((hover[1] * cell - 2) / cssW) * 100}%`, top: `${((hover[0] * cell - 2) / cssH) * 100}%`,
            width: cell + 4, height: cell + 4,
          }} />
        )}
      </div>
      <div className="sc-readout" aria-live="polite">{hover ? readout(order[hover[0]], order[hover[1]]) : null}</div>
    </div>
  )
}
