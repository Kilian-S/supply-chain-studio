// Colour scales for model pages (validated reference palette; see the dataviz method).

export const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
// Beyond eight groups we add two muted extras; every view also shows a legend and hover labels.
export const EXTENDED = [...CATEGORICAL, '#6b7280', '#8c6d3f']
export const cat = (i: number) => EXTENDED[i % EXTENDED.length]

const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

function hex(c: string) { return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) }
function mix(a: string, b: string, t: number) {
  const x = hex(a); const y = hex(b)
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(',')})`
}

/** Sequential blue for 0..1. */
export function sequential(t: number) {
  const x = Math.max(0, Math.min(1, t)) * (SEQ.length - 1)
  const i = Math.min(Math.floor(x), SEQ.length - 2)
  return mix(SEQ[i], SEQ[i + 1], x - i)
}

/** Diverging blue (negative) - grey - red (positive) for -1..1. */
export function diverging(t: number) {
  const v = Math.max(-1, Math.min(1, t))
  return v < 0 ? mix('#f0efec', '#256abf', -v) : mix('#f0efec', '#d03b3b', v)
}

/** Text colour that stays readable on a given fill strength. */
export const inkOn = (strength: number) => (Math.abs(strength) > 0.55 ? '#ffffff' : '#1e1b2e')
