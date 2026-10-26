// Formatting and shared look helpers.

export const money = (v: number, digits = 0) =>
  (v < 0 ? '-$' : '$') + Math.abs(v).toLocaleString('en-GB', { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const moneyShort = (v: number) => {
  const a = Math.abs(v)
  const sign = v < 0 ? '-' : ''
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(2)}M`
  if (a >= 1e3) return `${sign}$${(a / 1e3).toFixed(0)}k`
  return `${sign}$${a.toFixed(0)}`
}

export const num = (v: number, digits = 0) =>
  v.toLocaleString('en-GB', { maximumFractionDigits: digits, minimumFractionDigits: digits })

export const pct = (v: number) => `${(v * 100).toFixed(1)}%`

// Region colours by region id: a pastel for the map and a darker step of the same hue for chart
// lines (validated with the dataviz palette checker; colour follows the region everywhere).
const PASTELS = ['#f4b6c7', '#c8b5f0', '#ead77a', '#96dccb', '#d2ae86', '#b9d9f5', '#f6c79a', '#c5e3a6']
const SERIES = ['#d55181', '#6250d6', '#c98500', '#1baf7a', '#a8673a', '#2a78d6', '#eb6834', '#008300']

export const regionPastel = (id: number) => PASTELS[(id - 1) % PASTELS.length]
export const regionSeries = (id: number) => SERIES[(id - 1) % SERIES.length]
