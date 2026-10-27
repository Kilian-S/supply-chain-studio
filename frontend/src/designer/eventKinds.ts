import type { EventKind, ScenarioEvent } from '../api'

// One colour per event kind, from the validated categorical palette in fixed order. Bars always
// carry their name as a label, so colour is never the only cue.
export const KINDS: { kind: EventKind; label: string; color: string; hint: string }[] = [
  { kind: 'demand', label: 'Demand change', color: '#2a78d6', hint: 'Demand rises or falls by a factor.' },
  { kind: 'cost', label: 'Cost change', color: '#eb6834', hint: 'Production, transport, fulfilment or holding costs change.' },
  { kind: 'market_opening', label: 'Market opening', color: '#1baf7a', hint: 'A region has no demand until the start day, then ramps up.' },
  { kind: 'price', label: 'Price change', color: '#eda100', hint: 'The selling price changes for customers in some regions.' },
  { kind: 'lane', label: 'Lane disruption', color: '#e87ba4', hint: 'Lanes touching some regions close, slow down or get dearer.' },
  { kind: 'interest', label: 'Interest rate', color: '#008300', hint: 'The interest rate on cash changes.' },
  { kind: 'outage', label: 'Facility outage', color: '#6250d6', hint: 'A factory or warehouse stops working for a while.' },
  { kind: 'end_of_life', label: 'End of life', color: '#e34948', hint: 'Demand declines linearly to zero.' },
]

export const kindMeta = (k: EventKind) => KINDS.find((x) => x.kind === k)!

export function newEvent(kind: EventKind, day: number, endDay: number, taken: string[]): ScenarioEvent {
  let n = 1
  while (taken.includes(`${kind}-${n}`)) n++
  const base: ScenarioEvent = {
    id: `${kind}-${n}`, name: kindMeta(kind).label, kind, start_day: day, end_day: null, ramp_days: 0,
    visibility: 'announced', description: '', regions: null, factor: 1, cost_items: ['production', 'transport'],
    rate: null, closed: false, lead_time_factor: 1, facility: 'factory',
  }
  switch (kind) {
    case 'demand': return { ...base, name: 'Demand -30%', factor: 0.7 }
    case 'cost': return { ...base, name: 'Costs +25%', factor: 1.25 }
    case 'market_opening': return { ...base, name: 'Market opens', ramp_days: 30, regions: [] }
    case 'price': return { ...base, name: 'Price -10%', factor: 0.9 }
    case 'lane': return { ...base, name: 'Lanes closed', closed: true, end_day: Math.min(day + 59, endDay), regions: [] }
    case 'interest': return { ...base, name: 'Interest 4%', rate: 0.04 }
    case 'outage': return { ...base, name: 'Factory outage', end_day: Math.min(day + 29, endDay), regions: [] }
    case 'end_of_life': return { ...base, name: 'End of life', end_day: Math.min(day + 30, endDay) }
  }
}

export function summary(e: ScenarioEvent, regionName: (id: number) => string): string {
  const where = e.regions == null ? 'all regions' : e.regions.length ? e.regions.map(regionName).join(', ') : 'no region chosen'
  const pctChange = (f: number) => `${f >= 1 ? '+' : ''}${Math.round((f - 1) * 100)}%`
  const when = e.end_day == null ? `from day ${e.start_day}` : `days ${e.start_day} to ${e.end_day}`
  switch (e.kind) {
    case 'demand': return `Demand ${pctChange(e.factor)} in ${where}, ${when}`
    case 'cost': return `${e.cost_items.join(', ')} ${pctChange(e.factor)} (${where}), ${when}`
    case 'market_opening': return `${where} start ordering on day ${e.start_day}`
    case 'price': return `Price ${pctChange(e.factor)} in ${where}, ${when}`
    case 'lane': return `${e.closed ? 'Lanes closed' : `Lead times x${e.lead_time_factor}, costs ${pctChange(e.factor)}`} for ${where}, ${when}`
    case 'interest': return `Interest ${((e.rate ?? 0) * 100).toFixed(1)}%, ${when}`
    case 'outage': return `${e.facility === 'factory' ? 'Factory' : 'Warehouse'} out in ${where}, ${when}`
    case 'end_of_life': return `Demand falls to zero, ${when}`
  }
}
