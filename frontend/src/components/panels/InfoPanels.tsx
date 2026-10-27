import { useEffect, useState } from 'react'
import { api, type CashStatement, type DayRecord, type LoggedDecision, type RunState } from '../../api'
import { money, num, pct } from '../../format'
import type { Selection } from '../MapView'
import { kindMeta, summary } from '../../designer/eventKinds'
import { PanelHead, Stat, regionName } from './common'

export function describe(state: RunState, d: LoggedDecision): string {
  const n = (id: unknown) => regionName(state, Number(id))
  switch (d.kind) {
    case 'set_route':
      return `Route ${n(d.factory)} factory to ${n(d.warehouse)} warehouse: order point ${num(Number(d.order_point))}, quantity ${num(Number(d.quantity))}, ${d.mode}, priority ${d.priority}`
    case 'set_policy':
      return `Fulfilment policy: ${d.policy}`
    case 'build_factory':
      return `Build factory in ${n(d.region)} with capacity ${num(Number(d.capacity), 2)} (${money(Number(d.cost))})`
    case 'add_capacity':
      return `Add ${num(Number(d.amount), 2)} capacity in ${n(d.region)} (${money(Number(d.cost))})`
    case 'build_warehouse':
      return `Build warehouse in ${n(d.region)} (${money(Number(d.cost))})`
    case 'set_service':
      return `${d.allowed ? 'Allow' : 'Block'} ${n(d.region)} demand from the ${n(d.warehouse)} warehouse`
    default:
      return d.kind
  }
}

export function HistoryPanel({ state, log }: { state: RunState; log: LoggedDecision[] }) {
  return (
    <>
      <PanelHead eyebrow="History" title="Your decisions" />
      {log.length === 0 ? <p className="note">No decisions yet.</p> : (
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th className="num">Time</th><th>Decision</th></tr></thead>
            <tbody>
              {[...log].reverse().map((d, i) => (
                <tr key={i}><td className="num">{d.time.toFixed(2)}</td><td>{describe(state, d)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

const LABELS: Record<string, string> = {
  revenue: 'Revenue', interest: 'Interest',
  capacity_and_buildings: 'Capacity and buildings', production: 'Production',
  inbound_shipping_truck: 'Inbound shipping, truck', inbound_shipping_mail: 'Inbound shipping, mail',
  outbound_shipping: 'Outbound shipping (fulfilment)', holding: 'Holding (warehouse and transit)',
}

export function CashPanel({ state }: { state: RunState }) {
  const [s, setS] = useState<CashStatement | null>(null)
  useEffect(() => { api.cashStatement(state.run_id).then(setS) }, [state.run_id, state.day])
  if (!s) return null
  return (
    <>
      <PanelHead eyebrow="Cash" title="Sources and uses of cash" />
      <div className="table-wrap">
        <table className="grid">
          <tbody>
            <tr><td>Starting cash</td><td className="num">{money(s.starting_cash, 2)}</td></tr>
            <tr><th colSpan={2}>Sources</th></tr>
            {Object.entries(s.sources).map(([k, v]) => <tr key={k}><td>{LABELS[k] ?? k}</td><td className="num">{money(v, 2)}</td></tr>)}
            <tr><th colSpan={2}>Uses</th></tr>
            {Object.entries(s.uses).map(([k, v]) => <tr key={k}><td>{LABELS[k] ?? k}</td><td className="num">{money(-v, 2)}</td></tr>)}
            <tr><th>Cash balance</th><th className="num">{money(s.cash, 2)}</th></tr>
          </tbody>
        </table>
      </div>
    </>
  )
}

export function OverviewPanel({ state, records, onSelect }: { state: RunState; records: DayRecord[]; onSelect: (s: Selection) => void }) {
  const s = state.summary
  const last30 = records.slice(-30)
  const sold30 = last30.reduce((a, r) => a + Object.values(r.fulfilled).reduce((x, y) => x + y, 0), 0)
  return (
    <>
      <PanelHead eyebrow={state.scenario.name} title={state.finished ? 'Run finished' : `Day ${state.day}`}
        pill={state.finished ? 'FINISHED' : 'IN PLAY'} pillKind={state.finished ? 'off' : 'ok'} />
      <p className="note">{state.scenario.description}</p>
      <div className="stats">
        <Stat label="Cash" value={money(state.cash)} />
        <Stat label="Fill rate" value={s.demand ? pct(s.fill_rate) : 'n/a'} />
        <Stat label="Lost drums" value={num(s.lost)} />
        <Stat label="Drums sold" value={num(s.fulfilled)} />
        <Stat label="Sold, last 30 days" value={num(sold30)} />
        <Stat label="Stock in network" value={num(s.leftover_inventory)} />
      </div>
      <div className="section">
        <h3>Scheduled events</h3>
        {state.events.length ? state.events.map((e) => (
          <p className="note" key={e.id}>
            <i className="swatch" style={{ background: kindMeta(e.kind).color, width: 10, height: 10, borderRadius: 3, marginRight: 6 }} />
            <b>{e.name}</b>{e.active ? ' (under way)' : ''}: {summary(e, (id) => regionName(state, id))}.{e.description ? ` ${e.description}` : ''}
          </p>
        )) : <p className="note">None announced. Surprises, if any, appear here when they happen.</p>}
      </div>
      <div className="section">
        <h3>How to play</h3>
        <p className="note">
          Click a region to build there, a factory to set capacity and production routes, a warehouse to see stock and choose the
          markets it serves, and headquarters for demand, lost sales, cash and the fulfilment policy. Advance time with the
          controls at the bottom; you can change anything between days. Play runs from day {state.scenario.timeline.start_day} to
          day {state.scenario.timeline.end_day}.
        </p>
        <button className="btn" style={{ alignSelf: 'flex-start' }} onClick={() => onSelect({ kind: 'hq' })}>Open headquarters</button>
      </div>
    </>
  )
}
