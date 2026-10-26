import { useEffect, useState } from 'react'
import type { Decision, LaneTerms, Mode, Route, RunState } from '../../api'
import { money } from '../../format'

export type Act = (d: Decision | Decision[]) => Promise<boolean>

export const regionName = (state: RunState, id: number) =>
  state.scenario.regions.find((r) => r.id === id)?.name ?? `Region ${id}`

export function lane(state: RunState, a: number, b: number): LaneTerms {
  const ra = state.scenario.regions.find((r) => r.id === a)!
  const rb = state.scenario.regions.find((r) => r.id === b)!
  const tier = a === b ? 'same_region' : ra.landmass === rb.landmass ? 'same_landmass' : 'cross_water'
  return state.scenario.lanes[tier]
}

export const modeLabel = (state: RunState, a: number, b: number, mode: Mode) => {
  const l = lane(state, a, b)
  return mode === 'truck'
    ? `truck ${money(l.truck_cost)}, ${l.truck_days} d`
    : `mail ${money(l.mail_cost_per_drum)}/drum, ${l.mail_days} d`
}

const short = (v: number) => (v >= 1000 ? `$${v / 1000}k` : `$${v}`)

export const modeShort = (state: RunState, a: number, b: number, mode: Mode) => {
  const l = lane(state, a, b)
  return mode === 'truck' ? `truck ${short(l.truck_cost)} · ${l.truck_days} d` : `mail ${short(l.mail_cost_per_drum)}/drum · ${l.mail_days} d`
}

export function PanelHead({ eyebrow, title, pill, pillKind }: { eyebrow: string; title: string; pill?: string; pillKind?: 'ok' | 'wait' | 'off' }) {
  return (
    <div className="panel-head">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h2>{title}</h2>
      </div>
      {pill && <span className={`pill ${pillKind ?? 'ok'}`}>{pill}</span>}
    </div>
  )
}

export function Stat({ label, value }: { label: string; value: string }) {
  return <div className="stat"><span>{label}</span><span>{value}</span></div>
}

type Draft = Pick<Route, 'order_point' | 'quantity' | 'mode' | 'priority'>
const blank: Draft = { order_point: 0, quantity: 0, mode: 'truck', priority: 0 }

/**
 * The original's "Production and Shipping" table. One row per factory-warehouse pair; edits are
 * collected and applied together with one button, each row as a single atomic change (the
 * original applied fields one by one, which made the outcome depend on the order of changes).
 */
export function RouteTable({ state, act, pairs, rowLabel }: {
  state: RunState
  act: Act
  pairs: [number, number][] // [factory, warehouse]
  rowLabel: (factory: number, warehouse: number) => string
}) {
  const current = (f: number, w: number): Draft =>
    state.routes.find((r) => r.factory === f && r.warehouse === w) ?? blank
  const key = (f: number, w: number) => `${f}-${w}`
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})

  // Drop drafts that now equal the live settings (after applying, or if the run changes).
  useEffect(() => {
    setDrafts((ds) => {
      const out: Record<string, Draft> = {}
      for (const [k, d] of Object.entries(ds)) {
        const [f, w] = k.split('-').map(Number)
        const c = current(f, w)
        if (d.order_point !== c.order_point || d.quantity !== c.quantity || d.mode !== c.mode || d.priority !== c.priority) out[k] = d
      }
      return out
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.routes])

  const value = (f: number, w: number) => drafts[key(f, w)] ?? current(f, w)
  const set = (f: number, w: number, patch: Partial<Draft>) =>
    setDrafts((ds) => ({ ...ds, [key(f, w)]: { ...value(f, w), ...patch } }))
  const changed = (f: number, w: number, field: keyof Draft) =>
    drafts[key(f, w)] != null && drafts[key(f, w)][field] !== current(f, w)[field]
  const intIn = (raw: string) => Math.max(0, Math.floor(Number(raw) || 0))

  const apply = async () => {
    const decisions: Decision[] = Object.entries(drafts).map(([k, d]) => {
      const [factory, warehouse] = k.split('-').map(Number)
      return { kind: 'set_route', factory, warehouse, ...d }
    })
    if (decisions.length && (await act(decisions))) setDrafts({})
  }

  if (!pairs.length) return <p className="note">No routes yet: this needs at least one factory and one warehouse.</p>

  return (
    <div className="section">
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>Route</th><th>Shipping</th>
              <th className="num" title="Order point">Order pt</th><th className="num" title="Quantity">Qty</th>
              <th className="num" title="Priority (highest first)">Prio</th>
            </tr>
          </thead>
          <tbody>
            {pairs.map(([f, w]) => {
              const v = value(f, w)
              return (
                <tr key={key(f, w)}>
                  <td>{rowLabel(f, w)}</td>
                  <td style={{ minWidth: 128 }}>
                    <select className={`select ${changed(f, w, 'mode') ? 'changed' : ''}`} value={v.mode}
                      aria-label={`Shipping method ${rowLabel(f, w)}`}
                      onChange={(e) => set(f, w, { mode: e.target.value as Mode })}>
                      <option value="truck">{modeShort(state, f, w, 'truck')}</option>
                      <option value="mail">{modeShort(state, f, w, 'mail')}</option>
                    </select>
                  </td>
                  {(['order_point', 'quantity', 'priority'] as const).map((field) => (
                    <td key={field} style={{ width: field === 'priority' ? 50 : field === 'quantity' ? 62 : 72 }}>
                      <input className={`input ${changed(f, w, field) ? 'changed' : ''}`} inputMode="numeric"
                        aria-label={`${field.replace('_', ' ')} ${rowLabel(f, w)}`}
                        value={v[field]} onChange={(e) => set(f, w, { [field]: intIn(e.target.value) })} />
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button className="btn primary" disabled={!Object.keys(drafts).length} onClick={apply}>Apply changes</button>
        {Object.keys(drafts).length > 0 && <button className="btn" onClick={() => setDrafts({})}>Discard</button>}
        <span className="note">A batch starts when stock on hand plus en route is at or below the order point.</span>
      </div>
    </div>
  )
}
