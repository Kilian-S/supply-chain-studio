import { useEffect, useState } from 'react'
import { api, type DemandHistory, type RunState } from '../../api'
import { money, num, regionSeries } from '../../format'
import LineChart from '../LineChart'
import type { Selection } from '../MapView'
import { type Act, PanelHead, lane, regionName } from './common'

export default function RegionPanel({ state, act, region, onSelect }: { state: RunState; act: Act; region: number; onSelect: (s: Selection) => void }) {
  const r = state.scenario.regions.find((x) => x.id === region)!
  const econ = state.scenario.economics
  const rules = state.scenario.rules
  const factory = state.factories.find((f) => f.region === region)
  const warehouse = state.warehouses.find((w) => w.region === region)
  const buildable = rules.buildable_regions == null || rules.buildable_regions.includes(region)
  const [capacity, setCapacity] = useState('')
  const cap = Number(capacity)
  const [demand, setDemand] = useState<DemandHistory | null>(null)

  useEffect(() => { api.demand(state.run_id).then(setDemand) }, [state.run_id, state.day, region])

  const recent = demand ? demand.demand[r.name]?.slice(-180) ?? [] : []
  const days = demand ? demand.days.slice(-180) : []
  const avg30 = recent.length ? recent.slice(-30).reduce((a, b) => a + b, 0) / Math.min(30, recent.length) : 0

  return (
    <>
      <PanelHead eyebrow={r.landmass === 'island' ? 'Island region' : 'Region'} title={r.name} />
      <div className="stats">
        <div className="stat"><span>Demand, last 30 days</span><span>{num(avg30, 1)}/day</span></div>
        <div className="stat"><span>Factory</span><span>{factory ? (factory.operational ? 'yes' : 'building') : 'none'}</span></div>
        <div className="stat"><span>Warehouse</span><span>{warehouse ? (warehouse.operational ? 'yes' : 'building') : 'none'}</span></div>
      </div>

      <div className="toggle-row">
        {factory && <button className="btn" onClick={() => onSelect({ kind: 'factory', region })}>Open factory</button>}
        {warehouse && <button className="btn" onClick={() => onSelect({ kind: 'warehouse', region })}>Open warehouse</button>}
      </div>

      {buildable && !factory && (
        <div className="section">
          <h3>Build a factory</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <label className="field" style={{ flex: 1 }}>
              Capacity (drums/day)
              <input className="input" inputMode="decimal" value={capacity} onChange={(e) => setCapacity(e.target.value)} />
            </label>
            <button className="btn primary" disabled={!(cap > 0)}
              onClick={async () => { if (await act({ kind: 'set_capacity', region, total: cap })) setCapacity('') }}>Build</button>
          </div>
          <p className="note">
            {money(econ.factory_build_cost)} building + {money(econ.capacity_cost_per_unit)} per drum/day
            {cap > 0 ? ` = ${money(econ.factory_build_cost + cap * econ.capacity_cost_per_unit)}` : ''}, paid now; ready after {econ.factory_build_days} days.
          </p>
        </div>
      )}
      {buildable && !warehouse && (
        <div className="section">
          <h3>Build a warehouse</h3>
          <button className="btn primary" style={{ alignSelf: 'flex-start' }} onClick={() => act({ kind: 'build_warehouse', region })}>
            Build for {money(econ.warehouse_build_cost)}
          </button>
          <p className="note">Ready after {econ.warehouse_build_days} days. A new warehouse serves every region until you block some.</p>
        </div>
      )}
      {!buildable && <p className="note">This scenario does not allow building here.</p>}

      <div className="section">
        <h3>Transport from {r.name}</h3>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>To</th><th className="num">Truck (200)</th><th className="num">Days</th><th className="num">Mail/drum</th><th className="num">Days</th></tr></thead>
            <tbody>
              {state.scenario.regions.map((o) => {
                const l = lane(state, region, o.id)
                return (
                  <tr key={o.id}>
                    <td>{regionName(state, o.id)}</td><td className="num">{money(l.truck_cost)}</td><td className="num">{l.truck_days}</td>
                    <td className="num">{money(l.mail_cost_per_drum)}</td><td className="num">{l.mail_days}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {demand && (
        <LineChart title={`Demand in ${r.name}, last 180 days`} x={days}
          series={[{ id: 'd', label: r.name, color: regionSeries(region), values: recent }]} />
      )}
    </>
  )
}
