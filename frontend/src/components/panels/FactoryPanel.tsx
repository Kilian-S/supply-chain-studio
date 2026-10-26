import { useState } from 'react'
import type { DayRecord, RunState } from '../../api'
import { money, num, regionSeries } from '../../format'
import LineChart from '../LineChart'
import { type Act, PanelHead, RouteTable, Stat, modeLabel, regionName } from './common'

export default function FactoryPanel({ state, records, act, region }: { state: RunState; records: DayRecord[]; act: Act; region: number }) {
  const f = state.factories.find((x) => x.region === region)!
  const econ = state.scenario.economics
  const scheduled = f.capacity + f.pending_capacity.reduce((s, p) => s + p.amount, 0)
  const [total, setTotal] = useState('')
  const target = Number(total)
  const addition = total !== '' && Number.isFinite(target) ? target - scheduled : 0
  const name = regionName(state, region)

  const progress = f.batch ? Math.max(0, Math.min(1, (state.time - f.batch.start) / (f.batch.end - f.batch.start))) : 0
  const pairs = state.warehouses.map((w) => [region, w.region] as [number, number])

  return (
    <>
      <PanelHead eyebrow="Factory" title={name}
        pill={f.operational ? 'OPERATIONAL' : 'UNDER CONSTRUCTION'} pillKind={f.operational ? 'ok' : 'wait'} />
      {!f.operational && f.online_at != null && <p className="note">Comes online on day {Math.ceil(f.online_at)}. You can set its routes now.</p>}

      <div className="stats">
        <Stat label="Capacity" value={`${num(f.capacity, 2)}/day`} />
        <Stat label="Scheduled" value={`${num(scheduled, 2)}/day`} />
        <Stat label="Work in progress" value={f.batch ? num(f.batch.quantity) : '0'} />
      </div>

      <div className="section">
        <h3>Production</h3>
        {f.batch ? (
          <>
            <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
              <div style={{ width: `${progress * 100}%` }} />
            </div>
            <p className="note">
              Producing {num(f.batch.quantity)} drums for the {regionName(state, f.batch.warehouse)} warehouse, shipped by{' '}
              {modeLabel(state, region, f.batch.warehouse, f.batch.mode)}; finishes on day {Math.ceil(f.batch.end)}.
            </p>
          </>
        ) : <p className="note">Idle. A batch starts as soon as a route's order point is reached and cash covers it.</p>}
        <p className="note">
          A batch costs {money(econ.batch_fixed_cost)} + {money(econ.unit_cost)} per drum; shipping is paid when production starts.
          One batch at a time; time = quantity / capacity.
        </p>
      </div>

      {state.scenario.rules.allow_capacity_additions && (
        <div className="section">
          <h3>Capacity</h3>
          {f.pending_capacity.map((p, i) => (
            <p className="note" key={i}>+{num(p.amount, 2)} drums/day arriving on day {Math.ceil(p.online_at)}.</p>
          ))}
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <label className="field" style={{ flex: 1 }}>
              New total capacity (drums/day)
              <input className="input" inputMode="decimal" placeholder={num(scheduled, 2)} value={total}
                onChange={(e) => setTotal(e.target.value)} />
            </label>
            <button className="btn primary" disabled={!(addition > 0)}
              onClick={async () => { if (await act({ kind: 'set_capacity', region, total: target })) setTotal('') }}>
              Schedule
            </button>
          </div>
          <p className="note">
            {addition > 0
              ? `Adds ${num(addition, 2)}/day for ${money(addition * econ.capacity_cost_per_unit)}, paid now, ready in ${econ.capacity_build_days} days.`
              : `${money(econ.capacity_cost_per_unit)} per drum/day of capacity, ready after ${econ.capacity_build_days} days. Capacity can never be removed.`}
          </p>
        </div>
      )}

      <div className="section">
        <h3>Production and shipping</h3>
        <RouteTable state={state} act={act} pairs={pairs} rowLabel={(_, w) => `to ${regionName(state, w)}`} />
      </div>

      <LineChart
        title="Work in progress"
        x={records.map((r) => r.day)}
        series={[{ id: 'wip', label: 'WIP', color: regionSeries(region), values: records.map((r) => r.wip[region] ?? 0) }]}
      />
    </>
  )
}
