import { useEffect, useRef, useState } from 'react'
import { api, type DayRecord, type DemandHistory, type Policy, type RunState } from '../../api'
import { money, moneyShort, regionSeries } from '../../format'
import LineChart from '../LineChart'
import { type Act, PanelHead } from './common'

type Range = 'play' | '180' | 'all'

export default function HQPanel({ state, records, act }: { state: RunState; records: DayRecord[]; act: Act }) {
  const [demand, setDemand] = useState<DemandHistory | null>(null)
  const [range, setRange] = useState<Range>('play')
  const inFlight = useRef(false)
  const econ = state.scenario.economics

  useEffect(() => {
    if (inFlight.current) return
    inFlight.current = true
    api.demand(state.run_id).then(setDemand).finally(() => { inFlight.current = false })
  }, [state.run_id, state.day])

  const start = state.scenario.timeline.start_day
  const from = !demand ? 0
    : range === 'all' ? 0
    : range === '180' ? Math.max(0, demand.days.length - 180)
    : Math.max(0, demand.days.findIndex((d) => d >= start - 60))
  const days = demand ? demand.days.slice(from) : []
  const regions = state.scenario.regions
  const policies = state.scenario.rules.allowed_policies

  return (
    <>
      <PanelHead eyebrow="Headquarters" title="Corporate data and settings" />
      <p className="note">
        Revenue is {money(econ.price_per_drum)} per drum. An order not shipped within {state.scenario.rules.lost_sale_window_days} day
        is lost to a competitor. Cash earns {(econ.interest_rate_annual * 100).toFixed(0)}% a year, compounded daily.
      </p>

      <div className="section">
        <h3>Fulfilment policy</h3>
        <div className="toggle-row" role="radiogroup" aria-label="Fulfilment policy">
          {policies.map((p: Policy) => (
            <button key={p} className={`btn ${state.policy === p ? 'primary' : ''}`} role="radio" aria-checked={state.policy === p}
              onClick={() => state.policy !== p && act({ kind: 'set_policy', policy: p })}>
              {p === 'nearest' ? 'Nearest' : 'Soonest'}
            </button>
          ))}
        </div>
        <p className="note">
          When several warehouses could fill an order: <b>nearest</b> uses the closest warehouse that has stock or will have it within
          the window; <b>soonest</b> uses stock on hand anywhere first.
        </p>
      </div>

      <div className="toggle-row" aria-label="Chart range">
        {([['play', 'Play window'], ['180', 'Last 180 days'], ['all', 'All history']] as [Range, string][]).map(([r, label]) => (
          <button key={r} className="btn small" aria-pressed={range === r} style={range === r ? { background: '#e8e6fb' } : undefined}
            onClick={() => setRange(r)}>{label}</button>
        ))}
      </div>

      {demand && (
        <>
          <LineChart
            title="Demand per region (drums/day)"
            x={days}
            series={regions.map((r) => ({ id: String(r.id), label: r.name, color: regionSeries(r.id), values: (demand.demand[r.name] ?? []).slice(from) }))}
          />
          <LineChart
            title="Lost demand (drums/day)"
            x={days}
            series={regions.map((r) => ({ id: String(r.id), label: r.name, color: regionSeries(r.id), values: (demand.lost[r.name] ?? []).slice(from) }))}
          />
        </>
      )}
      <LineChart
        title="Cash balance"
        x={[start - 1, ...records.map((r) => r.day)]}
        series={[{ id: 'cash', label: 'Cash', color: '#2a78d6', values: [state.starting_cash, ...records.map((r) => r.cash)] }]}
        format={moneyShort}
      />
    </>
  )
}
