import type { DayRecord, RunState } from '../../api'
import { money, num } from '../../format'
import LineChart from '../LineChart'
import { type Act, PanelHead, RouteTable, Stat, lane, regionName } from './common'

export default function WarehousePanel({ state, records, act, region }: { state: RunState; records: DayRecord[]; act: Act; region: number }) {
  const w = state.warehouses.find((x) => x.region === region)!
  const name = regionName(state, region)
  const pairs = state.factories.map((f) => [f.region, region] as [number, number])

  return (
    <>
      <PanelHead eyebrow="Warehouse" title={name}
        pill={w.operational ? 'OPERATIONAL' : 'UNDER CONSTRUCTION'} pillKind={w.operational ? 'ok' : 'wait'} />
      {!w.operational && w.online_at != null && <p className="note">Comes online on day {Math.ceil(w.online_at)}.</p>}

      <div className="stats">
        <Stat label="On hand" value={num(w.on_hand)} />
        <Stat label="En route, truck" value={num(w.en_route_truck)} />
        <Stat label="En route, mail" value={num(w.en_route_mail)} />
      </div>

      <div className="section">
        <h3>Markets served</h3>
        <p className="note">Which customer regions this warehouse may ship to. Fulfilment is by mail; the cost depends on distance.</p>
        <div className="toggle-row">
          {state.scenario.regions.map((r) => {
            const on = w.serves.includes(r.id)
            return (
              <label key={r.id} className="check">
                <input type="checkbox" checked={on}
                  onChange={() => act({ kind: 'set_service', region: r.id, warehouse: region, allowed: !on })} />
                <span>{r.name}</span>
                <span className="muted small mono">{money(lane(state, region, r.id).mail_cost_per_drum)}</span>
              </label>
            )
          })}
        </div>
      </div>

      <div className="section">
        <h3>Inbound shipments</h3>
        {w.shipments.length ? (
          <div className="table-wrap">
            <table className="grid">
              <thead><tr><th>From</th><th>Mode</th><th className="num">Drums</th><th className="num">Arrives</th></tr></thead>
              <tbody>
                {w.shipments.map((s, i) => (
                  <tr key={i}>
                    <td>{regionName(state, s.factory)}</td><td>{s.mode}</td>
                    <td className="num">{num(s.quantity)}</td><td className="num">day {Math.ceil(s.arrive)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="note">Nothing on the road.</p>}
      </div>

      <div className="section">
        <h3>Inbound routes from factories</h3>
        <RouteTable state={state} act={act} pairs={pairs} rowLabel={(f) => `from ${regionName(state, f)}`} />
      </div>

      <LineChart
        title="Inventory"
        x={records.map((r) => r.day)}
        series={[
          { id: 'on_hand', label: 'On hand', color: '#2a78d6', values: records.map((r) => r.on_hand[region] ?? 0) },
          { id: 'truck', label: 'En route, truck', color: '#eb6834', values: records.map((r) => r.en_route_truck[region] ?? 0) },
          { id: 'mail', label: 'En route, mail', color: '#1baf7a', values: records.map((r) => r.en_route_mail[region] ?? 0) },
        ]}
      />
    </>
  )
}
