import type { Policy, ScenarioDoc } from '../api'

interface Props {
  doc: ScenarioDoc
  onChange: (doc: ScenarioDoc) => void
  idEditable: boolean
}

type Econ = ScenarioDoc['economics']

const ECON_FIELDS: [keyof Econ, string, string][] = [
  ['price_per_drum', 'Selling price', '$/drum'],
  ['batch_fixed_cost', 'Batch fixed cost', '$'],
  ['unit_cost', 'Production cost', '$/drum'],
  ['holding_cost_per_drum_year', 'Holding cost', '$/drum/yr'],
  ['interest_rate_annual', 'Interest rate', 'per year'],
  ['capacity_cost_per_unit', 'Capacity cost', '$ per drum/day'],
  ['capacity_build_days', 'Capacity lead time', 'days'],
  ['factory_build_cost', 'Factory building', '$'],
  ['factory_build_days', 'Factory lead time', 'days'],
  ['warehouse_build_cost', 'Warehouse', '$'],
  ['warehouse_build_days', 'Warehouse lead time', 'days'],
  ['truck_capacity', 'Truck capacity', 'drums'],
]

function Num({ label, value, unit, onChange, step }: { label: string; value: number; unit?: string; onChange: (v: number) => void; step?: number }) {
  return (
    <label className="field">
      <span>{label}{unit && <span className="muted"> ({unit})</span>}</span>
      <input className="input" type="number" step={step ?? 'any'} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export default function SettingsForm({ doc, onChange, idEditable }: Props) {
  const set = (patch: Partial<ScenarioDoc>) => onChange({ ...doc, ...patch })
  const regionName = (id: number) => doc.regions.find((r) => r.id === id)?.name ?? String(id)
  const start = doc.start
  const setStart = (patch: Partial<ScenarioDoc['start']>) => set({ start: { ...start, ...patch } })
  const buildable = doc.rules.buildable_regions
  const free = (taken: number[]) => doc.regions.find((r) => !taken.includes(r.id))?.id

  return (
    <div className="settings">
      <section className="section">
        <h3>Basics</h3>
        <label className="field">Name<input className="input sans" value={doc.name} onChange={(e) => set({ name: e.target.value })} /></label>
        <label className="field">
          File id {idEditable ? '(lower case, digits, hyphens)' : ''}
          <input className="input" value={doc.id} disabled={!idEditable}
            onChange={(e) => set({ id: e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-') })} />
        </label>
        <label className="field">Description
          <textarea className="input sans" rows={3} value={doc.description} onChange={(e) => set({ description: e.target.value })} />
        </label>
        <div className="form-grid">
          <Num label="First play day" value={doc.timeline.start_day} onChange={(v) => set({ timeline: { ...doc.timeline, start_day: Math.round(v) } })} />
          <Num label="Last play day" value={doc.timeline.end_day} onChange={(v) => set({ timeline: { ...doc.timeline, end_day: Math.round(v) } })} />
          <Num label="Random seed" value={doc.seed} onChange={(v) => set({ seed: Math.round(v) })} />
        </div>
      </section>

      <section className="section">
        <h3>Economics</h3>
        <div className="form-grid">
          {ECON_FIELDS.map(([k, label, unit]) => (
            <Num key={k} label={label} unit={unit} value={doc.economics[k] as number}
              onChange={(v) => set({ economics: { ...doc.economics, [k]: v } })} />
          ))}
        </div>
        <label className="check">
          <input type="checkbox" checked={doc.economics.holding_cost_in_transit}
            onChange={() => set({ economics: { ...doc.economics, holding_cost_in_transit: !doc.economics.holding_cost_in_transit } })} />
          Holding cost also applies in transit
        </label>
      </section>

      <section className="section">
        <h3>Rules</h3>
        <div className="form-grid">
          <Num label="Lost-sale window" unit="days" value={doc.rules.lost_sale_window_days}
            onChange={(v) => set({ rules: { ...doc.rules, lost_sale_window_days: v } })} />
        </div>
        <div className="toggle-row">
          {(['nearest', 'soonest'] as Policy[]).map((p) => (
            <label key={p} className="check">
              <input type="checkbox" checked={doc.rules.allowed_policies.includes(p)}
                onChange={() => set({ rules: { ...doc.rules, allowed_policies: doc.rules.allowed_policies.includes(p) ? doc.rules.allowed_policies.filter((x) => x !== p) : [...doc.rules.allowed_policies, p] } })} />
              Allow {p}
            </label>
          ))}
        </div>
        {([
          ['cash_check_on_production', 'Production needs cash in hand'],
          ['allow_capacity_additions', 'Capacity can be added'],
          ['new_warehouse_serves_all', 'New warehouses serve every region'],
        ] as const).map(([k, label]) => (
          <label key={k} className="check">
            <input type="checkbox" checked={doc.rules[k]} onChange={() => set({ rules: { ...doc.rules, [k]: !doc.rules[k] } })} />
            {label}
          </label>
        ))}
        <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
          <legend style={{ marginBottom: 4 }}>Player may build in</legend>
          <div className="toggle-row">
            <label className="check"><input type="checkbox" checked={buildable == null}
              onChange={() => set({ rules: { ...doc.rules, buildable_regions: buildable == null ? [] : null } })} />Every region</label>
            {buildable != null && doc.regions.map((r) => (
              <label key={r.id} className="check">
                <input type="checkbox" checked={buildable.includes(r.id)}
                  onChange={() => set({ rules: { ...doc.rules, buildable_regions: buildable.includes(r.id) ? buildable.filter((x) => x !== r.id) : [...buildable, r.id] } })} />
                {r.name}
              </label>
            ))}
          </div>
        </fieldset>
      </section>

      <section className="section">
        <h3>Starting position</h3>
        <div className="form-grid">
          <Num label="Cash" unit="$" value={start.cash} onChange={(v) => setStart({ cash: v })} />
          <label className="field">Fulfilment policy
            <select className="select" value={start.fulfilment_policy} onChange={(e) => setStart({ fulfilment_policy: e.target.value as Policy })}>
              <option value="nearest">nearest</option><option value="soonest">soonest</option>
            </select>
          </label>
        </div>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>Factory in</th><th className="num">Capacity</th><th /></tr></thead>
            <tbody>
              {start.factories.map((f, i) => (
                <tr key={i}>
                  <td><select className="select" value={f.region} aria-label="Factory region"
                    onChange={(e) => setStart({ factories: start.factories.map((x, j) => j === i ? { ...x, region: Number(e.target.value) } : x) })}>
                    {doc.regions.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}</select></td>
                  <td><input className="input" type="number" value={f.capacity} aria-label="Capacity"
                    onChange={(e) => setStart({ factories: start.factories.map((x, j) => j === i ? { ...x, capacity: Number(e.target.value) } : x) })} /></td>
                  <td><button className="btn small" onClick={() => setStart({ factories: start.factories.filter((_, j) => j !== i) })}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button className="btn small" style={{ alignSelf: 'flex-start' }} disabled={free(start.factories.map((f) => f.region)) == null}
          onClick={() => setStart({ factories: [...start.factories, { region: free(start.factories.map((f) => f.region))!, capacity: 20 }] })}>
          Add factory
        </button>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>Warehouse in</th><th className="num">Stock</th><th>Serves</th><th /></tr></thead>
            <tbody>
              {start.warehouses.map((w, i) => (
                <tr key={i}>
                  <td><select className="select" value={w.region} aria-label="Warehouse region"
                    onChange={(e) => setStart({ warehouses: start.warehouses.map((x, j) => j === i ? { ...x, region: Number(e.target.value) } : x) })}>
                    {doc.regions.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}</select></td>
                  <td><input className="input" type="number" value={w.inventory} aria-label="Starting stock"
                    onChange={(e) => setStart({ warehouses: start.warehouses.map((x, j) => j === i ? { ...x, inventory: Math.round(Number(e.target.value)) } : x) })} /></td>
                  <td>
                    <select className="select" aria-label="Markets served at the start" value={w.serves == null ? 'home' : 'all'}
                      onChange={(e) => setStart({ warehouses: start.warehouses.map((x, j) => j === i ? { ...x, serves: e.target.value === 'home' ? null : doc.regions.map((r) => r.id) } : x) })}>
                      <option value="home">{regionName(w.region)} only</option><option value="all">all regions</option>
                    </select>
                  </td>
                  <td><button className="btn small" onClick={() => setStart({ warehouses: start.warehouses.filter((_, j) => j !== i) })}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button className="btn small" style={{ alignSelf: 'flex-start' }} disabled={free(start.warehouses.map((w) => w.region)) == null}
          onClick={() => setStart({ warehouses: [...start.warehouses, { region: free(start.warehouses.map((w) => w.region))!, inventory: 0, serves: null }] })}>
          Add warehouse
        </button>
      </section>

      <section className="section">
        <h3>Customer orders</h3>
        <p className="note">Daily demand comes from the recorded series; it is split into orders of this size (events reshape it on top).</p>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>Region</th><th className="num">Mean order</th><th className="num">Fixed size</th></tr></thead>
            <tbody>
              {Object.entries(doc.demand).map(([name, d]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td><input className="input" type="number" value={d.mean_order_size} disabled={d.fixed_order_size != null} aria-label={`Mean order size ${name}`}
                    onChange={(e) => set({ demand: { ...doc.demand, [name]: { ...d, mean_order_size: Number(e.target.value) } } })} /></td>
                  <td><input className="input" type="number" value={d.fixed_order_size ?? ''} placeholder="random" aria-label={`Fixed order size ${name}`}
                    onChange={(e) => set({ demand: { ...doc.demand, [name]: { ...d, fixed_order_size: e.target.value === '' ? null : Math.round(Number(e.target.value)) } } })} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
