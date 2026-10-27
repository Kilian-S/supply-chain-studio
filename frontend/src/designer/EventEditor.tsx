import type { CostItem, Region, ScenarioEvent } from '../api'
import { kindMeta, summary } from './eventKinds'

interface Props {
  event: ScenarioEvent
  regions: Region[]
  startDay: number
  endDay: number
  onChange: (e: ScenarioEvent) => void
  onDelete: () => void
  onDuplicate: () => void
}

const usesRegions = new Set(['demand', 'market_opening', 'cost', 'price', 'lane', 'outage', 'end_of_life'])
const permanentAllowed = new Set(['demand', 'market_opening', 'cost', 'price', 'lane', 'interest'])
const COST_ITEMS: CostItem[] = ['production', 'transport', 'fulfilment', 'holding']

// Factors are edited as a percentage change (0.7 <-> -30).
const toPct = (f: number) => Math.round((f - 1) * 1000) / 10
const fromPct = (p: number) => Math.max(0, 1 + p / 100)

function NumberField({ label, value, onChange, suffix, step = 1, min }: {
  label: string; value: number; onChange: (v: number) => void; suffix?: string; step?: number; min?: number
}) {
  return (
    <label className="field">
      {label}
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <input className="input" type="number" step={step} min={min} value={Number.isFinite(value) ? value : ''}
          onChange={(e) => onChange(Number(e.target.value))} style={{ width: '100%' }} />
        {suffix && <span className="muted small">{suffix}</span>}
      </span>
    </label>
  )
}

export default function EventEditor({ event: e, regions, startDay, endDay, onChange, onDelete, onDuplicate }: Props) {
  const set = (patch: Partial<ScenarioEvent>) => onChange({ ...e, ...patch })
  const meta = kindMeta(e.kind)
  const name = (id: number) => regions.find((r) => r.id === id)?.name ?? String(id)
  const allRegions = e.regions == null
  const toggleRegion = (id: number) => {
    const current = e.regions ?? []
    set({ regions: current.includes(id) ? current.filter((x) => x !== id) : [...current, id].sort() })
  }
  const clampDay = (d: number) => Math.max(startDay, Math.min(endDay, Math.round(d)))

  return (
    <div className="section editor">
      <div className="panel-head" style={{ alignItems: 'center' }}>
        <div>
          <div className="eyebrow" style={{ color: '#1e1b2e' }}>
            <i className="swatch" style={{ background: meta.color, width: 10, height: 10, borderRadius: 3, marginRight: 6 }} />
            {meta.label}
          </div>
          <p className="note" style={{ marginTop: 4 }}>{summary(e, name)}</p>
        </div>
        <div className="toggle-row">
          <button className="btn small" onClick={onDuplicate}>Duplicate</button>
          <button className="btn small" onClick={onDelete}>Delete</button>
        </div>
      </div>

      <div className="form-grid">
        <label className="field" style={{ gridColumn: 'span 2' }}>
          Name
          <input className="input" value={e.name} onChange={(x) => set({ name: x.target.value })} />
        </label>
        <NumberField label="Start day" value={e.start_day} onChange={(v) => set({ start_day: clampDay(v) })} />
        <label className="field">
          End day
          <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input className="input" type="number" value={e.end_day ?? ''} disabled={e.end_day == null}
              placeholder="open-ended" style={{ width: '100%' }}
              onChange={(x) => set({ end_day: clampDay(Number(x.target.value)) })} />
          </span>
        </label>
        {permanentAllowed.has(e.kind) && (
          <label className="check" style={{ gridColumn: 'span 2' }}>
            <input type="checkbox" checked={e.end_day == null}
              onChange={() => set({ end_day: e.end_day == null ? Math.min(e.start_day + 59, endDay) : null })} />
            Permanent (no end day)
          </label>
        )}
        {e.kind !== 'end_of_life' && e.kind !== 'interest' && e.kind !== 'outage' && (
          <NumberField label="Ramp-up (days)" value={e.ramp_days} min={0} onChange={(v) => set({ ramp_days: Math.max(0, Math.round(v)) })} />
        )}
        <fieldset className="field" style={{ border: 0, padding: 0, margin: 0 }}>
          <legend style={{ marginBottom: 4 }}>Player sees it</legend>
          <div className="toggle-row">
            {(['announced', 'surprise'] as const).map((v) => (
              <label key={v} className="check" style={{ minHeight: 32 }}>
                <input type="radio" name={`vis-${e.id}`} checked={e.visibility === v} onChange={() => set({ visibility: v })} />
                {v === 'announced' ? 'In advance' : 'Surprise'}
              </label>
            ))}
          </div>
        </fieldset>

        {(e.kind === 'demand' || e.kind === 'price') && (
          <NumberField label={e.kind === 'demand' ? 'Demand change' : 'Price change'} value={toPct(e.factor)} suffix="%"
            onChange={(v) => set({ factor: fromPct(v) })} />
        )}
        {e.kind === 'cost' && (
          <>
            <NumberField label="Cost change" value={toPct(e.factor)} suffix="%" onChange={(v) => set({ factor: fromPct(v) })} />
            <fieldset className="field" style={{ border: 0, padding: 0, margin: 0, gridColumn: 'span 2' }}>
              <legend style={{ marginBottom: 4 }}>Costs affected</legend>
              <div className="toggle-row">
                {COST_ITEMS.map((c) => (
                  <label key={c} className="check" style={{ minHeight: 32 }}>
                    <input type="checkbox" checked={e.cost_items.includes(c)}
                      onChange={() => set({ cost_items: e.cost_items.includes(c) ? e.cost_items.filter((x) => x !== c) : [...e.cost_items, c] })} />
                    {c}
                  </label>
                ))}
              </div>
            </fieldset>
          </>
        )}
        {e.kind === 'interest' && (
          <NumberField label="New interest rate" value={Math.round((e.rate ?? 0) * 1000) / 10} suffix="% a year" step={0.5}
            onChange={(v) => set({ rate: Math.max(0, v) / 100 })} />
        )}
        {e.kind === 'lane' && (
          <>
            <label className="check" style={{ gridColumn: 'span 2' }}>
              <input type="checkbox" checked={e.closed} onChange={() => set({ closed: !e.closed })} />
              Lanes closed (no shipments, no deliveries)
            </label>
            {!e.closed && (
              <>
                <NumberField label="Lead time multiplier" value={e.lead_time_factor} step={0.1} min={0.1}
                  onChange={(v) => set({ lead_time_factor: Math.max(0.1, v) })} />
                <NumberField label="Transport cost change" value={toPct(e.factor)} suffix="%" onChange={(v) => set({ factor: fromPct(v) })} />
              </>
            )}
          </>
        )}
        {e.kind === 'outage' && (
          <label className="field">
            Facility
            <select className="select" value={e.facility} onChange={(x) => set({ facility: x.target.value as 'factory' | 'warehouse' })}>
              <option value="factory">Factory (production pauses)</option>
              <option value="warehouse">Warehouse (no shipping out)</option>
            </select>
          </label>
        )}

        {usesRegions.has(e.kind) && (
          <fieldset className="field" style={{ border: 0, padding: 0, margin: 0, gridColumn: 'span 2' }}>
            <legend style={{ marginBottom: 4 }}>{e.kind === 'lane' ? 'Lanes touching' : 'Regions'}</legend>
            <div className="toggle-row">
              {e.kind !== 'market_opening' && e.kind !== 'outage' && (
                <label className="check" style={{ minHeight: 32 }}>
                  <input type="checkbox" checked={allRegions} onChange={() => set({ regions: allRegions ? [] : null })} />
                  All
                </label>
              )}
              {!allRegions && regions.map((r) => (
                <label key={r.id} className="check" style={{ minHeight: 32 }}>
                  <input type="checkbox" checked={(e.regions ?? []).includes(r.id)} onChange={() => toggleRegion(r.id)} />
                  {r.name}
                </label>
              ))}
            </div>
          </fieldset>
        )}
        <label className="field" style={{ gridColumn: 'span 2' }}>
          Note shown to the player
          <input className="input" style={{ fontFamily: 'var(--sans)' }} value={e.description} onChange={(x) => set({ description: x.target.value })} />
        </label>
      </div>
    </div>
  )
}
