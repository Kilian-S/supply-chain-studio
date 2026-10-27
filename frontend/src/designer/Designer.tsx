import { useEffect, useMemo, useState } from 'react'
import { api, type RunState, type ScenarioDoc, type ScenarioEvent, type ScenarioPreview } from '../api'
import LineChart from '../components/LineChart'
import { regionSeries } from '../format'
import EventEditor from './EventEditor'
import EventTimeline from './EventTimeline'
import SettingsForm from './SettingsForm'
import { KINDS, newEvent } from './eventKinds'

interface Props {
  scenarioId: string
  takenIds: string[]
  onBack: () => void
  onPlay: (run: RunState) => void
}

// 7-day moving average, so daily noise does not hide what the events do.
const smooth = (xs: number[]) => xs.map((_, i) => {
  const w = xs.slice(Math.max(0, i - 6), i + 1)
  return w.reduce((a, b) => a + b, 0) / w.length
})

export default function Designer({ scenarioId, takenIds, onBack, onPlay }: Props) {
  const [doc, setDoc] = useState<ScenarioDoc | null>(null)
  const [savedAs, setSavedAs] = useState<string | null>(null) // id on disk, once saved
  const [savedJson, setSavedJson] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const [preview, setPreview] = useState<ScenarioPreview | null>(null)
  const [problem, setProblem] = useState('')
  const [status, setStatus] = useState('')

  useEffect(() => {
    api.scenario(scenarioId).then((d) => {
      if (d.standard) {
        // Standard scenarios are read-only: open an editable copy.
        let id = `${d.id}-copy`
        for (let n = 2; takenIds.includes(id); n++) id = `${d.id}-copy-${n}`
        setDoc({ ...d, id, name: `${d.name} (copy)`, standard: false })
      } else {
        setDoc(d)
        setSavedAs(d.id)
        setSavedJson(JSON.stringify(d))
      }
    }).catch((e) => setProblem(e.message))
  }, [scenarioId, takenIds])

  // Live preview of what the timeline does, debounced while editing.
  useEffect(() => {
    if (!doc) return
    const t = setTimeout(() => {
      api.preview(doc).then((p) => { setPreview(p); setProblem('') }).catch((e) => setProblem(e.message))
    }, 350)
    return () => clearTimeout(t)
  }, [doc])

  const dirty = doc != null && JSON.stringify(doc) !== savedJson

  const charts = useMemo(() => {
    if (!preview || !doc) return null
    const names = Object.keys(preview.with_events)
    const total = (src: Record<string, number[]>) => preview.days.map((_, i) => names.reduce((a, n) => a + src[n][i], 0))
    return {
      total: [
        { id: 'base', label: 'Recorded', color: '#8a8699', values: smooth(total(preview.baseline)) },
        { id: 'ev', label: 'With events', color: '#2a78d6', values: smooth(total(preview.with_events)) },
      ],
      regional: doc.regions.map((r) => ({ id: String(r.id), label: r.name, color: regionSeries(r.id), values: preview.demand_multiplier[r.name] })),
      money: [
        { id: 'production', label: 'Production', color: '#2a78d6', values: preview.multipliers.production },
        { id: 'transport', label: 'Transport', color: '#eb6834', values: preview.multipliers.transport },
        { id: 'fulfilment', label: 'Fulfilment', color: '#1baf7a', values: preview.multipliers.fulfilment },
        { id: 'holding', label: 'Holding', color: '#eda100', values: preview.multipliers.holding },
        { id: 'price', label: 'Price', color: '#e87ba4', values: preview.multipliers.price },
      ],
      interest: [{ id: 'rate', label: 'Interest rate', color: '#008300', values: preview.multipliers.interest_rate.map((r) => r * 100) }],
    }
  }, [preview, doc])

  if (!doc) return <div className="start">{problem ? <div className="error">{problem}</div> : 'Loading...'}</div>

  const { start_day, end_day } = doc.timeline
  const event = doc.events.find((e) => e.id === selected) ?? null
  const setEvent = (e: ScenarioEvent) => setDoc({ ...doc, events: doc.events.map((x) => (x.id === e.id ? e : x)) })
  const addEvent = (kind: ScenarioEvent['kind']) => {
    const day = Math.round(start_day + (end_day - start_day) / 3)
    const e = newEvent(kind, day, end_day, doc.events.map((x) => x.id))
    setDoc({ ...doc, events: [...doc.events, e] })
    setSelected(e.id)
  }

  const save = async (): Promise<string | null> => {
    setStatus('')
    try {
      if (!savedAs && takenIds.includes(doc.id)) throw new Error(`a scenario called "${doc.id}" already exists; choose another file id`)
      const saved = await api.saveScenario(doc)
      setDoc(saved)
      setSavedAs(saved.id)
      setSavedJson(JSON.stringify(saved))
      setStatus('Saved')
      return saved.id
    } catch (e) {
      setProblem((e as Error).message)
      return null
    }
  }
  const play = async () => {
    const id = dirty || !savedAs ? await save() : savedAs
    if (id) onPlay(await api.newRun(id))
  }
  const remove = async () => {
    if (!savedAs) return onBack()
    await api.deleteScenario(savedAs)
    onBack()
  }

  return (
    <div className="designer">
      <div className="designer-bar">
        <button className="btn" onClick={onBack}>Back</button>
        <div style={{ minWidth: 0 }}>
          <div className="eyebrow">Scenario designer</div>
          <strong className="designer-title">{doc.name}</strong>
        </div>
        <div className="spacer" />
        <span className="muted small" role="status">{status && !dirty ? status : dirty ? 'Unsaved changes' : ''}</span>
        {savedAs && <button className="btn" onClick={remove}>Delete</button>}
        <button className="btn" onClick={save} disabled={!dirty && savedAs != null}>Save</button>
        <button className="btn primary" onClick={play}>Save and play</button>
      </div>
      {problem && <div className="error designer-error" role="alert">{problem}</div>}
      <div className="designer-body">
        <aside className="designer-side">
          <SettingsForm doc={doc} onChange={setDoc} idEditable={!savedAs} />
        </aside>
        <main className="designer-main">
          <section className="section">
            <div className="panel-head" style={{ alignItems: 'center' }}>
              <div>
                <h3>Event timeline</h3>
                <p className="note">Drag a bar to move it, drag its ends to change the dates; striped bars are surprises.</p>
              </div>
            </div>
            <div className="toggle-row" aria-label="Add an event">
              {KINDS.map((k) => (
                <button key={k.kind} className="btn small" title={k.hint} onClick={() => addEvent(k.kind)}>
                  <i className="swatch" style={{ background: k.color, width: 10, height: 10, borderRadius: 3 }} />+ {k.label}
                </button>
              ))}
            </div>
            <EventTimeline events={doc.events} startDay={start_day} endDay={end_day} selected={selected}
              onSelect={setSelected} onChange={setEvent} />
          </section>

          {event && (
            <EventEditor event={event} regions={doc.regions} startDay={start_day} endDay={end_day} onChange={setEvent}
              onDelete={() => { setDoc({ ...doc, events: doc.events.filter((x) => x.id !== event.id) }); setSelected(null) }}
              onDuplicate={() => {
                const id = newEvent(event.kind, start_day, end_day, doc.events.map((x) => x.id)).id
                const copy = { ...event, id, name: `${event.name} (copy)` }
                setDoc({ ...doc, events: [...doc.events, copy] })
                setSelected(copy.id)
              }} />
          )}

          {charts && preview && (
            <div className="preview-grid">
              <LineChart title="Total demand, drums/day (7-day average)" x={preview.days} series={charts.total} />
              <LineChart title="Demand multiplier by region" x={preview.days} series={charts.regional} format={(v) => v.toFixed(2)} />
              <LineChart title="Cost and price multipliers" x={preview.days} series={charts.money} format={(v) => v.toFixed(2)} />
              <LineChart title="Interest rate (% a year)" x={preview.days} series={charts.interest} format={(v) => v.toFixed(1)} />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}
