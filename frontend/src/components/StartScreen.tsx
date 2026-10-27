import { useEffect, useState } from 'react'
import { api, type RunState, type ScenarioListItem } from '../api'

interface Props {
  onStart: (run: RunState) => void
  onDesign: (scenarioId: string, takenIds: string[]) => void
}

export default function StartScreen({ onStart, onDesign }: Props) {
  const [scenarios, setScenarios] = useState<ScenarioListItem[]>([])
  const [chosen, setChosen] = useState<string>('')
  const [seed, setSeed] = useState('0')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.scenarios()
      .then((list) => { setScenarios(list); setChosen((c) => c || list[0]?.id || '') })
      .catch((e) => setError(`Could not reach the backend: ${e.message}. Is it running on port 8000?`))
  }, [])

  const start = async () => {
    setBusy(true)
    try {
      onStart(await api.newRun(chosen, Number(seed) || 0))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const ids = scenarios.map((s) => s.id)
  const groups: [string, ScenarioListItem[]][] = [
    ['Standard scenarios', scenarios.filter((s) => s.standard)],
    ['Your scenarios', scenarios.filter((s) => !s.standard)],
  ]

  return (
    <div className="start">
      <div className="start-inner">
        <div>
          <div className="eyebrow">New run</div>
          <h1>Choose a scenario</h1>
          <p className="note" style={{ maxWidth: 640, marginTop: 8 }}>
            Run a chemical producer's supply chain on the continent of Pangea: build factories and warehouses, set production
            and shipping, choose which markets to serve, and finish with as much cash as possible. Scenarios fix the world and
            a timeline of events (shocks, disruptions, price changes) before play starts.
          </p>
        </div>
        {error && <div className="error">{error}</div>}
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <label className="field">
            Random seed (order timing)
            <input className="input" style={{ width: 120 }} inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </label>
          <button className="btn primary" disabled={!chosen || busy} onClick={start}>Start run</button>
          <button className="btn" disabled={!chosen} onClick={() => onDesign(chosen, ids)}>Open in designer</button>
          <button className="btn" onClick={() => onDesign('classic-network', ids)}>New scenario</button>
        </div>
        {groups.map(([title, list]) => list.length > 0 && (
          <section key={title} className="section">
            <h3>{title}</h3>
            <div className="cards">
              {list.map((s) => (
                <button key={s.id} className="card" aria-pressed={chosen === s.id} onClick={() => setChosen(s.id)}
                  onDoubleClick={() => onDesign(s.id, ids)}>
                  <h3>{s.name}</h3>
                  <span className="note">{s.description}</span>
                  <span className="toggle-row" style={{ marginTop: 'auto' }}>
                    <span className="tag">{s.regions} region{s.regions > 1 ? 's' : ''}</span>
                    <span className="tag">days {s.start_day} to {s.end_day}</span>
                    <span className="tag">{s.events ? `${s.events} event${s.events > 1 ? 's' : ''}` : 'no events'}</span>
                  </span>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  )
}
