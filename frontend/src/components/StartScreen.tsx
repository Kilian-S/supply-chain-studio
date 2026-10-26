import { useEffect, useState } from 'react'
import { api, type RunState, type ScenarioListItem } from '../api'

export default function StartScreen({ onStart }: { onStart: (run: RunState) => void }) {
  const [scenarios, setScenarios] = useState<ScenarioListItem[]>([])
  const [chosen, setChosen] = useState<string>('')
  const [seed, setSeed] = useState('0')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.scenarios()
      .then((list) => { setScenarios(list); setChosen((c) => c || (list.find((s) => s.id === 'classic-network') ?? list[0])?.id || '') })
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

  return (
    <div className="start">
      <div className="start-inner">
        <div>
          <div className="eyebrow">New run</div>
          <h1>Choose a scenario</h1>
          <p className="note" style={{ maxWidth: 640, marginTop: 8 }}>
            Run a chemical producer's supply chain on the continent of Pangea: build factories and warehouses, set production
            and shipping, choose which markets to serve, and finish with as much cash as possible.
          </p>
        </div>
        {error && <div className="error">{error}</div>}
        <div className="cards">
          {scenarios.map((s) => (
            <button key={s.id} className="card" aria-pressed={chosen === s.id} onClick={() => setChosen(s.id)}>
              <h3>{s.name}</h3>
              <span className="note">{s.description}</span>
              <span className="small muted mono">{s.regions} regions · days {s.start_day} to {s.end_day}</span>
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
          <label className="field">
            Random seed (order timing)
            <input className="input" style={{ width: 120 }} inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value)} />
          </label>
          <button className="btn primary" disabled={!chosen || busy} onClick={start}>Start run</button>
        </div>
      </div>
    </div>
  )
}
