import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type DayRecord, type Decision, type LoggedDecision, type RunState } from './api'
import { money, num, pct } from './format'
import { CloseIcon, LogoMark } from './components/icons'
import MapView, { type Selection } from './components/MapView'
import StartScreen from './components/StartScreen'
import Timeline from './components/Timeline'
import Designer from './designer/Designer'
import { kindMeta } from './designer/eventKinds'
import FactoryPanel from './components/panels/FactoryPanel'
import HQPanel from './components/panels/HQPanel'
import { CashPanel, HistoryPanel, OverviewPanel } from './components/panels/InfoPanels'
import RegionPanel from './components/panels/RegionPanel'
import WarehousePanel from './components/panels/WarehousePanel'
import ProductAffinityPage from './models/affinity/ProductAffinityPage'
import ModelsHome from './models/ModelsHome'
import SkuSegmentationPage from './models/sku/SkuSegmentationPage'

type View = 'map' | 'history' | 'cash'

type Screen = { kind: 'start' } | { kind: 'design'; id: string; taken: string[] } | { kind: 'play'; run: RunState }

function useHash() {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

export default function App() {
  const hash = useHash()
  const [screen, setScreen] = useState<Screen>({ kind: 'start' })
  const toStart = () => setScreen({ kind: 'start' })
  const play = (run: RunState) => setScreen({ kind: 'play', run })
  const inModels = hash.startsWith('#/models')
  if (!inModels && screen.kind === 'play') {
    return <div className="app"><Game key={screen.run.run_id} initial={screen.run} onExit={toStart} /></div>
  }
  let body: React.ReactNode
  if (hash.startsWith('#/models/sku-segmentation')) body = <SkuSegmentationPage />
  else if (hash.startsWith('#/models/product-affinity')) body = <ProductAffinityPage />
  else if (inModels) body = <ModelsHome />
  else if (screen.kind === 'design') body = <Designer key={screen.id} scenarioId={screen.id} takenIds={screen.taken} onBack={toStart} onPlay={play} />
  else body = <StartScreen onStart={play} onDesign={(id, taken) => setScreen({ kind: 'design', id, taken })} />
  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#/" onClick={toStart}><LogoMark />Supply Chain Studio</a>
        <nav className="topnav" aria-label="Sections">
          <a className="navlink" href="#/" aria-current={!inModels ? 'page' : undefined}>Simulation</a>
          <a className="navlink" href="#/models" aria-current={inModels ? 'page' : undefined}>Models</a>
        </nav>
      </header>
      <div className={inModels ? 'page-scroll' : 'page-fill'}>{body}</div>
    </div>
  )
}

function Game({ initial, onExit }: { initial: RunState; onExit: () => void }) {
  const [state, setState] = useState(initial)
  const [records, setRecords] = useState<DayRecord[]>([])
  const [log, setLog] = useState<LoggedDecision[]>([])
  const [selection, setSelection] = useState<Selection>(null)
  const [view, setView] = useState<View>('map')
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(5)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const runId = initial.run_id
  const lastDay = useRef(initial.scenario.timeline.start_day - 1)

  const absorb = useCallback(async (next: RunState) => {
    setState(next)
    if (next.last_record && next.last_record.day > lastDay.current) {
      const fresh = await api.records(runId, lastDay.current + 1)
      if (fresh.length) {
        lastDay.current = fresh[fresh.length - 1].day
        setRecords((rs) => [...rs, ...fresh])
      }
    }
    return next
  }, [runId])

  const act = useCallback(async (d: Decision | Decision[]) => {
    setError('')
    try {
      for (const decision of Array.isArray(d) ? d : [d]) await absorb(await api.decide(runId, decision))
      setLog(await api.log(runId))
      return true
    } catch (e) {
      setError((e as Error).message)
      return false
    }
  }, [absorb, runId])

  const step = useCallback(async (days: number) => absorb(await api.step(runId, days)), [absorb, runId])

  // Play loop: advance in small chunks at the chosen pace until paused or finished.
  useEffect(() => {
    if (!playing) return
    let cancelled = false
    ;(async () => {
      while (!cancelled) {
        const t0 = performance.now()
        const days = Math.max(1, Math.round(speed / 5))
        let next: RunState
        try {
          next = await step(days)
        } catch (e) {
          setError((e as Error).message)
          setPlaying(false)
          return
        }
        if (next.finished) { setPlaying(false); return }
        const wait = (1000 * days) / speed - (performance.now() - t0)
        if (wait > 0) await new Promise((r) => setTimeout(r, wait))
      }
    })()
    return () => { cancelled = true }
  }, [playing, speed, step])

  const runTo = async (day: number) => {
    setBusy(true)
    try { await absorb(await api.runTo(runId, day)) } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }

  const select = (s: Selection) => { setSelection(s); if (s) setView('map') }
  const s = state.summary
  const decisionDays = [...new Set(log.map((d) => d.day))]

  let panel: React.ReactNode
  if (view === 'history') panel = <HistoryPanel state={state} log={log} />
  else if (view === 'cash') panel = <CashPanel state={state} />
  else if (selection?.kind === 'hq') panel = <HQPanel state={state} records={records} act={act} />
  else if (selection?.kind === 'factory' && state.factories.some((f) => f.region === selection.region))
    panel = <FactoryPanel key={selection.region} state={state} records={records} act={act} region={selection.region} />
  else if (selection?.kind === 'warehouse' && state.warehouses.some((w) => w.region === selection.region))
    panel = <WarehousePanel key={selection.region} state={state} records={records} act={act} region={selection.region} />
  else if (selection?.kind === 'region')
    panel = <RegionPanel key={selection.region} state={state} act={act} region={selection.region} onSelect={select} />
  else panel = <OverviewPanel state={state} records={records} onSelect={select} />

  return (
    <>
      <header className="topbar">
        <div className="brand"><LogoMark />Supply Chain Studio <small>{state.scenario.name}</small></div>
        <nav className="topnav" aria-label="Views">
          {([['map', 'Network'], ['history', 'History'], ['cash', 'Cash']] as [View, string][]).map(([v, label]) => (
            <button key={v} aria-pressed={view === v} onClick={() => setView(v)}>{label}</button>
          ))}
        </nav>
        <div className="spacer" />
        <div className="kpis">
          <div className="kpi"><span>DAY</span><span>{state.finished ? state.scenario.timeline.end_day : state.day}</span></div>
          <div className="kpi"><span>CASH</span><span>{money(state.cash)}</span></div>
          <div className="kpi"><span>FILL RATE</span><span>{s.demand ? pct(s.fill_rate) : 'n/a'}</span></div>
          <div className="kpi"><span>LOST DRUMS</span><span>{num(s.lost)}</span></div>
        </div>
        <div className="topnav">
          <a className="navlink" href="#/models" target="_blank" rel="noreferrer">Models</a>
          <button onClick={() => { setPlaying(false); onExit() }}>New run</button>
        </div>
      </header>
      <main className="main">
        <section className="stage" aria-label="Network map">
          <div className="map-legend" aria-label="Legend">
            <span><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="3" fill="#fff" stroke="#1e1b2e" strokeWidth="1.6" /></svg>Factory</span>
            <span><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="3" fill="#4338ca" stroke="#1e1b2e" strokeWidth="1.6" /></svg>Warehouse (stock on hand)</span>
            <span><svg width="14" height="14" aria-hidden="true"><rect x="1" y="1" width="12" height="12" rx="3" fill="#fff" stroke="#1e1b2e" strokeWidth="1.6" strokeDasharray="3 2" /></svg>Under construction</span>
            <span><svg width="20" height="10" aria-hidden="true"><rect x="2" y="1" width="16" height="8" rx="2" fill="#fff" stroke="#1e1b2e" strokeWidth="1.5" /></svg>Truck</span>
            <span><svg width="16" height="10" aria-hidden="true"><rect x="1" y="1" width="14" height="8" rx="1" fill="#ead77a" stroke="#1e1b2e" strokeWidth="1.4" /></svg>Mail</span>
          </div>
          <News state={state} />
          <MapView state={state} selection={selection} onSelect={select} />
          <Timeline state={state} playing={playing} speed={speed} busy={busy} decisionDays={decisionDays}
            onPlay={setPlaying} onSpeed={setSpeed} onStep={() => step(1).catch((e) => setError(e.message))} onRunTo={runTo} />
        </section>
        <aside className="side" aria-label="Details">
          {(selection || view !== 'map') && (
            <button className="btn small" style={{ alignSelf: 'flex-end', marginBottom: -8 }}
              onClick={() => { setSelection(null); setView('map') }} aria-label="Close panel">
              <CloseIcon size={12} /> Close
            </button>
          )}
          {error && <div className="error" role="alert">{error}</div>}
          {panel}
        </aside>
      </main>
    </>
  )
}

// Events under way, or announced and starting within 30 days.
function News({ state }: { state: RunState }) {
  const now = state.day
  const items = state.events
    .filter((e) => e.active || (e.start_day > now && e.start_day - now <= 30))
    .sort((a, b) => a.start_day - b.start_day)
  if (!items.length) return null
  return (
    <div className="news" role="status" aria-live="polite">
      <b>{items.some((e) => e.active) ? 'News' : 'Coming up'}</b>
      <span>
        {items.map((e, i) => (
          <span key={e.id}>
            {i > 0 && ' · '}
            <i className="swatch" style={{ background: kindMeta(e.kind).color, width: 10, height: 10, borderRadius: 3, marginRight: 5 }} />
            {e.name}{e.active ? '' : ` on day ${e.start_day}`}
            {e.description ? `: ${e.description}` : ''}
          </span>
        ))}
      </span>
    </div>
  )
}
