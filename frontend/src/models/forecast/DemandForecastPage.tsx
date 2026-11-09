import { useEffect, useState } from 'react'
import LineChart, { type Series } from '../../components/LineChart'
import { diverging } from '../charts/colors'
import Fan, { type FanBand } from '../charts/Fan'
import { prefersReducedMotion } from '../motion'
import { modelFile } from '../types'
import { Bars, Drawer, Numbers, Screen, Years } from '../ui'
import '../showcase.css'
import { COPY as C, LEVELS, METHOD_RULE, METHODS, PLANS, UPPER, type Level, type Method } from './copy'

const MODEL = 'demand-forecast'
const SCREENS = ['top', 'sales', 'split', 'methods', 'validation', 'forecast', 'wape', 'truth', 'results']
const ACCENT = '#4338ca'
const INK = '#1e1b2e'
const SETS_TEN = ['Training set', 'Training set', 'Training set', 'Validation set', 'Validation set', 'Test set', 'Test set', 'Test set', 'Test set', 'Test set']
const LINE_STYLE: Record<Method, { color: string; dash?: string }> = {
  'Seasonal naive': { color: '#8f8b9e', dash: '2 4' },
  ETS: { color: '#4f4b62' },
  LightGBM: { color: '#7a768c', dash: '6 4' },
  'Neural network': { color: ACCENT },
}

// ---------------------------------------------------------------- data shapes
interface Cell { wql: number; wape: number; bias: number; coverage: number; wape_mean: number; bias_mean: number }
type Table = Record<Level, Cell[]>
interface TruthCell { wape_truth_mean: number; truth_inside_our_p10_p90?: number; true_p10_p90_width_vs_ours?: number }
type TruthTable = Record<Level, TruthCell[]>
interface Capacity {
  rho: number[]; rho_game: number[]
  curves: Record<string, { expected: number[] }>
  per_region: ({ region: number } & Record<string, unknown>)[]
}
interface Overview {
  facts: { series_validation: Record<Level, number>; series_test: Record<Level, number> }
  validation: Record<Method, { setting: Record<string, unknown>; best: { width_upper: number; width_lower: number; reconciliation: string; score: number; table: Table } }>
  chosen: Method
  test: Record<Method, { table: Table; score: number }>
  truth: { test: TruthTable }
  capacity: Capacity
  second_future: { test: Record<Method, { table: Table; score: number }>; truth: TruthTable; capacity: Capacity; world: { rules: { launches_per_year: number; growth_stop_years: number[] } } }
}
type Rows = number[][] // P10, P50, P90 (and the mean) by month
interface Preset { level: Level; region: number; category: number; sku_id: number; actual: number[]; validation: Record<Method, Rows> }
interface RegionSeries { level: Level; region: number; actual: number[]; test: Rows; truth_test: Rows; actual_second_future: number[]; truth_second_future: Rows }
interface SeriesData { presets: Preset[]; regions: RegionSeries[] }
interface Labels { regions: Record<string, string>; categories: string[]; skus: Record<string, string> }

// ---------------------------------------------------------------- formats
const dec = (v: number, d = 0) => v.toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d })
const signed = (v: number, d = 1) => `${v < 0 ? '−' : '+'}${dec(Math.abs(v) * 100, d)}%`
const yearsWord = (n: number) => ['no forecast years', 'one forecast year', 'two forecast years', 'three forecast years', 'four forecast years', 'all five forecast years'][n]
const band = (rows: Rows, from: number): FanBand => ({ from, lo: rows[0], mid: rows[1], hi: rows[2] })

function presetName(p: Preset, L: Labels) {
  const parts: string[] = []
  if (p.region > 0) parts.push(L.regions[String(p.region)])
  if (p.category >= 0 && p.sku_id < 0) parts.push(L.categories[p.category])
  if (p.sku_id >= 0) parts.push(L.skus[String(p.sku_id)])
  return parts.join(', ')
}
const seriesName = (r: RegionSeries, L: Labels) => (r.region > 0 ? L.regions[String(r.region)] : C.total)

function Seg<T extends string | number>({ label, options, value, onChange, name }: {
  label: string; options: readonly T[]; value: T; onChange: (v: T) => void; name: (v: T) => string
}) {
  return (
    <div className="sc-control">{label}
      <div className="sc-seg" role="group" aria-label={label}>
        {options.map((x) => <button key={String(x)} aria-pressed={value === x} onClick={() => onChange(x)}>{name(x)}</button>)}
      </div>
    </div>
  )
}

// The key under every fan, swatches drawn as in the chart.
function FanKey({ truth = false }: { truth?: boolean }) {
  return (
    <div className="sc-key" aria-hidden="true">
      <span><i className="ln" style={{ background: '#a8a4b5' }} />{C.key.history}</span>
      <span><i className="ln" style={{ background: INK }} />{C.key.actual}</span>
      <span><i className="ln" style={{ background: ACCENT }} />{C.key.p50}</span>
      <span><i style={{ background: ACCENT, opacity: 0.14 }} />{truth ? C.truth.ours : C.key.band}</span>
      {truth && <span><i style={{ background: INK, opacity: 0.38 }} />{C.truth.true}</span>}
    </div>
  )
}

// ---------------------------------------------------------------- page
export default function DemandForecastPage() {
  const [o, setO] = useState<Overview | null>(null)
  const [s, setS] = useState<SeriesData | null>(null)
  const [L, setL] = useState<Labels | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState('top')

  useEffect(() => {
    Promise.all([modelFile<Overview>(MODEL, 'overview.json'), modelFile<SeriesData>(MODEL, 'series.json'), modelFile<Labels>(MODEL, 'labels.json')])
      .then(([a, b, c]) => { setO(a); setS(b); setL(c) })
      .catch((e) => setError(String(e)))
  }, [])

  const ready = !!(o && s && L)
  useEffect(() => {
    if (!ready) return
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) setActive(e.target.id) }), { rootMargin: '-45% 0px -45% 0px' })
    SCREENS.forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [ready])

  if (error) return <div className="showcase"><div className="sc-page"><p className="sc-caption" style={{ paddingTop: 40 }}>{C.failed(error)}</p></div></div>
  if (!o || !s || !L) return <div className="showcase"><div className="sc-page"><p className="sc-caption" style={{ paddingTop: 40 }}>{C.loading}</p></div></div>

  const total = Object.values(o.facts.series_validation).reduce((a, b) => a + b, 0)
  return (
    <div className="showcase">
      <nav className="sc-dots" aria-label="Screens">
        {SCREENS.map((id) => <a key={id} href={`#/models/${MODEL}`} className={active === id ? 'on' : ''} aria-label={id}
          onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' }) }} />)}
      </nav>
      <div className="sc-page">
        <a className="sc-back" href="#/models">{C.back}</a>
        <section id="top" className="sc-hero cover">
          <h1>{C.title}</h1>
          <p className="sc-sub">{C.question}</p>
          <Numbers items={[
            { value: dec(total), label: C.opening.series },
            { value: dec(LEVELS.length), label: C.opening.levels },
            { value: '5', label: C.opening.years },
          ]} />
        </section>
        <SalesScreen o={o} s={s} L={L} />
        <SplitScreen o={o} />
        <MethodsScreen o={o} s={s} />
        <ValidationScreen o={o} />
        <ForecastScreen s={s} L={L} />
        <WapeScreen o={o} />
        <TruthScreen o={o} s={s} L={L} />
        <ResultsScreen o={o} L={L} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- 1. monthly sales
function SalesScreen({ o, s, L }: { o: Overview; s: SeriesData; L: Labels }) {
  return (
    <Screen id="sales" headline={C.sales.headline} caption={C.sales.caption}
      drawer={<Drawer>
        <p>{C.sales.drawer.levels}</p>
        <div className="table-wrap" style={{ maxWidth: 420 }}>
          <table className="grid">
            <thead><tr><th>{C.sales.drawer.colLevel}</th><th className="num">{C.sales.drawer.colSeries}</th></tr></thead>
            <tbody>{LEVELS.map((lv) => <tr key={lv}><td>{lv}</td><td className="num">{dec(o.facts.series_validation[lv])}</td></tr>)}</tbody>
          </table>
        </div>
        <p>{C.sales.drawer.pairs(dec(o.facts.series_validation['Region × SKU']), dec(o.facts.series_test['Region × SKU']))}</p>
        <p>{C.sales.drawer.units}</p>
      </Drawer>}>
      <div className="sc-grid fixed-3">
        {s.presets.map((p) => (
          <div key={p.level}>
            <div className="sc-card-title">{p.level}</div>
            <p className="sc-card-text">{p.level === 'Total' ? '\u00a0' : presetName(p, L)}</p>
            <Fan compact height={150} title={C.sales.chartTitle(p.level)} actual={p.actual.slice(0, 60)} format={(v) => dec(v)} />
          </div>
        ))}
      </div>
    </Screen>
  )
}

// ---------------------------------------------------------------- 2. data split
function SplitScreen({ o }: { o: Overview }) {
  const r = o.second_future.world.rules
  return (
    <Screen id="split" headline={C.split.headline}
      drawer={<Drawer>
        <p>{C.split.drawer.validation}</p>
        <p>{C.split.drawer.future(dec(r.launches_per_year, 1), dec(r.growth_stop_years[0]), dec(r.growth_stop_years[1]))}</p>
        <p>{C.split.drawer.changes}</p>
      </Drawer>}>
      <Years sets={SETS_TEN} />
    </Screen>
  )
}

// ---------------------------------------------------------------- 3. methods
const settingText = (x: Record<string, unknown>) => Object.entries(x).map(([k, v]) => `${k.replace(/_/g, ' ')} ${String(v).replace(/_/g, ' ')}`).join(', ')

function MethodsScreen({ o, s }: { o: Overview; s: SeriesData }) {
  const [method, setMethod] = useState<Method>(o.chosen)
  const [level, setLevel] = useState<Level>('Region')
  const p = s.presets.find((x) => x.level === level)!
  return (
    <Screen id="methods" headline={C.methods.headline} caption={C.methods.caption}
      drawer={<Drawer>
        <p>{C.methods.drawer.read}</p>
        <p>{C.methods.drawer.ets}</p>
        <p>{C.methods.drawer.learned(settingText(o.validation.LightGBM.setting), settingText(o.validation['Neural network'].setting))}</p>
        <p>{C.methods.drawer.fallback}</p>
        <p>{C.methods.drawer.paths}</p>
        <p>{C.methods.drawer.reconcile}</p>
      </Drawer>}>
      <div className="sc-split">
        <table className="sc-rules">
          <thead><tr><th>{C.methods.colMethod}</th><th>{C.methods.colRule}</th></tr></thead>
          <tbody>{METHODS.map((m) => <tr key={m}><th scope="row">{m}</th><td>{METHOD_RULE[m]}</td></tr>)}</tbody>
        </table>
        <div>
          <div className="sc-controls">
            <Seg label={C.methods.method} options={METHODS} value={method} onChange={setMethod} name={String} />
            <Seg label={C.level} options={LEVELS} value={level} onChange={setLevel} name={String} />
          </div>
          <Fan title={C.methods.chartTitle(method, level)} actual={p.actual.slice(0, 60)} band={band(p.validation[method], 37)} height={300} width={640}
            yLabel={C.forecast.yTitle} format={(v) => dec(v)} />
          <FanKey />
        </div>
      </div>
    </Screen>
  )
}

// ---------------------------------------------------------------- 4. validation
function ValidationScreen({ o }: { o: Overview }) {
  const mean = (cells: Cell[]) => cells.reduce((a, c) => a + c.wql, 0) / cells.length
  const value = (m: Method, lv: Level) => mean(o.validation[m].best.table[lv])
  const best = Object.fromEntries([...LEVELS, 'mean'].map((lv) => [lv, Math.min(...METHODS.map((m) => (lv === 'mean' ? o.validation[m].best.score : value(m, lv as Level))))]))
  const cell = (v: number, b: number) => {
    const excess = v / b - 1
    if (excess < 0.1) return undefined
    const t = 0.4 + 0.6 * Math.min((excess - 0.1) / 0.9, 1)
    return { background: diverging(t), color: t > 0.8 ? '#ffffff' : INK }
  }
  return (
    <Screen id="validation" headline={C.validation.headline} caption={C.validation.caption}
      drawer={<Drawer>
        <p>{C.validation.drawer.loss}</p>
        <p>{C.validation.drawer.colour}</p>
        <p>{C.validation.drawer.chosen(o.chosen)}</p>
        <p>{C.validation.drawer.horizon}</p>
        <p>{C.validation.drawer.width}</p>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>{C.validation.colMethod}</th><th>{C.validation.drawer.colSetting}</th><th className="num">{C.validation.drawer.colWidth}</th><th>{C.validation.drawer.colRec}</th></tr></thead>
            <tbody>{METHODS.map((m) => {
              const b = o.validation[m].best
              return <tr key={m}><td>{m}</td><td>{settingText(o.validation[m].setting) || C.validation.drawer.none}</td><td className="num">{dec(b.width_upper, 2)}, {dec(b.width_lower, 2)}</td><td>{b.reconciliation}</td></tr>
            })}</tbody>
          </table>
        </div>
      </Drawer>}>
      <div className="sc-fv-wrap">
        <table className="sc-fv sc-val">
          <thead><tr><th>{C.validation.colMethod}</th>{LEVELS.map((lv) => <th key={lv}>{lv}</th>)}<th>{C.validation.colMean}</th></tr></thead>
          <tbody>{METHODS.map((m) => (
            <tr key={m} className={m === o.chosen ? 'chosen' : undefined}>
              <th scope="row">{m}</th>
              {LEVELS.map((lv) => { const v = value(m, lv); return <td key={lv} className="num" style={cell(v, best[lv])}>{dec(v * 100, 1)}</td> })}
              <td className="num" style={cell(o.validation[m].best.score, best.mean)}>{dec(o.validation[m].best.score * 100, 1)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
    </Screen>
  )
}

// ---------------------------------------------------------------- 5. five-year forecast
// The page shows one draw of the test years (the model's "second future"); the first draw is disclosed
// in the data-split, WAPE and results drawers.
const actual10 = (r: RegionSeries) => [...r.actual.slice(0, 60), ...r.actual_second_future]

function ForecastScreen({ s, L }: { s: SeriesData; L: Labels }) {
  const [i, setI] = useState(0)
  const r = s.regions[i]
  return (
    <Screen id="forecast" headline={C.forecast.headline}
      drawer={<Drawer><p>{C.forecast.drawer.read}</p><p>{C.forecast.drawer.growth}</p></Drawer>}>
      <div className="sc-controls">
        <Seg label={C.series} options={s.regions.map((_, k) => k)} value={i} onChange={setI} name={(k) => seriesName(s.regions[k], L)} />
      </div>
      <Fan title={C.forecast.chartTitle(seriesName(r, L))} actual={actual10(r)} band={band(r.test, 61)} yLabel={C.forecast.yTitle} format={(v) => dec(v)} />
      <FanKey />
    </Screen>
  )
}

// ---------------------------------------------------------------- 6. WAPE by forecast year
function leadYears(t: Record<Method, { table: Table }>) {
  const etsRows = t.ETS.table.Total
  let n = 0
  for (let j = 0; j < etsRows.length; j++) {
    const learned = Math.min(t.LightGBM.table.Total[j].wape_mean, t['Neural network'].table.Total[j].wape_mean)
    if (learned < etsRows[j].wape_mean) n++
    else break
  }
  return n
}

function WapeScreen({ o }: { o: Overview }) {
  const [level, setLevel] = useState<Level>('Total')
  const t = o.second_future.test
  const truth = o.second_future.truth
  const series: Series[] = [
    ...METHODS.map((m) => ({ id: m, label: m, color: LINE_STYLE[m].color, dash: LINE_STYLE[m].dash, values: t[m].table[level].map((c) => c.wape_mean * 100) })),
    { id: 'truth', label: C.wape.truth, color: INK, dash: '1 3', values: truth[level].map((c) => c.wape_truth_mean * 100) },
  ]
  const learned: Method[] = ['LightGBM', 'Neural network']
  const bestLearned = learned.reduce((a, b) => (t[b].score < t[a].score ? b : a))
  const scores = METHODS.filter((m) => m !== 'Seasonal naive').map((m) => `${m} ${dec(t[m].score * 100, 1)}%`).join(', ')
  return (
    <Screen id="wape" headline={C.wape.headline}
      drawer={<Drawer>
        <p>{C.wape.drawer.lead(yearsWord(leadYears(t)))}</p>
        <p>{C.wape.drawer.unseen}</p>
        <p>{bestLearned === o.chosen ? C.wape.drawer.overallSame(scores) : C.wape.drawer.overall(scores, bestLearned)}</p>
        <p>{C.wape.drawer.firstDraw(yearsWord(leadYears(o.test)))}</p>
        <p>{C.wape.drawer.truthLine}</p>
        <h4>{C.wape.drawer.skuTitle}</h4>
        <div className="table-wrap" style={{ maxWidth: 560 }}>
          <table className="grid">
            <thead><tr><th>{C.wape.drawer.colMethod}</th>{(['SKU', 'Region × SKU'] as Level[]).flatMap((lv) => [1, 2].map((y) => <th key={lv + y} className="num">{lv}, {C.wape.drawer.colYear(y)}</th>))}</tr></thead>
            <tbody>{METHODS.map((m) => <tr key={m}><td>{m}</td>{(['SKU', 'Region × SKU'] as Level[]).flatMap((lv) => t[m].table[lv].map((c, j) => <td key={lv + j} className="num">{dec(c.wape_mean * 100, 1)}</td>))}</tr>)}</tbody>
          </table>
        </div>
      </Drawer>}>
      <div className="sc-controls">
        <Seg label={C.level} options={UPPER} value={level} onChange={setLevel} name={String} />
      </div>
      <LineChart title={C.wape.chartTitle(level)} x={[1, 2, 3, 4, 5]} xLabel={C.wape.xTitle} yLabel={C.wape.yTitle} series={series} height={320}
        format={(v) => dec(v)} hideTitle hideToggle hideLegend endLabels endGroup={0.3} />
    </Screen>
  )
}

// ---------------------------------------------------------------- 7. forecast versus ground truth
function TruthScreen({ o, s, L }: { o: Overview; s: SeriesData; L: Labels }) {
  const [i, setI] = useState(0)
  const r = s.regions[i]
  const lv: Level = r.region > 0 ? 'Region' : 'Total'
  const rows = o.second_future.truth[lv]
  return (
    <Screen id="truth" headline={C.truth.headline}
      drawer={<Drawer>
        <p>{C.truth.drawer.reps}</p>
        <h4>{C.truth.drawer.tableTitle(lv === 'Total' ? C.truth.drawer.total : C.truth.drawer.regions)}</h4>
        <div className="table-wrap" style={{ maxWidth: 720 }}>
          <table className="grid">
            <thead><tr><th>{C.truth.drawer.colYear}</th><th className="num">{C.truth.drawer.colInside}</th><th className="num">{C.truth.drawer.colWidth}</th></tr></thead>
            <tbody>{rows.map((x, j) => <tr key={j}><td>{j + 1}</td><td className="num">{dec((x.truth_inside_our_p10_p90 ?? 0) * 100)}</td><td className="num">{dec(x.true_p10_p90_width_vs_ours ?? 0, 2)}</td></tr>)}</tbody>
          </table>
        </div>
        <p>{C.truth.drawer.wide(dec(1 / Math.max(...rows.map((x) => x.true_p10_p90_width_vs_ours ?? 1))), dec(1 / Math.min(...rows.map((x) => x.true_p10_p90_width_vs_ours ?? 1))))}</p>
        <p>{C.truth.drawer.check}</p>
      </Drawer>}>
      <div className="sc-controls">
        <Seg label={C.series} options={s.regions.map((_, k) => k)} value={i} onChange={setI} name={(k) => seriesName(s.regions[k], L)} />
      </div>
      <Fan title={C.truth.chartTitle(seriesName(r, L))} actual={actual10(r)} band={band(r.test, 61)}
        reference={band(r.truth_second_future, 61)} start={49} yLabel={C.forecast.yTitle} format={(v) => dec(v)} />
      <FanKey truth />
    </Screen>
  )
}

// ---------------------------------------------------------------- 8. results
function atGame(c: Capacity) {
  const target = c.rho_game.reduce((a, b) => a + b, 0) / c.rho_game.length
  return c.rho.reduce((best, r, k) => (Math.abs(r - target) < Math.abs(c.rho[best] - target) ? k : best), 0)
}
const over = (c: Capacity, key: string, k: number) => c.curves[key].expected[k] / c.curves['Ground truth'].expected[k] - 1
const cheapest = (c: Capacity) => {
  const k = atGame(c)
  return PLANS.reduce((a, b) => (over(c, b.key, k) < over(c, a.key, k) ? b : a))
}

function ResultsScreen({ o, L }: { o: Overview; L: Labels }) {
  const cap = o.second_future.capacity
  const k = atGame(cap)
  const ratios = [0.25, 0.5, 1, 2, 4].map((t) => cap.rho.reduce((b, r, j) => (Math.abs(r - t) < Math.abs(cap.rho[b] - t) ? j : b), 0))
  const cols = [...ratios.slice(0, 2), k, ...ratios.slice(2)]
  const regionPlans = PLANS.filter((p) => cap.per_region[0]?.[p.key])
  const first = cheapest(o.capacity)
  const k1 = atGame(o.capacity)
  return (
    <Screen id="results" headline={C.results.headline} caption={C.results.caption}
      drawer={<Drawer>
        <p>{C.results.drawer.decision}</p>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>{C.results.drawer.colPlan}</th><th>{C.results.drawer.colRule}</th></tr></thead>
            <tbody>{PLANS.map((p) => <tr key={p.key}><td style={{ whiteSpace: 'nowrap' }}>{p.label}</td><td>{p.rule}</td></tr>)}</tbody>
          </table>
        </div>
        <p>{C.results.drawer.ranking(cheapest(cap).label)}</p>
        <p>{C.results.drawer.ours}</p>
        <p>{C.results.drawer.firstDraw(first.label.toLowerCase(), signed(over(o.capacity, 'Our forecast', k1)))}</p>
        <p>{C.results.drawer.flatters}</p>
        <h4>{C.results.drawer.ratioTitle}</h4>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>{C.results.drawer.colPlan}</th>{cols.map((j) => <th key={j} className="num">{dec(cap.rho[j], 2)}{j === k ? ` (${C.results.drawer.game})` : ''}</th>)}</tr></thead>
            <tbody>{PLANS.map((p) => <tr key={p.key}><td style={{ whiteSpace: 'nowrap' }}>{p.label}</td>{cols.map((j) => <td key={j} className="num">{signed(over(cap, p.key, j))}</td>)}</tr>)}</tbody>
          </table>
        </div>
        <h4>{C.results.drawer.regionTitle}</h4>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>{C.results.drawer.colRegion}</th>{regionPlans.map((p) => <th key={p.key} className="num">{p.label}</th>)}</tr></thead>
            <tbody>{cap.per_region.map((row) => <tr key={row.region}><td>{L.regions[String(row.region)]}</td>{regionPlans.map((p) => <td key={p.key} className="num">{dec((row[p.key] as { capacity: number }).capacity)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </Drawer>}>
      <Bars format={(v) => signed(v).replace('%', '')} rows={PLANS.map((p) => ({ label: p.label, value: over(cap, p.key, k), ours: p.key === 'Our forecast' }))} />
    </Screen>
  )
}
