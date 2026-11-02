import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import LineChart, { type Series } from '../../components/LineChart'
import { EXTENDED } from '../charts/colors'
import Formula from '../charts/Formula'
import PairMatrix from '../charts/PairMatrix'
import Scatter, { type ScatterNote } from '../charts/Scatter'
import { prefersReducedMotion } from '../motion'
import Projector from '../sku/Projector'
import { modelFile } from '../types'
import { Bars, Drawer, Numbers, Screen, Years } from '../ui'
import '../showcase.css'
import { COPY as C, RULE_HOW, RULE_SIMPLE, RULES, RULES_IN_PROSE, type RuleKey } from './copy'
import FcColumns from './FcColumns'
import type { AffinityGraph, AffinityMatrix, AffinityNode, AffinityOverview, PriceRow } from './types'

const MODEL = 'product-affinity'
const SCREENS = ['top', 'data', 'counts', 'chance', 'network', 'truth', 'assign', 'results']
const SCREEN_NAMES: Record<string, string> = {
  top: C.title, data: C.data.headline, counts: C.counts.headline, chance: C.chance.headline,
  network: C.network.headline, truth: C.truth.headline, assign: C.assign.headline, results: C.results.headline,
}
const ACCENT = '#4338ca'
// Without the violet (too close to the indigo accent) and the slate grey (grey means unlinked).
const PALETTE = EXTENDED.filter((c) => c !== '#6250d6' && c !== '#6b7280')
const UNLINKED = '#d6d4de'
const OTHER_PAIRS = 'rgba(122, 118, 140, 0.3)'
const RULE_ORDER: RuleKey[] = ['random', 'category', 'popularity', 'best_sellers', 'communities', 'hidden_bundles']
const PLAIN_RULES: RuleKey[] = ['random', 'category', 'popularity', 'best_sellers']

// Number formats (en-GB grouping, tabular figures come from the CSS).
const dec = (v: number, d = 0) => v.toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d })
const pc = (v: number, d = 0) => `${dec(v * 100, d)}%`
const level = (v: number) => `${+(v * 100).toPrecision(3)}%`
const usd = (v: number) => `$${dec(v)}`
const usdCents = (v: number) => `$${dec(v, 2)}`
const change = (r: number, d: number) => {
  const zero = Number((Math.abs(r) * 100).toFixed(d)) === 0
  return `${zero ? '' : r < 0 ? '−' : '+'}${dec(Math.abs(r) * 100, d)}%`
}
const joinList = (xs: string[]) => (xs.length < 2 ? xs.join('') : C.list(xs.slice(0, -1).join(', '), xs[xs.length - 1]))
const steps = (max: number) => {
  const out: number[] = []
  for (let e = 0; 10 ** e < max; e++) for (const k of [1, 2, 5]) if (k * 10 ** e < max) out.push(k * 10 ** e)
  return [...out, max]
}

/** Text with inline \( ... \) and block \[ ... \] formulas. */
function Rich({ text }: { text: string }) {
  const parts = text.split(/\\\((.+?)\\\)|\\\[(.+?)\\\]/s)
  const out: ReactNode[] = []
  for (let i = 0; i < parts.length; i += 3) {
    if (parts[i]) out.push(parts[i])
    if (parts[i + 1] != null) out.push(<Formula key={i} tex={parts[i + 1]} />)
    if (parts[i + 2] != null) out.push(<Formula key={i} tex={parts[i + 2]} block />)
  }
  return <>{out}</>
}
const P = ({ text }: { text: string }) => <p><Rich text={text} /></p>

// =================================================================================================
// Everything derived from the graph once: orderings, community colours, partners.

type Derived = ReturnType<typeof derive>

function derive(g: AffinityGraph) {
  const nodes = g.nodes
  const bySales = nodes.map((_, i) => i).sort((a, b) => nodes[b].baskets - nodes[a].baskets || a - b)
  const comms = [...g.communities].sort((a, b) => b.size - a.size || a.id - b.id)
  const members = new Map<number, number[]>()
  for (const i of bySales) {
    if (nodes[i].alone) continue
    const list = members.get(nodes[i].community) ?? []
    list.push(i)
    members.set(nodes[i].community, list)
  }
  const unlinked = bySales.filter((i) => nodes[i].alone)
  const byCommunity = [...comms.flatMap((c) => members.get(c.id) ?? []), ...unlinked]
  const commName = new Map(comms.map((c) => [c.id, c.name]))

  // Community colours, largest first: each takes the colour whose nearest community already using it
  // is furthest away in the layout, so communities placed near each other get different colours.
  const centroid = (ids: number[]) => [0, 1, 2].map((k) => ids.reduce((s, i) => s + nodes[i].xyz[k], 0) / Math.max(ids.length, 1))
  const placed: { p: number[]; k: number }[] = []
  const commColour = new Map<number, string>()
  for (const c of comms) {
    const p = centroid(members.get(c.id) ?? [])
    let best = 0, bestD = -1
    PALETTE.forEach((_, k) => {
      let dist = Infinity
      for (const q of placed) if (q.k === k) dist = Math.min(dist, Math.hypot(q.p[0] - p[0], q.p[1] - p[1], q.p[2] - p[2]))
      if (dist > bestD) { bestD = dist; best = k }
    })
    placed.push({ p, k: best })
    commColour.set(c.id, PALETTE[best])
  }
  const colour = (i: number) => (nodes[i].alone ? UNLINKED : commColour.get(nodes[i].community) ?? UNLINKED)
  const categories = [...new Set(nodes.map((x) => x.category))]
  const catColour = new Map(categories.map((c, k) => [c, PALETTE[k % PALETTE.length]]))

  const partners = new Map<number, { j: number; lift: number }[]>()
  const link = (i: number, j: number, lift: number) => { const l = partners.get(i) ?? []; l.push({ j, lift }); partners.set(i, l) }
  for (const e of g.edges) { link(e.a, e.b, e.lift); link(e.b, e.a, e.lift) }

  return { nodes, bySales, comms, members, unlinked, byCommunity, commName, commColour, colour, categories, catColour, partners }
}

// Everything derived from the pair matrix once: the scatter, the lift checks and the top pairs.
type PairStats = NonNullable<ReturnType<typeof pairStats>>

function pairStats(m: AffinityMatrix | null, g: AffinityGraph | null) {
  if (!m || !g) return null
  const n = g.nodes.length
  const linked = new Set(g.edges.map((e) => e.a * n + e.b))
  const bx: number[] = [], by: number[] = []
  const seen: [number, number, number][] = []
  let big = 0, small = 0, topLift = 0, topCount = 0, maxCount = 0, maxLift = 1
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const c = m.count[i][j]
      if (c < 1) continue
      const L = m.lift[i][j]
      seen.push([i, j, c])
      maxCount = Math.max(maxCount, c)
      maxLift = Math.max(maxLift, L)
      if (L > 3) { big++; if (c < 5) small++ }
      if (!linked.has(i * n + j)) {
        bx.push(m.expected[i][j]); by.push(c)
        if (L > topLift) { topLift = L; topCount = c }
      }
    }
  }
  const fx = g.edges.map((e) => m.expected[e.a][e.b])
  const fy = g.edges.map((e) => e.count)
  const top20 = [...seen].sort((a, b) => b[2] - a[2]).slice(0, 20)

  // Share of the 200 highest-lift pairs inside a hidden bundle, for every minimum count from 3 to 10.
  const bundle = g.nodes.map((x) => x.bundle)
  const cutShares = [3, 4, 5, 6, 7, 8, 9, 10].map((cut) => {
    const top = seen.filter((p) => p[2] >= cut).sort((a, b) => m.lift[b[0]][b[1]] - m.lift[a[0]][a[1]]).slice(0, 200)
    return top.filter(([i, j]) => bundle[i] > 0 && bundle[i] === bundle[j]).length / Math.max(top.length, 1)
  })

  // Direct labels: above the left end of the linked pairs, and under the right end of the grey mass.
  const lg = Math.log10
  const q = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.floor(p * (xs.length - 1))]
  const lq = q(fx.map(lg), 0.2)
  let upper = -Infinity
  fx.forEach((x, k) => { if (lg(x) <= lq) upper = Math.max(upper, lg(fy[k])) })
  const gq = q(bx.map(lg), 0.92)
  let lower = Infinity
  bx.forEach((x, k) => { if (Math.abs(lg(x) - gq) < 0.1) lower = Math.min(lower, lg(by[k])) })
  const notes: ScatterNote[] = [
    { x: 10 ** lq, y: 10 ** (upper + 0.25), text: C.chance.linked, color: ACCENT, anchor: 'end' },
    { x: 10 ** gq, y: 10 ** (lower - 0.3), text: C.chance.other, color: '#5e5a72', anchor: 'start' },
  ]
  const allX = [...fx, ...bx], allY = [...fy, ...by]
  const domainX: [number, number] = [Math.min(...allX) / 1.6, Math.max(...allX) * 1.6]
  const domainY: [number, number] = [0.75, Math.max(...allY) * 1.6]
  return { background: { x: bx, y: by, color: OTHER_PAIRS, radius: 1.5 }, fx, fy, notes, domainX, domainY, big, small, topLift, topCount, maxCount, maxLift, top20, cutShares }
}

// =================================================================================================

export default function ProductAffinityPage() {
  const [o, setO] = useState<AffinityOverview | null>(null)
  const [g, setG] = useState<AffinityGraph | null>(null)
  const [error, setError] = useState('')
  const [near, setNear] = useState(false)
  const [m, setM] = useState<AffinityMatrix | null>(null)
  const [mFailed, setMFailed] = useState(false)
  const [active, setActive] = useState('top')
  const [fcs, setFcs] = useState(2) // shared by the assignment and results screens

  useEffect(() => {
    Promise.all([modelFile<AffinityOverview>(MODEL, 'overview.json'), modelFile<AffinityGraph>(MODEL, 'graph.json')])
      .then(([ov, gr]) => { setO(ov); setG(gr) })
      .catch((e) => setError(e.message))
  }, [])

  const ready = !!(o && g)
  useEffect(() => {
    if (!ready) return
    const io = new IntersectionObserver((es) => {
      const v = es.filter((e) => e.isIntersecting)[0]
      if (v) setActive(v.target.id)
    }, { rootMargin: '-45% 0px -45% 0px' })
    SCREENS.forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el) })
    // The pair matrix (about 1 MB) is fetched when the screens that need it come near.
    const soon = new IntersectionObserver((es) => {
      if (es.some((e) => e.isIntersecting)) { setNear(true); soon.disconnect() }
    }, { rootMargin: '150% 0px 150% 0px' })
    ;['counts', 'chance', 'network'].forEach((id) => { const el = document.getElementById(id); if (el) soon.observe(el) })
    return () => { io.disconnect(); soon.disconnect() }
  }, [ready])

  useEffect(() => {
    if (!near) return
    let live = true
    modelFile<AffinityMatrix>(MODEL, 'matrix.json').then((x) => { if (live) setM(x) }).catch(() => { if (live) setMFailed(true) })
    return () => { live = false }
  }, [near])

  const d = useMemo(() => (g ? derive(g) : null), [g])
  const ps = useMemo(() => pairStats(m, g), [m, g])

  if (error) return <div className="showcase"><div className="sc-page"><p className="sc-caption" style={{ paddingTop: 40 }}>{C.failed(error)}</p></div></div>
  if (!o || !g || !d) return <div className="showcase"><div className="sc-page"><p className="sc-caption" style={{ paddingTop: 40 }}>{C.loading}</p></div></div>

  const matrix = { m, ps, failed: mFailed }
  return (
    <div className="showcase">
      <nav className="sc-dots" aria-label="Screens">
        {SCREENS.map((id) => <a key={id} href="#/models/product-affinity" className={active === id ? 'on' : ''} aria-label={SCREEN_NAMES[id]}
          onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' }) }} />)}
      </nav>
      <div className="sc-page">
        <a className="sc-back" href="#/models">{C.back}</a>

        <section id="top" className="sc-hero cover">
          <h1>{C.title}</h1>
          <p className="sc-sub">{C.question}</p>
          <Numbers items={[
            { value: dec(g.nodes.length), label: C.opening.skus },
            { value: dec(d.categories.length), label: C.opening.categories },
            { value: dec(o.facts.pairs_possible), label: C.opening.pairs },
          ]} />
        </section>

        <DataScreen o={o} />
        <CountsScreen o={o} d={d} {...matrix} />
        <ChanceScreen g={g} {...matrix} />
        <NetworkScreen o={o} g={g} d={d} {...matrix} />
        <TruthScreen o={o} ps={ps} />
        <AssignScreen o={o} d={d} fcs={fcs} setFcs={setFcs} />
        <ResultsScreen o={o} fcs={fcs} setFcs={setFcs} />
      </div>
    </div>
  )
}

interface MatrixProps { m: AffinityMatrix | null; ps: PairStats | null; failed: boolean }

function Frame({ failed, height }: { failed: boolean; height?: number }) {
  return <div className="sc-matrix-frame" style={height ? { aspectRatio: 'auto', height } : undefined}>{failed ? C.counts.failed : C.counts.loading}</div>
}

function Seg<T extends string | number>({ label, options, value, onChange, name }: {
  label: string; options: T[]; value: T; onChange: (v: T) => void; name: (v: T) => string
}) {
  return (
    <div className="sc-control">{label}
      <div className="sc-seg" role="group" aria-label={label}>
        {options.map((x) => <button key={String(x)} aria-pressed={value === x} onClick={() => onChange(x)}>{name(x)}</button>)}
      </div>
    </div>
  )
}

// =================================================================================================
// Screen 1: the orders.

function DataScreen({ o }: { o: AffinityOverview }) {
  const counts = o.facts.basket_size_counts_all
  const total = counts.reduce((a, b) => a + b, 0)
  const multi = 1 - (counts[1] ?? 0) / total
  return (
    <Screen id="data" headline={C.data.headline}
      drawer={<Drawer>
        <p>{C.data.drawer.once}</p>
        <p>{C.data.drawer.multi(pc(multi))}</p>
        <p>{C.data.drawer.sets}</p>
      </Drawer>}>
      <Numbers items={[
        { value: dec(o.facts.orders_by_year.reduce((a, b) => a + b, 0)), label: C.data.orders },
        { value: dec(o.facts.skus_per_order_all, 1), label: C.data.skusPerOrder },
      ]} />
      <Years />
      <SizeBars counts={counts} />
    </Screen>
  )
}

function niceStep(v: number) {
  const raw = v / 4
  const p = 10 ** Math.floor(Math.log10(raw))
  for (const k of [1, 2, 2.5, 5, 10]) if (k * p >= raw) return k * p
  return 10 * p
}

function SizeBars({ counts }: { counts: number[] }) {
  const POOL = 8
  const bars = [
    ...Array.from({ length: POOL - 1 }, (_, k) => {
      const v = counts[k + 1] ?? 0
      return { label: String(k + 1), v, tip: k === 0 ? C.data.barOne(dec(v)) : C.data.barMany(dec(v), String(k + 1)) }
    }),
    (() => { const v = counts.slice(POOL).reduce((a, b) => a + b, 0); return { label: C.data.pooled(String(POOL)), v, tip: C.data.barPooled(dec(v), String(POOL)) } })(),
  ]
  const W = 760, H = 300, pad = { l: 62, r: 8, t: 12, b: 48 }
  // Orders with one SKU cannot be split: light grey. Orders with two or more: dark.
  const top = Math.max(...bars.map((b) => b.v)) / 1000
  const step = niceStep(top)
  const yMax = Math.ceil(top / step) * step
  const ticks = Array.from({ length: Math.round(yMax / step) + 1 }, (_, k) => k * step)
  const band = (W - pad.l - pad.r) / bars.length
  const py = (v: number) => H - pad.b - (v / yMax) * (H - pad.t - pad.b)
  return (
    <div className="chart sc-sizes">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={C.data.chartTitle}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={pad.l} x2={W - pad.r} y1={py(t)} y2={py(t)} />
            <text className="axis-text" x={pad.l - 8} y={py(t) + 3} textAnchor="end">{dec(t)}</text>
          </g>
        ))}
        {bars.map((b, k) => {
          const x = pad.l + k * band + band * 0.14
          return (
            <g key={b.label}>
              <rect x={x} y={py(b.v / 1000)} width={band * 0.72} height={py(0) - py(b.v / 1000)} fill={k === 0 ? '#dcdae5' : '#4f4b62'}><title>{b.tip}</title></rect>
              <text className="axis-text sc-sizes-x" x={x + band * 0.36} y={H - pad.b + 16} textAnchor="middle">{b.label}</text>
            </g>
          )
        })}
        <line x1={pad.l} x2={W - pad.r} y1={py(0)} y2={py(0)} stroke="#8a8699" />
        <text className="axis-title" x={(pad.l + W - pad.r) / 2} y={H - 8} textAnchor="middle">{C.data.xTitle}</text>
        <text className="axis-title" transform={`translate(14,${(pad.t + H - pad.b) / 2}) rotate(-90)`} textAnchor="middle">{C.data.yTitle}</text>
      </svg>
    </div>
  )
}

// =================================================================================================
// Screen 2: raw counts.

function CountsScreen({ o, d, m, ps, failed }: { o: AffinityOverview; d: Derived } & MatrixProps) {
  const value = useCallback((a: number, b: number) => (m ? m.count[a][b] : 0), [m])
  const st = useMemo(() => (ps ? steps(ps.maxCount) : [1]), [ps])
  const readout = useCallback((a: number, b: number) => (
    <><b>{C.pair(d.nodes[a].name, d.nodes[b].name)}</b><span><span className="k">{C.counts.legend}</span> {m ? dec(m.count[a][b]) : ''}</span></>
  ), [d, m])
  return (
    <Screen id="counts" headline={C.counts.headline}
      drawer={<Drawer>
        <P text={C.counts.drawer.formula} />
        <p>{C.counts.drawer.seen(dec(o.facts.pairs_seen), dec(o.facts.pairs_possible))}</p>
        <p>{C.counts.drawer.chance}</p>
        <h4>{C.counts.drawer.topTitle}</h4>
        {ps ? (
          <div className="table-wrap" style={{ maxWidth: 640 }}>
            <table className="grid">
              <thead><tr><th>{C.counts.drawer.colPair}</th><th className="num">{C.counts.drawer.colBoth}</th><th className="num">{C.counts.drawer.colEach}</th></tr></thead>
              <tbody>
                {ps.top20.map(([i, j, c]) => (
                  <tr key={`${i}-${j}`}>
                    <td>{C.pair(d.nodes[i].name, d.nodes[j].name)}</td>
                    <td className="num">{dec(c)}</td>
                    <td className="num">{C.pair(dec(d.nodes[i].baskets), dec(d.nodes[j].baskets))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="sc-caption">{failed ? C.counts.failed : C.counts.loading}</p>}
      </Drawer>}>
      {m && ps
        ? <PairMatrix order={d.bySales} value={value} steps={st} legendTitle={C.counts.legend} formatStep={dec} readout={readout} title={C.counts.matrixTitle} />
        : <Frame failed={failed} />}
    </Screen>
  )
}

// =================================================================================================
// Screen 3: counts against chance.

function ChanceScreen({ g, m, ps, failed }: { g: AffinityGraph } & MatrixProps) {
  const label = useCallback((k: number) => {
    const e = g.edges[k]
    return (
      <>
        <div><b>{C.pair(g.nodes[e.a].name, g.nodes[e.b].name)}</b></div>
        <div className="row"><span className="muted">{C.chance.both}</span><b>{dec(e.count)}</b></div>
        <div className="row"><span className="muted">{C.chance.lift}</span><b>{dec(e.lift, 1)}</b></div>
      </>
    )
  }, [g])
  const tick = useCallback((v: number) => v.toLocaleString('en-GB', { maximumSignificantDigits: 2 }), [])
  return (
    <Screen id="chance" headline={C.chance.headline}
      drawer={<Drawer>
        <p>{C.chance.drawer.strata}</p>
        <P text={C.chance.drawer.expected} />
        <P text={C.chance.drawer.lift} />
        <P text={C.chance.drawer.pvalue} />
        <p>{C.chance.drawer.bh}</p>
        {ps && <p>{C.chance.drawer.luck(dec(ps.big), dec(ps.small), dec(ps.topLift), ps.topCount === 1 ? C.chance.drawer.oneOrder : C.chance.drawer.orders(dec(ps.topCount)))}</p>}
        <p>{C.chance.drawer.level}</p>
      </Drawer>}>
      {m && ps ? (
        <Scatter title={C.chance.chartTitle} x={ps.fx} y={ps.fy} color={() => ACCENT} label={label}
          xLabel={C.chance.x} yLabel={C.chance.y} logX logY width={1040} height={540} radius={3.5}
          domainX={ps.domainX} domainY={ps.domainY} tick={tick} background={ps.background} diagonal={C.chance.diagonal} notes={ps.notes} />
      ) : <Frame failed={failed} height={540} />}
    </Screen>
  )
}

// =================================================================================================
// Screen 4: the network.

type ColourBy = 'community' | 'category'
const SPREAD = 0.05 // typical distance of a community member from its centre, in layout units
const OUTER = 0.85 // distance of the outermost community centre (the unlinked shell is at about 1)

function NetworkScreen({ o, g, d, m, ps, failed }: { o: AffinityOverview; g: AffinityGraph; d: Derived } & MatrixProps) {
  const [colourBy, setColourBy] = useState<ColourBy>('community')
  const [isolate, setIsolate] = useState<string | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [rotate, setRotate] = useState(() => !prefersReducedMotion())
  const nodes = d.nodes

  // The stored layout puts the members of a community almost on one point (they pull hard on each
  // other) and packs the communities in the middle, inside a shell of unlinked SKUs. For display,
  // community centres move outward along their own direction (the furthest to OUTER, the shell stays
  // as stored), and members spread around their centre so each SKU and each line can be seen.
  const coords = useMemo(() => {
    const out = nodes.map((x) => [...x.xyz])
    const groups = [...d.members.values()].filter((ids) => ids.length > 1)
    const centre = (ids: number[]) => [0, 1, 2].map((k) => ids.reduce((s, i) => s + nodes[i].xyz[k], 0) / ids.length)
    const rMax = Math.max(...groups.map((ids) => Math.hypot(...centre(ids))), 1e-9)
    for (const ids of groups) {
      const c = centre(ids)
      const r = Math.hypot(...c) || 1e-9
      const to = c.map((v) => (v / r) * OUTER * (r / rMax) ** 0.6)
      const rms = Math.sqrt(ids.reduce((s, i) => s + nodes[i].xyz.reduce((t, v, k) => t + (v - c[k]) ** 2, 0), 0) / ids.length) || 1e-9
      for (const i of ids) out[i] = nodes[i].xyz.map((v, k) => to[k] + ((v - c[k]) * SPREAD) / rms)
    }
    return out
  }, [nodes, d])
  const sizes = useMemo(() => {
    const sorted = nodes.map((x) => x.baskets).sort((a, b) => a - b)
    const median = sorted[Math.floor(sorted.length / 2)] || 1
    return nodes.map((x) => Math.min(2.2, Math.max(0.6, Math.sqrt(x.baskets / median))))
  }, [nodes])
  const edges = useMemo(() => {
    const ls = g.edges.map((e) => Math.log(e.lift))
    const lo = Math.min(...ls), hi = Math.max(...ls)
    return g.edges.map((e, k) => ({ a: e.a, b: e.b, strength: (ls[k] - lo) / (hi - lo || 1) }))
  }, [g])
  const view = useMemo(() => {
    if (colourBy === 'community') {
      return {
        keys: nodes.map((x) => (x.alone ? 'unlinked' : String(x.community))),
        colors: nodes.map((_, i) => d.colour(i)),
        legend: d.comms.map((c) => ({ key: String(c.id), name: c.name, color: d.commColour.get(c.id) ?? UNLINKED, n: c.size })),
      }
    }
    return {
      keys: nodes.map((x) => x.category),
      colors: nodes.map((x) => d.catColour.get(x.category) ?? UNLINKED),
      legend: d.categories.map((c) => ({ key: c, name: c, color: d.catColour.get(c) ?? UNLINKED, n: nodes.filter((x) => x.category === c).length })),
    }
  }, [colourBy, nodes, d])
  const labelFor = useMemo(() => {
    const l = isolate != null ? view.legend.find((x) => x.key === isolate) : undefined
    return l ? { [l.key]: l.name } : {}
  }, [isolate, view])
  const neighbours = useMemo(() => (selected == null ? [] : (d.partners.get(selected) ?? []).map((p) => p.j)), [selected, d])
  const tooltip = useCallback((i: number) => C.network.tip(nodes[i].name, nodes[i].category), [nodes])

  const liftValue = useCallback((a: number, b: number) => (m && m.count[a][b] > 0 && m.lift[a][b] > 1 ? m.lift[a][b] : 0), [m])
  const liftSteps = useMemo(() => (ps ? steps(Math.ceil(ps.maxLift)) : [1]), [ps])
  const liftReadout = useCallback((a: number, b: number) => (
    <><b>{C.pair(nodes[a].name, nodes[b].name)}</b><span><span className="k">{C.network.drawer.legend}</span> {m ? dec(m.lift[a][b], 1) : ''}</span></>
  ), [nodes, m])

  const sizesOf = d.comms.map((c) => c.size)
  const cross = g.edges.filter((e) => nodes[e.a].category !== nodes[e.b].category).length
  const setColour = (x: ColourBy) => { setColourBy(x); setIsolate(null) }

  return (
    <Screen id="network" headline={C.network.headline} caption={C.network.caption}
      drawer={<Drawer>
        <P text={C.network.drawer.louvain} />
        <p>
          {C.network.drawer.resolutions(joinList(o.settings.resolutions.map((r) => dec(r, r % 1 ? 1 : 0))))}
          {o.resolution_invariant ? ` ${C.network.drawer.invariant}` : ''} {C.network.drawer.ties}
        </p>
        <p>{C.network.drawer.sizes(dec(Math.min(...sizesOf)), dec(Math.max(...sizesOf)))}</p>
        <p>{C.network.drawer.cross(dec(g.edges.length), dec(cross))}</p>
        <p>{C.network.drawer.matrix}</p>
        {m && ps
          ? <PairMatrix order={d.byCommunity} value={liftValue} steps={liftSteps} legendTitle={C.network.drawer.legend} formatStep={dec} readout={liftReadout} title={C.network.drawer.matrixTitle} />
          : <Frame failed={failed} />}
      </Drawer>}>
      <div className="sc-controls">
        <Seg label={C.network.colourBy} options={['community', 'category'] as ColourBy[]} value={colourBy} onChange={setColour}
          name={(x) => (x === 'community' ? C.network.community : C.network.category)} />
        <label className="sc-control" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, height: 38, color: 'var(--sc-ink)' }}>
          <input type="checkbox" checked={rotate} onChange={() => setRotate(!rotate)} /> {C.network.turn}
        </label>
      </div>
      <div className="sc-map">
        <Projector coords={coords} colors={view.colors} keys={view.keys} labelFor={labelFor} tooltip={tooltip}
          selected={selected} neighbours={neighbours} isolate={isolate} autoRotate={rotate} onSelect={setSelected}
          sizes={sizes} edges={edges} />
        <aside className="sc-map-side fixed">
          {selected != null ? (
            <SkuPanel d={d} i={selected} colors={view.colors} onSelect={setSelected} />
          ) : (
            <>
              {colourBy === 'community' && (
                <div className="sc-map-lines">
                  <p className="num"><i className="sc-dot" style={{ visibility: 'hidden' }} />{C.network.communities(dec(d.comms.length))}</p>
                  <p className="num"><i className="sc-dot" style={{ background: UNLINKED }} />{C.network.unlinked(dec(d.unlinked.length))}</p>
                </div>
              )}
              <div className="sc-legend scroll">
                {view.legend.map((l) => (
                  <button key={l.key} aria-pressed={isolate === l.key} onClick={() => setIsolate(isolate === l.key ? null : l.key)}>
                    <i className="sc-dot" style={{ background: l.color }} /><span>{l.name}</span>
                  </button>
                ))}
              </div>
            </>
          )}
        </aside>
      </div>
    </Screen>
  )
}

function SkuPanel({ d, i, colors, onSelect }: { d: Derived; i: number; colors: string[]; onSelect: (i: number | null) => void }) {
  const node: AffinityNode = d.nodes[i]
  const lifts = new Map((d.partners.get(i) ?? []).map((p) => [p.j, p.lift]))
  const inside = node.alone ? [] : (d.members.get(node.community) ?? []).filter((j) => j !== i)
  const rows = [
    ...inside.map((j) => ({ j, lift: lifts.get(j) })).sort((a, b) => (b.lift ?? 0) - (a.lift ?? 0)),
    ...[...lifts.entries()].filter(([j]) => !inside.includes(j)).map(([j, lift]) => ({ j, lift })).sort((a, b) => (b.lift ?? 0) - (a.lift ?? 0)),
  ]
  return (
    <div className="sc-sku">
      <div className="sc-sku-head">
        <h3>{node.name}</h3>
        <button className="sc-button" onClick={() => onSelect(null)}>{C.network.close}</button>
      </div>
      <dl className="sc-facts">
        <dt>{C.network.category}</dt><dd>{node.category}</dd>
        <dt>{C.network.orders}</dt><dd>{dec(node.baskets)}</dd>
      </dl>
      <div className="sc-list-head"><span>{C.network.partners}</span>{rows.length > 0 && <span>{C.network.lift}</span>}</div>
      {rows.length ? (
        <div className="sc-legend scroll">
          {rows.map(({ j, lift }) => (
            <button key={j} onClick={() => onSelect(j)}>
              <i className="sc-dot" style={{ background: colors[j] }} />
              <span>{d.nodes[j].name} <small className="sc-caption">{d.nodes[j].category}</small></span>
              <span className="num">{lift != null ? dec(lift, 1) : ''}</span>
            </button>
          ))}
        </div>
      ) : <p className="sc-caption">{C.network.noPartners}</p>}
    </div>
  )
}

// =================================================================================================
// Screen 5: the check against the hidden bundles.

function TruthScreen({ o, ps }: { o: AffinityOverview; ps: PairStats | null }) {
  const T = o.truth.check
  const Q = o.quarter_note
  const rows = [
    { label: C.truth.categories, t: T.categories },
    { label: C.truth.counts, t: T.count_communities },
    { label: C.truth.ours, t: T.communities, ours: true },
  ]
  const liftShare = pc(o.facts.top200_lift_true_share)
  const cuts = ps?.cutShares.map((s) => pc(s))
  return (
    <Screen id="truth" headline={C.truth.headline} caption={C.truth.caption}
      drawer={<Drawer>
        <p>{C.truth.drawer.plants(dec(o.truth.bundles), dec(o.truth.skus_in_bundles))}</p>
        <p>{C.truth.drawer.measures(dec(T.communities.pairs_true))}</p>
        <div className="table-wrap" style={{ maxWidth: 640 }}>
          <table className="grid">
            <thead><tr><th>{C.truth.drawer.colGrouping}</th><th className="num">{C.truth.drawer.colPrecision}</th><th className="num">{C.truth.drawer.colRecall}</th><th className="num">{C.truth.drawer.colTogether}</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.label} className={r.ours ? 'hl' : ''}>
                  <td>{r.label}</td><td className="num">{dec(r.t.precision, 2)}</td><td className="num">{dec(r.t.recall, 2)}</td><td className="num">{dec(r.t.pairs_together)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>{C.truth.drawer.counts(dec(o.chosen.count_baseline_resolution, 1), dec(T.count_communities.bundles_exact), dec(o.truth.bundles),
          T.categories.bundles_exact === 0 ? C.truth.drawer.none : dec(T.categories.bundles_exact))}</p>
        <p>
          {C.truth.drawer.top(pc(o.facts.top200_count_true_share), liftShare)}
          {cuts && ` ${cuts.every((s) => s === liftShare) ? C.truth.drawer.cutSame
            : C.truth.drawer.cutRange(pc(Math.min(...ps!.cutShares)), pc(Math.max(...ps!.cutShares)))}`}
        </p>
        <p>{Q.without_quarter.val_cost < Q.with_quarter.val_cost
          ? C.truth.drawer.quarter(usd(Q.without_quarter.val_cost), usd(Q.with_quarter.val_cost), dec(Q.without_quarter.truth.ari, 2))
          : C.truth.drawer.quarterCheaper(usd(Q.with_quarter.val_cost), usd(Q.without_quarter.val_cost), dec(Q.without_quarter.truth.ari, 2))}</p>
      </Drawer>}>
      <Bars format={(v) => dec(v, 2)} max={1} rows={rows.map((r) => ({ label: r.label, value: Math.max(0, r.t.ari), ours: r.ours }))} />
    </Screen>
  )
}

// =================================================================================================
// Screen 6: SKUs placed into FCs.

function AssignScreen({ o, d, fcs, setFcs }: { o: AffinityOverview; d: Derived; fcs: number; setFcs: (n: number) => void }) {
  const [rule, setRule] = useState<RuleKey>('random')
  const [selected, setSelected] = useState<number | null>(null)
  const nodes = d.nodes
  const assign = o.assignments[rule][String(fcs)]
  const run = useCallback((i: number) => (nodes[i].alone ? `u${i}` : `c${nodes[i].community}`), [nodes])
  const community = useCallback((i: number) => (nodes[i].alone ? C.assign.unlinked : d.commName.get(nodes[i].community) ?? ''), [nodes, d])
  const describe = useCallback((i: number) => (
    <>
      <span><span className="k">{C.assign.sku}</span> <b>{nodes[i].name}</b></span>
      <span>{nodes[i].alone ? C.assign.unlinked : <><span className="k">{C.assign.community}</span> {community(i)}</>}</span>
      <span>{C.assign.fc(String(assign[i] + 1))}</span>
    </>
  ), [nodes, community, assign])
  const name = useCallback((i: number) => C.assign.square(nodes[i].name, community(i), C.assign.fc(String(assign[i] + 1))), [nodes, community, assign])
  const heading = useCallback((f: number) => C.assign.fc(String(f + 1)), [])
  // A community is split when its SKUs sit in more than one FC under this rule and FC count.
  const split = useMemo(() => {
    const out = new Set<number>()
    for (const [c, ids] of d.members) if (ids.length > 1 && ids.some((i) => assign[i] !== assign[ids[0]])) out.add(c)
    return out
  }, [d, assign])
  const pale = useCallback((i: number) => !nodes[i].alone && split.has(nodes[i].community), [nodes, split])
  // Rows of the tallest column under any rule at this FC count, so switching rules never jumps.
  const minRows = useMemo(() => Math.ceil(Math.max(...RULE_ORDER.flatMap((r) => {
    const a = o.assignments[r][String(fcs)]
    return Array.from({ length: fcs }, (_, f) => a.filter((x) => x === f).length)
  })) / 10), [o, fcs])

  // Both limits per rule, over every number of FCs: SKUs beyond the SKU limit, share beyond the workload limit.
  const over = useMemo(() => {
    const n = nodes.length
    const load = nodes.map((x) => x.baskets)
    const total = load.reduce((a, b) => a + b, 0)
    const out = {} as Record<RuleKey, { skus: number; load: number }>
    for (const r of RULE_ORDER) {
      const worst = { skus: 0, load: 0 }
      for (const k of o.settings.fcs) {
        const a = o.assignments[r][String(k)]
        const skuCap = Math.ceil((n / k) * (1 + o.settings.slack)), loadCap = (total / k) * (1 + o.settings.slack)
        for (let f = 0; f < k; f++) {
          let s = 0, w = 0
          a.forEach((x, i) => { if (x === f) { s++; w += load[i] } })
          worst.skus = Math.max(worst.skus, s - skuCap)
          worst.load = Math.max(worst.load, w / loadCap - 1)
        }
      }
      out[r] = { skus: worst.skus, load: worst.load > 1e-9 ? worst.load : 0 }
    }
    return out
  }, [o, nodes])
  const exceptions = RULE_ORDER.filter((r) => over[r].skus > 0 || over[r].load > 0)
  const bs = over.best_sellers
  const W = o.workload_note.validation_without_workload_limit
  const first = o.settings.fcs[0]
  const noLimit = W.best_sellers.find((x) => x.fcs === first)
  const noLimitOurs = W.communities.find((x) => x.fcs === first)

  // The settings grid, one row per false discovery rate (its cheapest resolution, first on ties).
  const levels = o.settings.fdr_levels.map((q) => {
    const rows = o.grid.filter((r) => r.fdr === q)
    return rows.reduce((a, b) => (b.val_cost < a.val_cost ? b : a), rows[0])
  }).filter(Boolean)
  const tieGroups: typeof levels[] = []
  for (const r of levels) {
    const grp = tieGroups.find((t) => t[0].edges === r.edges && Math.abs(t[0].val_cost - r.val_cost) < 1e-6)
    if (grp) grp.push(r); else tieGroups.push([r])
  }
  const costs = levels.map((r) => r.val_cost)
  const gap = (Math.max(...costs) - Math.min(...costs)) / Math.min(...costs)

  return (
    <Screen id="assign" headline={C.assign.headline} caption={C.assign.caption}
      drawer={<Drawer>
        <p>{C.assign.drawer.reading}</p>
        <p>{C.assign.drawer.split}</p>
        <p>{C.assign.drawer.site(String(o.settings.central_region), pc(o.settings.slack))}</p>
        <div className="table-wrap" style={{ maxWidth: 860 }}>
          <table className="grid">
            <thead><tr><th>{C.assign.drawer.colRule}</th><th>{C.assign.drawer.colHow}</th></tr></thead>
            <tbody>{RULE_ORDER.map((r) => <tr key={r}><td style={{ whiteSpace: 'nowrap' }}>{RULES[r]}</td><td>{RULE_HOW[r]}</td></tr>)}</tbody>
          </table>
        </div>
        <p>
          {exceptions.length ? C.assign.drawer.except(joinList(exceptions.map((r) => RULES_IN_PROSE[r]))) : C.assign.drawer.within}
          {exceptions.includes('best_sellers') && ` ${bs.skus > 0 && bs.load > 0 ? C.assign.drawer.overBoth(dec(bs.skus), pc(bs.load))
            : bs.skus > 0 ? C.assign.drawer.overSkus(dec(bs.skus)) : C.assign.drawer.overLoad(pc(bs.load))}`}
        </p>
        {noLimit && noLimitOurs && (
          <p>{C.assign.drawer.workload(pc(noLimit.split_share), String(first), pc(noLimit.load_share[0]), pc(noLimitOurs.split_share))}</p>
        )}
        <p>{C.assign.drawer.fee(usdCents(o.settings.parcel_fee), usdCents(o.settings.per_kg))}</p>
        <p>{C.assign.drawer.choose(String(Math.min(...o.settings.fcs)), String(Math.max(...o.settings.fcs)))}</p>
        <div className="table-wrap" style={{ maxWidth: 640 }}>
          <table className="grid">
            <thead><tr><th>{C.assign.drawer.colLevel}</th><th className="num">{C.assign.drawer.colPairs}</th><th className="num">{C.assign.drawer.colCost}</th></tr></thead>
            <tbody>
              {levels.map((r) => (
                <tr key={r.fdr} className={r.fdr === o.chosen.fdr ? 'hl' : ''}>
                  <td>{level(r.fdr)}</td><td className="num">{dec(r.edges)}</td><td className="num">{dec(r.val_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p>
          {C.assign.drawer.chosen(level(o.chosen.fdr))}
          {tieGroups.filter((t) => t.length > 1).map((t) => ` ${C.assign.drawer.ties(joinList(t.map((r) => level(r.fdr))), dec(t[0].edges))}`).join('')}
          {' '}{gap < 0.001 ? C.assign.drawer.gapUnder(pc(0.001, 1)) : C.assign.drawer.gap(pc(gap, 1))}
        </p>
        <p>{C.assign.drawer.twoChoices}</p>
      </Drawer>}>
      <div className="sc-controls">
        <Seg label={C.assign.rule} options={RULE_ORDER} value={rule} onChange={setRule} name={(r) => RULES[r]} />
        <Seg label={C.assign.fcs} options={o.settings.fcs} value={fcs} onChange={setFcs} name={String} />
      </div>
      <FcColumns fcs={fcs} assign={assign} order={d.byCommunity} run={run} color={d.colour} heading={heading}
        describe={describe} name={name} selected={selected} onSelect={setSelected} title={C.assign.columnsTitle} pale={pale} minRows={minRows} />
    </Screen>
  )
}

// =================================================================================================
// Screen 7: the business result on the test set.

const LINE_STYLE: Record<RuleKey, { color: string; dash?: string }> = {
  random: { color: '#8f8b9e', dash: '2 4' },
  popularity: { color: '#7a768c', dash: '8 4' },
  best_sellers: { color: '#666279', dash: '12 4 3 4' },
  category: { color: '#4f4b62' },
  communities: { color: ACCENT },
  hidden_bundles: { color: '#5e5a72', dash: '0.1 6' },
}

function ResultsScreen({ o, fcs, setFcs }: { o: AffinityOverview; fcs: number; setFcs: (n: number) => void }) {
  const R = o.results
  const at = (r: RuleKey, list: Record<RuleKey, PriceRow[]> = R, k = fcs) => list[r].find((x) => x.fcs === k)
  const ours = at('communities'), base = at('category')
  const ks = o.settings.fcs
  const fewer = ks.map((k) => 1 - (at('communities', R, k)?.split ?? 0) / (at('category', R, k)?.split ?? 1))
  const [lo, hi] = [Math.min(...fewer), Math.max(...fewer)]
  const same = ks.every((k) => {
    const a = at('communities', R, k), b = at('hidden_bundles', R, k)
    return a && b && a.split === b.split && a.parcels === b.parcels && Math.abs(a.cost - b.cost) < 1e-6
  })
  const V = o.results_validation, W = o.workload_note.validation_without_workload_limit
  const categoryFewest = ks.every((k) => PLAIN_RULES.every((r) => (at('category', V, k)?.split ?? Infinity) <= (at(r, V, k)?.split ?? Infinity)))
  const unlimitedFewer = ks.some((k) => (at('best_sellers', W, k)?.split ?? Infinity) < (at('category', V, k)?.split ?? -Infinity))
  const barOrder: RuleKey[] = ['random', 'popularity', 'best_sellers', 'category', 'communities', 'hidden_bundles']
  const series: Series[] = useMemo(() => (['random', 'popularity', 'best_sellers', 'category', 'communities', 'hidden_bundles'] as RuleKey[]).map((r) => ({
    id: r, label: RULES[r], color: LINE_STYLE[r].color, dash: LINE_STYLE[r].dash, labelColor: r === 'communities' ? ACCENT : '#5e5a72',
    values: o.settings.fcs.map((k) => (o.results[r].find((x) => x.fcs === k)?.split_share ?? 0) * 100),
  })), [o])
  const first = R.category[0]
  if (!ours || !base) return null

  return (
    <Screen id="results" headline={C.results.headline}
      drawer={<Drawer>
        <p>{C.results.drawer.relative}{categoryFewest && unlimitedFewer ? ` ${C.results.drawer.why}` : ''}</p>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>{C.results.drawer.colRule}</th><th>{C.results.drawer.colIdea}</th></tr></thead>
            <tbody>{RULE_ORDER.map((r) => <tr key={r}><td style={{ whiteSpace: 'nowrap' }}>{RULES[r]}</td><td>{RULE_SIMPLE[r]}</td></tr>)}</tbody>
          </table>
        </div>
        <p>{C.results.drawer.test(dec(first.orders), dec(first.multi))}</p>
        <p>{C.results.drawer.runs}</p>
        <div className="table-wrap">
          <table className="grid">
            <thead>
              <tr>
                <th>{C.results.drawer.colRule}</th><th className="num">{C.results.drawer.colFcs}</th><th className="num">{C.results.drawer.colSplit}</th>
                <th className="num">{C.results.drawer.colMulti}</th><th className="num">{C.results.drawer.colParcels}</th>
                <th className="num">{C.results.drawer.colCost}</th><th className="num">{C.results.drawer.colLoad}</th>
              </tr>
            </thead>
            <tbody>
              {RULE_ORDER.map((r) => R[r].map((row, k) => (
                <tr key={`${r}-${row.fcs}`} className={`${r === 'communities' ? 'hl' : ''}${k === 0 ? ' sc-group-start' : ''}`}>
                  <td style={{ whiteSpace: 'nowrap' }}>{k === 0 ? RULES[r] : ''}</td>
                  <td className="num">{row.fcs}</td>
                  <td className="num">{dec(row.split_share * 100, 1)}</td>
                  <td className="num">{dec(row.multi_split_share * 100, 1)}</td>
                  <td className="num">{dec(row.parcels)}</td>
                  <td className="num">{dec(row.cost)}</td>
                  <td className="num">{row.load_share.map((v) => dec(v * 100)).join(', ')}</td>
                </tr>
              )))}
            </tbody>
          </table>
        </div>
        <p className="sc-caption">{C.results.drawer.limit}</p>
        <p>{pc(lo) === pc(hi)
          ? C.results.drawer.rangeOne(String(Math.min(...ks)), String(Math.max(...ks)), pc(lo))
          : C.results.drawer.range(String(Math.min(...ks)), String(Math.max(...ks)), pc(lo), pc(hi))}</p>
        <p>{C.results.drawer.dip}</p>
        <LineChart title={C.results.drawer.chartTitle} x={o.settings.fcs} xLabel={C.results.drawer.chartX} series={series} height={300}
          format={(v) => dec(v)} hideTitle hideToggle hideLegend endLabels endGroup={0.5} yLabel={C.results.drawer.chartY} />
        <p>{C.results.drawer.cost}</p>
        {same && <p>{C.results.drawer.same}</p>}
      </Drawer>}>
      <div className="sc-controls">
        <Seg label={C.results.fcs} options={o.settings.fcs} value={fcs} onChange={setFcs} name={String} />
      </div>
      <Numbers items={[
        { value: change(ours.split / base.split - 1, 0), label: C.results.split },
        { value: change(ours.parcels / base.parcels - 1, 0), label: C.results.parcels },
        { value: change(ours.cost / base.cost - 1, 1), label: C.results.cost },
      ]} />
      <Bars format={(v) => pc(v)} rows={barOrder.map((r) => ({
        label: RULES[r], value: at(r)?.split_share ?? 0, ours: r === 'communities', note: r === 'hidden_bundles' ? C.results.hiddenNote : undefined,
      }))} />
    </Screen>
  )
}
