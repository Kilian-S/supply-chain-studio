import { useEffect, useMemo, useState } from 'react'
import LineChart from '../../components/LineChart'
import { cat } from '../charts/colors'
import Formula from '../charts/Formula'
import Heatmap from '../charts/Heatmap'
import { COLOUR_BY, FEATURE_NAMES, FINGERPRINT, POLICY_NAMES, STOCKING, TYPE_NAMES, money, pct } from '../lexicon'
import { modelFile, type Overview, type SkuRecord, type Variant } from '../types'
import { Bars, Drawer, Numbers, Screen, Sparkline } from '../ui'
import '../showcase.css'
import { COPY } from './copy'
import Projector from './Projector'

const SCREENS = ['top', 'question', 'data', 'fingerprint', 'map', 'groups', 'truth', 'worth', 'learned']
type ColourBy = 'cluster' | 'true_type' | 'category'

export default function SkuSegmentationPage() {
  const [o, setO] = useState<Overview | null>(null)
  const [skus, setSkus] = useState<SkuRecord[] | null>(null)
  const [base, setBase] = useState<Variant | null>(null)
  const [error, setError] = useState('')
  const [active, setActive] = useState('top')

  useEffect(() => {
    Promise.all([
      modelFile<Overview>('sku-segmentation', 'overview.json'),
      modelFile<SkuRecord[]>('sku-segmentation', 'skus.json'),
    ]).then(([ov, s]) => {
      setO(ov); setSkus(s)
      return modelFile<Variant>('sku-segmentation', `variants/m${ov.default.mask}.json`)
    }).then(setBase).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!base) return
    const io = new IntersectionObserver((es) => {
      const v = es.filter((e) => e.isIntersecting)[0]
      if (v) setActive(v.target.id)
    }, { rootMargin: '-45% 0px -45% 0px' })
    SCREENS.forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [base])

  if (error) return <div className="showcase"><div className="sc-page"><p className="sc-caption">Could not load the results: {error}</p></div></div>
  if (!o || !skus || !base) return <div className="showcase"><div className="sc-page"><p className="sc-caption" style={{ paddingTop: 40 }}>Loading...</p></div></div>

  const def = base.clusterings[`${o.default.model}-${o.default.k}`]
  const P = o.policies
  const ours = P[`cluster-${o.default.model}-${o.default.k}`]
  const saving = 1 - ours.total / P.uniform.total
  const bestFingerprint = [...o.subset_leaderboard].sort((a, b) => b.purity - a.purity)[0]

  return (
    <div className="showcase">
      <nav className="sc-dots" aria-label="Screens">
        {SCREENS.map((id) => <a key={id} href={`#/models/sku-segmentation`} className={active === id ? 'on' : ''} aria-label={id}
          onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }) }} />)}
      </nav>
      <div className="sc-page">
        <a className="sc-back" href="#/models">All models</a>

        <section id="top" className="sc-hero">
          <div className="sc-label">Product groups</div>
          <h1>{COPY.title}</h1>
          <p className="sc-sub">{COPY.intro}</p>
          <Numbers items={[
            { value: String(o.default.k), label: 'groups of products' },
            { value: pct(def.metrics.purity), label: 'matched to their true type', note: `Textbook rule: ${pct(o.baselines.vs_truth.abc_xyz.purity)}` },
            { value: `−${pct(saving)}`, label: 'stocking cost', note: 'against one rule for every product' },
          ]} />
        </section>

        <QuestionScreen o={o} />
        <DataScreen o={o} />
        <FingerprintScreen o={o} skus={skus} />
        <MapScreen o={o} skus={skus} />
        <GroupsScreen o={o} skus={skus} variant={base} />

        <Screen id="truth" {...COPY.screens.truth}
          drawer={<Drawer>
            <p>Two standard scores compare a grouping with the true types. The adjusted Rand index is 1 for a perfect match and 0 for chance; normalised mutual information is 1 for a perfect match.</p>
            <div className="table-wrap" style={{ maxWidth: 640 }}>
              <table className="grid">
                <thead><tr><th>Grouping</th><th className="num">Share matched</th><th className="num">Adjusted Rand index</th><th className="num">Mutual information</th></tr></thead>
                <tbody>
                  {(['abc', 'abc_xyz', 'sbc'] as const).map((k) => (
                    <tr key={k}><td>{POLICY_NAMES[k]}</td><td className="num">{pct(o.baselines.vs_truth[k].purity)}</td><td className="num">{o.baselines.vs_truth[k].ari.toFixed(2)}</td><td className="num">{o.baselines.vs_truth[k].nmi.toFixed(2)}</td></tr>
                  ))}
                  <tr><td>Our groups</td><td className="num">{pct(def.metrics.purity)}</td><td className="num">{def.metrics.ari_truth.toFixed(2)}</td><td className="num">{def.metrics.nmi_truth.toFixed(2)}</td></tr>
                </tbody>
              </table>
            </div>
            <h4>Our groups against the true types</h4>
            <Heatmap title="Groups against true types" mode="sequential" cell={42} rowWidth={240} colHeight={60}
              rows={def.names} cols={o.truth.types.map((t) => TYPE_NAMES[t])} format={(v) => String(v)}
              values={def.names.map((_, c) => o.truth.types.map((t) => skus.filter((s, i) => def.labels[i] === c && s.true_type === t).length))} />
            <h4>Choosing the number of groups</h4>
            <p>The default is the Gaussian mixture with the lowest Bayesian information criterion, <Formula tex={String.raw`\mathrm{BIC} = -2\ln\hat L + p\ln n`} />. Silhouette, bootstrap stability (8 refits on 80% subsamples) and the match with the true types are shown for both methods.</p>
            <div className="sc-grid cols-2">
              {(['silhouette', 'stability', 'ari_truth', 'bic'] as const).map((m) => (
                <div key={m}>
                  <LineChart title={{ silhouette: 'Silhouette', stability: 'Bootstrap stability', ari_truth: 'Adjusted Rand index vs true types', bic: 'BIC (mixture, lower is better)' }[m]}
                    x={o.k_range} xLabel="Groups" height={150} format={(v) => (m === 'bic' ? Math.round(v).toLocaleString('en-GB') : v.toFixed(2))}
                    series={(m === 'bic' ? ['gmm'] : o.models).map((mod, i) => ({ id: mod, label: mod === 'gmm' ? 'Gaussian mixture' : 'k-means', color: cat(i), values: o.k_range.map((k) => o.selection[`${mod}-${k}`][m]) }))} />
                </div>
              ))}
            </div>
            <p>The best fingerprint is picked out of 127 after the fact, so its share is optimistic. The default (all 16 numbers, chosen by BIC) was fixed in advance.</p>
          </Drawer>}>
          <Bars format={(v) => pct(v)} max={1} rows={[
            { label: 'Textbook rule: ABC', value: o.baselines.vs_truth.abc.purity, note: `${o.baselines.vs_truth.abc.groups} classes` },
            { label: 'Textbook rule: ABC/XYZ', value: o.baselines.vs_truth.abc_xyz.purity, note: `${o.baselines.vs_truth.abc_xyz.groups} classes` },
            { label: 'Demand classes', value: o.baselines.vs_truth.sbc.purity, note: `${o.baselines.vs_truth.sbc.groups} classes` },
            { label: 'Our groups', value: def.metrics.purity, note: `${o.default.k} groups, full fingerprint`, ours: true },
            { label: 'Our groups, best fingerprint', value: bestFingerprint.purity, note: `without ${o.groups.filter((g) => !bestFingerprint.groups.includes(g.key)).map((g) => FINGERPRINT[g.key].name.toLowerCase()).join(' or ')}` },
          ]} />
        </Screen>

        <Screen id="worth" {...COPY.screens.worth}
          drawer={<Drawer>
            <p>Each group gets one stocking rule: keep its products in every regional warehouse or only in the central warehouse, at an availability target of {o.policy_settings.service_levels.map((v) => pct(v, v > 0.99 ? 1 : 0)).join(', ')}. The cheapest rule per group on year {o.policy_settings.train_year} is then run on year {o.policy_settings.test_year} in a daily inventory simulation.</p>
            <p>Restocking: weekly review up to <Formula tex={String.raw`S = \mu(L+R) + z\,\sigma\sqrt{L+R}`} />, lead time {o.policy_settings.lead_days} days. Costs: holding (25% of unit cost a year plus $2 per litre), ${o.policy_settings.order_cost} per restocking order line, ${o.policy_settings.slot_cost} a year per product per warehouse, delivery by weight and distance, and the margin of every lost sale. Central stocking pools stock but delivers more slowly: {pct(o.policy_settings.slow_delivery_loss.same_landmass)} of mainland demand and {pct(o.policy_settings.slow_delivery_loss.cross_water)} across water does not wait.</p>
            <div className="table-wrap">
              <table className="grid">
                <thead><tr><th>Approach</th><th className="num">Rules</th><th className="num">Holding</th><th className="num">Restocking</th><th className="num">Shelf space</th><th className="num">Delivery</th><th className="num">Lost sales</th><th className="num">Total</th><th className="num">Served</th></tr></thead>
                <tbody>
                  {(['uniform', 'all_central', 'abc', 'abc_xyz', 'sbc', `cluster-${o.default.model}-${o.default.k}`, 'true_type', 'per_sku'] as const).map((k) => (
                    <tr key={k} className={k.startsWith('cluster') ? 'hl' : ''}>
                      <td>{POLICY_NAMES[k] ?? 'Our groups'}</td><td className="num">{P[k].rules}</td>
                      {(['holding', 'ordering', 'slots', 'outbound', 'lost_margin', 'total'] as const).map((c) => <td key={c} className="num">{money(P[k][c])}</td>)}
                      <td className="num">{pct(P[k].fill_rate, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>Our groups and the textbook ABC/XYZ rule cost about the same here ({pct(Math.abs(ours.total / P.abc_xyz.total - 1), 1)} apart). The difference is that a group can be assigned to a new product from its first weeks of sales and its attributes.</p>
          </Drawer>}>
          <Numbers items={[
            { value: `−${pct(saving)}`, label: 'stocking cost in the test year', note: `${money(ours.total)} instead of ${money(P.uniform.total)}` },
            { value: pct(ours.fill_rate, 1), label: 'of demand served', note: `One rule for every product: ${pct(P.uniform.fill_rate, 1)}` },
            { value: String(o.default.k), label: 'rules to manage', note: 'One per group' },
          ]} />
          <Bars format={money} rows={[
            { label: POLICY_NAMES.uniform, value: P.uniform.total, note: '1 rule' },
            { label: POLICY_NAMES.abc_xyz, value: P.abc_xyz.total, note: '9 rules' },
            { label: 'Our groups', value: ours.total, note: `${o.default.k} rules`, ours: true },
            { label: POLICY_NAMES.per_sku, value: P.per_sku.total, note: 'needs years of history for every product' },
          ]} />
        </Screen>

        <Screen id="learned" label={COPY.screens.learned.label} headline={COPY.screens.learned.headline}
          drawer={<Drawer title="Limits of this study">
            <p>The shop is simulated on purpose, so the true types are known. The stocking decision has two options per product, rules are fixed for a whole year, and there is one test year. Where to open or expand warehouses is a separate model.</p>
          </Drawer>}>
          <div className="sc-grid cols-3">
            {COPY.learned.map((t, i) => (
              <div key={i}><div className="sc-label" style={{ marginBottom: 8 }}>{String(i + 1).padStart(2, '0')}</div><p style={{ margin: 0, fontSize: 17, lineHeight: 1.45 }}>{t}</p></div>
            ))}
          </div>
        </Screen>
      </div>
    </div>
  )
}

// =================================================================================================

function QuestionScreen({ o }: { o: Overview }) {
  return (
    <Screen id="question" {...COPY.screens.question}
      drawer={<Drawer>
        <p>The shop is generated, so every product has a known hidden type. Revenue concentration, seasonality and how often customers order are calibrated to a real online retailer: <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii" target="_blank" rel="noreferrer">UCI Online Retail II</a> (1.07 million invoice lines, 2009 to 2011, CC BY 4.0).</p>
        <div className="table-wrap" style={{ maxWidth: 640 }}>
          <table className="grid">
            <thead><tr><th>Check</th><th className="num">Real retailer</th><th className="num">Our shop</th></tr></thead>
            <tbody>
              <tr><td>Revenue from the top 20% of products</td><td className="num">79%</td><td className="num">{pct(Number(o.world.checks.revenue_share_top20pct_skus))}</td></tr>
              <tr><td>Orders per customer per year</td><td className="num">3.1</td><td className="num">{Number(o.world.checks.orders_per_customer_per_year_mean).toFixed(1)}</td></tr>
              <tr><td>Products per order</td><td className="num">15 (wholesale)</td><td className="num">{Number(o.world.checks.mean_lines_per_order).toFixed(1)} (consumer)</td></tr>
            </tbody>
          </table>
        </div>
        <h4>Raw order lines</h4>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr>{Object.keys(o.data_sample[0]).map((k) => <th key={k}>{k}</th>)}</tr></thead>
            <tbody>{o.data_sample.slice(0, 8).map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      </Drawer>}>
      <div className="sc-grid fixed-3">
        {o.examples.map((e, i) => (
          <div key={e.sku_id}>
            <div className="sc-card-title"><i className="sc-dot" style={{ background: cat(i) }} />{TYPE_NAMES[e.true_type]}</div>
            <Sparkline values={e.weekly} color={cat(i)} label={`Weekly sales of a ${TYPE_NAMES[e.true_type].toLowerCase()} product`} />
            <p className="sc-card-text">{e.name}</p>
          </div>
        ))}
      </div>
    </Screen>
  )
}

function DataScreen({ o }: { o: Overview }) {
  const c = o.world.counts
  return (
    <Screen id="data" {...COPY.screens.data}>
      <Numbers items={[
        { value: c.order_lines.toLocaleString('en-GB'), label: 'order lines' },
        { value: c.orders.toLocaleString('en-GB'), label: 'orders' },
        { value: c.customers.toLocaleString('en-GB'), label: 'customers' },
        { value: String(c.regions), label: 'regions' },
      ]} />
      <div className="sc-years">
        {[1, 2, 3, 4, 5].map((y) => <div key={y}>Year {y}<small>{y < 5 ? 'Learn' : 'Test'}</small></div>)}
      </div>
    </Screen>
  )
}

function FingerprintScreen({ o, skus }: { o: Overview; skus: SkuRecord[] }) {
  const [pick, setPick] = useState(0)
  const ex = o.examples[pick]
  const sku = skus.find((s) => s.id === ex.sku_id)!
  const rank = (key: string, v: number) => skus.filter((s) => s.features[key] < v).length / (skus.length - 1)
  return (
    <Screen id="fingerprint" {...COPY.screens.fingerprint}
      drawer={<Drawer>
        <p>Each number is turned into a comparable scale before grouping: weekly totals instead of daily ones, only the weeks a product was on sale, a log scale for numbers that span orders of magnitude, then <Formula tex={String.raw`z = (x - \mu)/\sigma`} />. Each of the seven parts gets equal weight, <Formula tex={String.raw`1/\sqrt{n_g}`} /> per number, whatever its count.</p>
        <div className="table-wrap">
          <table className="grid">
            <thead><tr><th>Number</th><th>Definition</th><th>Meaning</th></tr></thead>
            <tbody>{o.features.map((f) => <tr key={f.key}><td>{f.label}</td><td><Formula tex={f.formula} /></td><td>{f.description}</td></tr>)}</tbody>
          </table>
        </div>
        <h4>How the 16 numbers relate</h4>
        <Heatmap title="Correlation between the numbers" mode="diverging" max={1} cell={28} rowWidth={150} colHeight={110}
          rows={o.features.map((f) => f.label)} cols={o.features.map((f) => f.label)} values={o.correlation} format={(v) => v.toFixed(1)} />
      </Drawer>}>
      <div className="sc-controls">
        <div className="sc-seg" role="group" aria-label="Example product">
          {o.examples.map((e, i) => <button key={e.sku_id} aria-pressed={pick === i} onClick={() => setPick(i)}>{TYPE_NAMES[e.true_type]}</button>)}
        </div>
      </div>
      <div className="sc-grid cols-4">
        {o.groups.map((g) => (
          <div key={g.key}>
            <div className="sc-card-title">{FINGERPRINT[g.key].name}</div>
            <p className="sc-card-text">{FINGERPRINT[g.key].text}</p>
            {g.features.map((f) => {
              const r = rank(f, sku.features[f])
              const label = FEATURE_NAMES[f]
              return (
                <div key={f} title={`${label}: higher than ${pct(r)} of products`} style={{ marginTop: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}><span>{label}</span><span className="num">{pct(r)}</span></div>
                  <div className="sc-bar-track" style={{ height: 6 }}><span className="sc-bar-fill" style={{ width: `${r * 100}%`, background: '#4338ca' }} /></div>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </Screen>
  )
}

// =================================================================================================

function MapScreen({ o, skus }: { o: Overview; skus: SkuRecord[] }) {
  const [mask, setMask] = useState(o.default.mask)
  const [model, setModel] = useState(o.default.model)
  const [k, setK] = useState(o.default.k)
  const [layout, setLayout] = useState<'umap3d' | 'pca3d'>('umap3d')
  const [colourBy, setColourBy] = useState<ColourBy>('cluster')
  const [variant, setVariant] = useState<Variant | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [isolate, setIsolate] = useState<string | null>(null)
  const [rotate, setRotate] = useState(true)

  useEffect(() => {
    let live = true
    modelFile<Variant>('sku-segmentation', `variants/m${mask}.json`).then((v) => { if (live) setVariant(v) })
    return () => { live = false }
  }, [mask])

  const c = variant?.clusterings[`${model}-${k}`]
  const view = useMemo(() => {
    if (!c) return null
    let keys: string[]; let order: string[]; let name: (key: string) => string
    if (colourBy === 'cluster') { keys = c.labels.map(String); order = c.names.map((_, i) => String(i)); name = (x) => c.names[Number(x)] }
    else if (colourBy === 'true_type') { keys = skus.map((s) => s.true_type); order = o.truth.types; name = (x) => TYPE_NAMES[x] }
    else { keys = skus.map((s) => s.category); order = o.world.categories; name = (x) => x }
    const colour = new Map(order.map((key, i) => [key, cat(i)]))
    return {
      keys, colors: keys.map((x) => colour.get(x)!),
      legend: order.map((key, i) => ({ key, name: name(key), color: cat(i), n: keys.filter((x) => x === key).length })),
      labelFor: Object.fromEntries(order.map((key) => [key, name(key)])),
    }
  }, [c, colourBy, skus, o])

  const sku = selected != null ? skus[selected] : null
  const neighbours = selected != null && variant ? variant.neighbours[selected].slice(0, 6) : []

  return (
    <Screen id="map" {...COPY.screens.map}
      drawer={<Drawer>
        <p>The map places products with similar fingerprints close together. Each fingerprint has 16 numbers, so the map squeezes it into three dimensions, which bends distances: neighbours on the map are close, far-apart regions are only roughly placed. The lines from a selected product go to its nearest neighbours in the full 16 numbers.</p>
        <div className="sc-controls">
          <div className="sc-control">Grouping method
            <div className="sc-seg" role="group" aria-label="Grouping method">
              <button aria-pressed={model === 'gmm'} onClick={() => setModel('gmm')}>Gaussian mixture</button>
              <button aria-pressed={model === 'kmeans'} onClick={() => setModel('kmeans')}>k-means</button>
            </div>
          </div>
          <div className="sc-control">Map projection
            <div className="sc-seg" role="group" aria-label="Map projection">
              <button aria-pressed={layout === 'umap3d'} onClick={() => setLayout('umap3d')}>UMAP</button>
              <button aria-pressed={layout === 'pca3d'} onClick={() => setLayout('pca3d')}>PCA</button>
            </div>
          </div>
        </div>
        <p><b>k-means</b> splits the products into k groups that minimise <Formula tex={String.raw`\sum_c \sum_{x \in C_c} \lVert x - \mu_c \rVert^2`} />. The <b>Gaussian mixture</b> models the fingerprints as <Formula tex={String.raw`p(x) = \sum_c \pi_c\, \mathcal N(x \mid \mu_c, \mathrm{diag}\,\sigma_c^2)`} /> and gives every product a probability per group. UMAP keeps local neighbourhoods; PCA keeps the directions of largest spread{variant ? ` (${pct(variant.explained_variance.reduce((a, b) => a + b, 0))} of it in three dimensions)` : ''}.</p>
        {c && <p>Current settings: silhouette {c.metrics.silhouette.toFixed(2)}, bootstrap stability {c.metrics.stability.toFixed(2)}, adjusted Rand index against the true types {c.metrics.ari_truth.toFixed(2)}. All 127 fingerprints x 2 methods x 9 group counts are computed in advance.</p>}
      </Drawer>}>
      <div className="sc-controls">
        <div className="sc-control">Colour by
          <div className="sc-seg" role="group" aria-label="Colour by">
            {(Object.keys(COLOUR_BY) as ColourBy[]).map((x) => <button key={x} aria-pressed={colourBy === x} onClick={() => { setColourBy(x); setIsolate(null) }}>{COLOUR_BY[x]}</button>)}
          </div>
        </div>
        <div className="sc-control">Number of groups
          <div className="sc-seg" role="group" aria-label="Number of groups">
            {o.k_range.map((x) => <button key={x} aria-pressed={k === x} onClick={() => { setK(x); setIsolate(null) }}>{x}</button>)}
          </div>
        </div>
        <label className="sc-control" style={{ flexDirection: 'row', alignItems: 'center', gap: 8, color: 'var(--sc-ink)' }}>
          <input type="checkbox" checked={rotate} onChange={() => setRotate(!rotate)} /> Turn slowly
        </label>
      </div>
      <div className="sc-control">Fingerprint used for grouping
        <div className="sc-chips">
          {o.groups.map((g, i) => (
            <button key={g.key} className="sc-chip" aria-pressed={!!(mask >> i & 1)}
              onClick={() => { const m = mask ^ (1 << i); if (m) { setMask(m); setIsolate(null) } }}>{FINGERPRINT[g.key].name}</button>
          ))}
        </div>
      </div>

      <div className="sc-map">
        {variant && view ? (
          <Projector coords={variant.embeddings[layout]} colors={view.colors} keys={view.keys} labelFor={view.labelFor}
            tooltip={(i) => `${skus[i].name} · ${skus[i].category}`} selected={selected} neighbours={neighbours}
            isolate={isolate} autoRotate={rotate} onSelect={setSelected} />
        ) : <div className="proj" />}
        <aside className="sc-map-side">
          {sku && c ? (
            <div>
              <div className="sc-label">{sku.category}</div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ margin: '4px 0', fontSize: 20 }}>{sku.name}</h3>
                <button className="sc-button" onClick={() => setSelected(null)}>Close</button>
              </div>
              <dl className="sc-facts">
                <dt>Group</dt><dd>{c.names[c.labels[selected!]]}</dd>
                <dt>True type</dt><dd>{TYPE_NAMES[sku.true_type]}</dd>
                <dt>Sells a week</dt><dd>{sku.features.mean_weekly_units.toFixed(1)}</dd>
                <dt>Weeks with a sale</dt><dd>{pct(1 - sku.features.zero_share)}</dd>
                <dt>Price</dt><dd>${sku.price.toFixed(2)}</dd>
              </dl>
              <Sparkline values={sku.weekly} label={`Weekly sales of ${sku.name}`} />
              <p className="sc-caption">Weekly sales, five years</p>
              <div className="sc-label" style={{ marginTop: 14 }}>Most similar products</div>
              <div className="sc-legend">
                {neighbours.map((j) => (
                  <button key={j} onClick={() => setSelected(j)}>
                    <i className="sc-dot" style={{ background: view?.colors[j] }} /><span>{skus[j].name}</span><span className="sc-caption">{TYPE_NAMES[skus[j].true_type]}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div>
              <div className="sc-label">{COLOUR_BY[colourBy]}</div>
              <p className="sc-caption" style={{ margin: '4px 0 8px' }}>Click to show one on its own.</p>
              <div className="sc-legend">
                {view?.legend.map((l) => (
                  <button key={l.key} aria-pressed={isolate === l.key} onClick={() => setIsolate(isolate === l.key ? null : l.key)}>
                    <i className="sc-dot" style={{ background: l.color }} /><span>{l.name}</span><span className="num sc-caption">{l.n}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
          {c && (
            <Numbers items={[{ value: pct(c.metrics.purity), label: 'matched to their true type' }]} />
          )}
        </aside>
      </div>
    </Screen>
  )
}

function GroupsScreen({ o, skus, variant }: { o: Overview; skus: SkuRecord[]; variant: Variant }) {
  const c = variant.clusterings[`${o.default.model}-${o.default.k}`]
  const weeks = Math.floor(o.world.train_days / 7)
  const typical = (g: number) => {
    const members = skus.filter((_, i) => c.labels[i] === g)
    return Array.from({ length: weeks }, (_, w) => {
      const xs = members.map((s) => s.weekly[w]).sort((a, b) => a - b)
      return xs[Math.floor(xs.length / 2)] ?? 0
    })
  }
  return (
    <Screen id="groups" {...COPY.screens.groups}
      drawer={<Drawer>
        <p>Names come from each group's average fingerprint: the main sales pattern, plus its strongest other trait when two groups share a pattern. Values in the cards are medians. Below, every group's average on all 16 numbers, in standard deviations from the shop average (blue below, red above).</p>
        <Heatmap title="Group profiles" mode="diverging" max={2} cell={30} rowWidth={250} colHeight={110}
          rows={c.names} cols={o.features.map((f) => f.label)} values={c.profile} format={(v) => v.toFixed(1)} />
      </Drawer>}>
      <div className="sc-grid cols-4">
        {c.names.map((name, g) => {
          const s = c.stats[g]
          return (
            <div key={name}>
              <div className="sc-card-title"><i className="sc-dot" style={{ background: cat(g) }} />{name}</div>
              <p className="sc-card-text">{c.sizes[g]} products</p>
              <Sparkline values={typical(g)} color={cat(g)} label={`Typical weekly sales in ${name}`} />
              <dl className="sc-facts">
                <dt>Sells a week</dt><dd>{s.mean_weekly_units.toFixed(1)}</dd>
                <dt>Weeks with a sale</dt><dd>{pct(s.sells_in_weeks)}</dd>
                <dt>Price</dt><dd>${s.price.toFixed(0)}</dd>
                <dt>Size</dt><dd>{s.volume_l.toFixed(1)} litres</dd>
              </dl>
              <div className="sc-rule">{STOCKING(c.policy[g].mode, c.policy[g].service)}</div>
            </div>
          )
        })}
      </div>
    </Screen>
  )
}
