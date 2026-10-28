import { useState } from 'react'
import LineChart from '../../components/LineChart'
import { cat } from '../charts/colors'
import Formula from '../charts/Formula'
import HBars from '../charts/HBars'
import Heatmap from '../charts/Heatmap'
import Histogram from '../charts/Histogram'
import Scatter from '../charts/Scatter'
import type { Overview, SkuRecord } from '../types'
import type { ExplorerState } from './Explorer'

const money = (v: number) => `$${Math.round(v).toLocaleString('en-GB')}`
const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export function Callout({ kind, title, children }: { kind: 'key' | 'intuition' | 'caveat'; title: string; children: React.ReactNode }) {
  return (
    <aside className={`callout ${kind}`}>
      <div className="callout-title">{title}</div>
      <div>{children}</div>
    </aside>
  )
}

export function Section({ id, n, title, children }: { id: string; n: number; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="model-section">
      <div className="section-number">{n}</div>
      <h2>{title}</h2>
      {children}
    </section>
  )
}

// =================================================================================================

export function Problem({ o }: { o: Overview }) {
  const d = o.selection[`gmm-${o.default.k}`]
  const best = o.subset_leaderboard[0]
  const pol = o.policies[`cluster-gmm-${o.default.k}`]
  return (
    <Section id="problem" n={1} title="The problem">
      <p className="lead">
        A retailer with thousands of products cannot write a stocking rule per product, and one rule for everything wastes money:
        fast sellers run out, slow sellers clog every warehouse. Segmentation groups products that behave alike so that each group
        gets its own policy: where to list it, how much safety stock, how often to replenish.
      </p>
      <div className="two-col">
        <div>
          <h3>Why it matters</h3>
          <ul className="tight">
            <li><b>Assortment Excellence</b>: segmentation and performance of the assortment drive stocking costs.</li>
            <li><b>Listing</b>: which products to list in every fulfilment centre and which only centrally.</li>
            <li><b>Replenishment and inventory</b>: service targets and safety stock by segment instead of one global setting.</li>
          </ul>
        </div>
        <div>
          <h3>What this setting adds</h3>
          <p>
            The world is simulated, so every SKU has a hidden <b>true demand type</b> that the model never sees. We can therefore
            measure how well clustering recovers reality, and price the policies it leads to in a simulation of the held-out year.
            Neither is possible with real data alone.
          </p>
        </div>
      </div>
      <div className="tiles">
        <div className="tile"><span>Clusters (lowest BIC)</span><b>{o.default.k}</b></div>
        <div className="tile"><span>True-type recovery, all features</span><b>ARI {d.ari_truth.toFixed(2)}</b><small>ABC/XYZ: {o.baselines.vs_truth.abc_xyz.ari.toFixed(2)}</small></div>
        <div className="tile"><span>Best feature subset</span><b>ARI {best.ari_truth.toFixed(2)}</b><small>{best.groups.join(', ')}</small></div>
        <div className="tile"><span>Cluster policies vs one rule for all</span><b>{pct(pol.total / o.policies.uniform.total - 1)}</b><small>held-out year, simulated</small></div>
      </div>
    </Section>
  )
}

// =================================================================================================

export function Data({ o }: { o: Overview }) {
  const c = o.world.counts
  const years = o.world.days / 365
  return (
    <Section id="data" n={2} title="Data">
      <p>
        Five years of order lines from the simulated retailer (world generator v1). Basket structure, revenue concentration,
        seasonality and customer order frequency are calibrated to <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii" target="_blank" rel="noreferrer">UCI Online Retail II</a> (CC BY 4.0)
        where it fits a consumer retailer; see <code>reference/calibration/README.md</code>.
      </p>
      <div className="tiles">
        {[['Regions', c.regions], ['SKUs', c.skus], ['Customers', c.customers], ['Orders', c.orders], ['Order lines', c.order_lines], ['Categories', o.world.categories.length]].map(([k, v]) => (
          <div className="tile" key={k as string}><span>{k}</span><b>{Number(v).toLocaleString('en-GB')}</b></div>
        ))}
      </div>
      <h3>Training and held-out data</h3>
      <div className="timeline-bar" aria-label="Years 1 to 4 for training, year 5 held out">
        {Array.from({ length: years }, (_, i) => (
          <div key={i} className={i < o.world.train_days / 365 ? 'train' : 'test'}>Year {i + 1}<small>{i < o.world.train_days / 365 ? 'features' : 'held out: policy test'}</small></div>
        ))}
      </div>
      <p className="note">Features use days 1 to {o.world.train_days} only. Year 5 is never seen by the clustering; it is used once, to price the policies.</p>
      <h3>A sample of raw order lines</h3>
      <div className="table-wrap">
        <table className="grid">
          <thead><tr>{Object.keys(o.data_sample[0]).map((k) => <th key={k}>{k}</th>)}</tr></thead>
          <tbody>{o.data_sample.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j} className={typeof v === 'number' ? 'num' : ''}>{String(v)}</td>)}</tr>)}</tbody>
        </table>
      </div>
      <h3>The hidden truth</h3>
      <p>Each SKU was generated with one of six demand types. The labels are used only for evaluation.</p>
      <HBars title="SKUs per true type" format={(v) => String(v)}
        bars={o.truth.types.map((t, i) => ({ label: t, value: o.truth.counts[i], color: cat(i) }))} />
    </Section>
  )
}

// =================================================================================================

export function Transformations({ o }: { o: Overview }) {
  const [ex, setEx] = useState(0)
  const [feat, setFeat] = useState('mean_weekly_units')
  const e = o.examples[ex]
  const f = o.features.find((x) => x.key === feat)!
  return (
    <Section id="transformations" n={3} title="Transformations">
      <p>Raw order lines become one comparable row per SKU in four steps.</p>
      <ol className="steps">
        <li><b>Aggregate to weeks.</b> Daily demand is dominated by noise and by when orders happen to land; weekly totals keep the pattern
          and drop the jitter. 1,460 daily values become 209 weekly values.</li>
        <li><b>Restrict to weeks on sale.</b> A product launched in year 3 should not look intermittent because of the two years before it existed.</li>
        <li><b>Log-transform skewed features.</b> Volume, revenue, price and size span orders of magnitude; <Formula tex="x' = \log(1 + x)" /> stops the
          biggest sellers from dominating every distance.</li>
        <li><b>Standardise and weight by group.</b> <Formula tex="z = (x' - \mu)/\sigma" />, then each feature in a group of size <Formula tex="n_g" /> is
          scaled by <Formula tex="1/\sqrt{n_g}" />, so every group contributes the same variance whether it has one feature or three.</li>
      </ol>
      <div className="toggle-row" role="tablist" aria-label="Example SKU">
        {o.examples.map((x, i) => (
          <button key={x.sku_id} className="chip" aria-pressed={ex === i} onClick={() => setEx(i)}>{x.true_type}</button>
        ))}
      </div>
      <div className="two-col">
        <LineChart title={`${e.name}: daily units`} x={e.daily.map((_, i) => i + 1)} series={[{ id: 'd', label: 'Daily', color: '#8a8699', values: e.daily }]} height={150} />
        <LineChart title={`${e.name}: weekly units`} x={e.weekly.map((_, i) => i + 1)} xLabel="Week" series={[{ id: 'w', label: 'Weekly', color: cat(ex), values: e.weekly }]} height={150} />
      </div>
      <h3>Before and after scaling</h3>
      <label className="field" style={{ maxWidth: 320 }}>Feature
        <select className="select" value={feat} onChange={(x) => setFeat(x.target.value)}>
          {o.features.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
        </select>
      </label>
      <div className="two-col">
        <Histogram hist={f.raw_hist} title={`${f.label}, raw (median ${f.raw_summary.median.toPrecision(3)})`} color="#8a8699" />
        <Histogram hist={f.z_hist} title={`${f.label}, ${f.log ? 'log-transformed and ' : ''}standardised`} />
      </div>
    </Section>
  )
}

// =================================================================================================

export function Features({ o }: { o: Overview }) {
  const labels = o.features.map((f) => f.label)
  return (
    <Section id="features" n={4} title="Feature engineering">
      <p>
        Sixteen features in seven groups describe how a product sells, what it is, and how it is bought. Demand features follow the
        forecasting literature (Syntetos-Boylan classes, Hyndman's seasonal strength); product and basket features carry what matters
        for logistics: a bulky, low-margin product is expensive to spread across many warehouses.
      </p>
      <div className="table-wrap">
        <table className="grid feature-table">
          <thead><tr><th>Group</th><th>Feature</th><th>Definition</th><th>Meaning</th></tr></thead>
          <tbody>
            {o.features.map((f, i) => (
              <tr key={f.key}>
                <td>{i === 0 || o.features[i - 1].group !== f.group ? o.groups.find((g) => g.key === f.group)!.label : ''}</td>
                <td><b>{f.label}</b>{f.log && <span className="tag" style={{ marginLeft: 6 }}>log</span>}</td>
                <td><Formula tex={f.formula} /></td>
                <td className="small">{f.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>Correlation between features</h3>
      <p>Pearson correlation after scaling. Strongly correlated features measure the same thing twice; the group weighting keeps such pairs
        (ADI and zero-demand weeks, volume and weight) from counting double.</p>
      <Heatmap title="Feature correlation" mode="diverging" max={1} rows={labels} cols={labels} values={o.correlation} cell={30} rowWidth={150} colHeight={110} format={(v) => v.toFixed(1)} />
      <Callout kind="key" title="Key terms">
        <b>ADI</b> (average demand interval): how many periods pass, on average, between periods with demand. <b>CV²</b>: squared
        coefficient of variation of the non-zero demand sizes. Together they define the Syntetos-Boylan classes used as a baseline below.
      </Callout>
    </Section>
  )
}

// =================================================================================================

export function Baselines({ o, skus }: { o: Overview; skus: SkuRecord[] }) {
  const g = o.baselines.abc_xyz_grid
  const vs = o.baselines.vs_truth
  return (
    <Section id="baselines" n={5} title="Baselines">
      <p>Before any machine learning: the two classifications a planner would use today. Both are rules, not learned.</p>
      <div className="two-col">
        <div>
          <h3>ABC / XYZ grid</h3>
          <p className="small">ABC by cumulative revenue (A: first 80%, B: next 15%, C: last 5%). XYZ by coefficient of variation of weekly demand
            (X below 0.5, Y up to 1.0, Z above).</p>
          <Heatmap title="ABC/XYZ grid" mode="sequential" rows={g.rows} cols={g.cols} values={g.counts} cell={56} rowWidth={40} colHeight={30} format={(v) => String(v)} />
        </div>
        <div>
          <h3>Syntetos-Boylan demand classes</h3>
          <p className="small">Cut-offs ADI = 1.32 and CV² = 0.49 split smooth, erratic, intermittent and lumpy demand. Points coloured by the hidden true type.</p>
          <Scatter title="ADI against CV squared" x={o.baselines.sbc_points.adi} y={o.baselines.sbc_points.cv2.map((v) => Math.max(v, 0.01))}
            logX logY vLines={[1.32]} hLines={[0.49]} xLabel="ADI (log)" yLabel="CV² (log)"
            color={(i) => cat(o.truth.types.indexOf(skus[i].true_type))} label={(i) => `${skus[i].name}: ${skus[i].true_type}, ${skus[i].sbc}`} />
          <div className="chart-legend">{o.truth.types.map((t, i) => <span key={t}><i className="swatch" style={{ background: cat(i) }} />{t}</span>)}</div>
        </div>
      </div>
      <h3>How well do the rules recover the true types?</h3>
      <p>Adjusted Rand index (ARI): 1 means identical groupings, 0 means no better than chance.
        <Formula block tex={String.raw`\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}{\tfrac12\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right] - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}`} /></p>
      <HBars title="Baseline recovery" format={(v) => v.toFixed(2)} max={1}
        bars={[['ABC', vs.abc.ari], ['ABC/XYZ', vs.abc_xyz.ari], ['Syntetos-Boylan', vs.sbc.ari]].map(([l, v]) => ({ label: l as string, value: v as number, color: '#8a8699' }))} />
    </Section>
  )
}

// =================================================================================================

export function Architecture({ o }: { o: Overview }) {
  const ks = o.k_range
  const series = (metric: 'silhouette' | 'stability' | 'ari_truth') => o.models.map((m, i) => ({
    id: m, label: m === 'gmm' ? 'Gaussian mixture' : 'k-means', color: cat(i), values: ks.map((k) => o.selection[`${m}-${k}`][metric]),
  }))
  return (
    <Section id="architecture" n={6} title="Model architecture and selection">
      <div className="two-col">
        <div>
          <h3>k-means</h3>
          <p>Partitions the SKUs into <i>k</i> groups that minimise the within-cluster sum of squares:</p>
          <Formula block tex={String.raw`\min_{C_1..C_k} \sum_{c=1}^{k} \sum_{x \in C_c} \lVert x - \mu_c \rVert^2`} />
          <p className="small">Lloyd's algorithm alternates assigning each point to its nearest centre and moving each centre to the mean of its points;
            10 random starts (k-means++), the best kept. Hard assignments, spherical clusters of similar size.</p>
        </div>
        <div>
          <h3>Gaussian mixture</h3>
          <p>Models the data as a mixture of <i>k</i> Gaussians with diagonal covariance and fits it by expectation-maximisation:</p>
          <Formula block tex={String.raw`p(x) = \sum_{c=1}^{k} \pi_c \, \mathcal{N}\!\left(x \mid \mu_c, \mathrm{diag}(\sigma_c^2)\right)`} />
          <p className="small">Soft assignments: every SKU gets a probability per cluster, so borderline products are visible. Clusters may differ in
            size and spread per feature. 3 starts, covariance floor 10⁻³.</p>
        </div>
      </div>
      <h3>Choosing k</h3>
      <p>No single score decides; we read four together, for k from 2 to 10, on all features:</p>
      <ul className="tight small">
        <li><b>Silhouette</b>: <Formula tex={String.raw`s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}`} />, how much closer a point is to its own cluster (a) than to the next (b).</li>
        <li><b>BIC</b> (mixture only): <Formula tex={String.raw`\mathrm{BIC} = -2 \ln \hat L + p \ln n`} />, fit penalised by parameter count; lower is better. The default k is its minimum.</li>
        <li><b>Stability</b>: refit on 8 random 80% subsamples and compare with the full fit (ARI). Clusters that move when the data moves are not real.</li>
        <li><b>Recovery</b> of the true types (ARI): available only because the world is simulated.</li>
      </ul>
      <div className="three-col">
        <LineChart title="Silhouette" x={ks} xLabel="k" series={series('silhouette')} format={(v) => v.toFixed(2)} height={160} />
        <LineChart title="Bootstrap stability" x={ks} xLabel="k" series={series('stability')} format={(v) => v.toFixed(2)} height={160} />
        <LineChart title="Recovery of true types (ARI)" x={ks} xLabel="k" series={series('ari_truth')} format={(v) => v.toFixed(2)} height={160} />
      </div>
      <LineChart title="BIC, Gaussian mixture (lower is better)" x={ks} xLabel="k" height={150} format={(v) => Math.round(v).toLocaleString('en-GB')}
        series={[{ id: 'bic', label: 'BIC', color: cat(1), values: ks.map((k) => o.selection[`gmm-${k}`].bic) }]} />
      <Callout kind="intuition" title="Reading the scores">
        Silhouettes around 0.1 to 0.25 mean the groups overlap: products form a continuum, not islands. That is normal for demand data
        and is why soft assignments (the mixture) and a business check (section 8) matter more than a high silhouette.
      </Callout>
    </Section>
  )
}

// =================================================================================================

export function Analysis({ o, load }: { o: Overview; load: (s: Partial<ExplorerState>) => void }) {
  const d = `cluster-gmm-${o.default.k}`
  const P = o.policies
  const policyRows: [string, string][] = [
    ['uniform', 'One rule: every SKU in every FC, 95%'],
    ['all_central', 'One rule: every SKU central only, 95%'],
    ['abc', 'ABC classes (3 rules)'],
    ['abc_xyz', 'ABC/XYZ grid (9 rules)'],
    ['sbc', 'Syntetos-Boylan classes (4 rules)'],
    [d, `Gaussian mixture, all features, k = ${o.default.k}`],
    ['cluster-kmeans-8', 'k-means, all features, k = 8'],
    ['true_type', 'True types (6 rules, oracle)'],
    ['per_sku', 'Tuned per SKU (300 rules)'],
  ]
  const s = o.policy_settings
  const lb = o.subset_leaderboard
  const byPolicy = [...lb].sort((a, b) => a.policy_vs_abc_xyz - b.policy_vs_abc_xyz)
  const rec = [
    { label: 'ABC/XYZ', value: o.baselines.vs_truth.abc_xyz.ari, color: '#8a8699' },
    { label: 'Syntetos-Boylan', value: o.baselines.vs_truth.sbc.ari, color: '#8a8699' },
    { label: `Mixture, all features, k=${o.default.k}`, value: o.selection[`gmm-${o.default.k}`].ari_truth, color: cat(0) },
    { label: `Best subset: ${lb[0].groups.join(' + ')}`, value: lb[0].ari_truth, color: cat(2), highlight: true },
  ]
  const toLoad = (r: (typeof lb)[number], key: string) => {
    const [model, k] = key.split('-')
    load({ mask: r.mask, model, k: Number(k), colourBy: 'cluster' })
  }
  return (
    <Section id="analysis" n={8} title="Analysis">
      <h3>1. Recovering the true demand types</h3>
      <HBars title="Recovery of true types" format={(v) => v.toFixed(2)} max={1} bars={rec} />
      <p>
        Clustering beats both rule-based baselines. But the best recovery does not come from all features: it comes from{' '}
        <b>{lb[0].groups.join(', ')}</b> (ARI {lb[0].ari_truth.toFixed(2)}). Physical and economic features are informative for logistics but are
        not part of the demand type, so they pull clusters towards "bulky" or "high margin" instead.
      </p>

      <h3>2. Which features make good clusters? Two different answers</h3>
      <div className="two-col">
        <div>
          <p className="small"><b>Best at recovering the true types</b></p>
          <SubsetTable rows={lb.slice(0, 6)} metric="ari" onLoad={(r) => toLoad(r, r.best)} />
        </div>
        <div>
          <p className="small"><b>Best stocking policies</b> (year-5 cost vs ABC/XYZ)</p>
          <SubsetTable rows={byPolicy.slice(0, 6)} metric="policy" onLoad={(r) => toLoad(r, r.best_policy)} />
        </div>
      </div>
      <Callout kind="caveat" title="Selection bias">
        The right-hand table picks the best of 127 subsets × 18 models by their year-5 result, so its numbers are optimistic. The honest
        figure is the pre-specified default (all features, k by BIC) in the table below. A clean comparison would select on a
        validation year; that is the next step once the world has a longer history.
      </Callout>

      <h3>3. The business check: pricing the policies</h3>
      <p>
        Each group gets a stocking rule: stock the SKU in <b>every regional FC</b> or <b>only in the central FC</b> ({s.central_region === 1 ? 'Calopeia' : `region ${s.central_region}`}), at a
        target service level of {s.service_levels.map((v) => `${v * 100}%`).join(', ')}. The rule per group is chosen on year {s.train_year} and then
        applied to year {s.test_year}, which no step has seen.
      </p>
      <div className="table-wrap" style={{ maxWidth: 760 }}>
        <table className="grid">
          <tbody>
            <tr><td>Replenishment</td><td>Weekly review, order-up-to <Formula tex={String.raw`S = \mu(L+R) + z_{\alpha}\,\sigma\sqrt{L+R}`} />, lead time {s.lead_days} days, from the previous year's daily demand</td></tr>
            <tr><td>Costs</td><td>Holding 25% of unit cost a year plus $2 per litre; {money(s.order_cost)} per replenishment line; {money(s.slot_cost)} a year per stocking point; outbound $1.50 + $0.35/kg, ×1.4 across the continent, ×2.2 across water; lost sales cost their margin</td></tr>
            <tr><td>Central stocking</td><td>Pools stock, but slower delivery loses {pct(s.slow_delivery_loss.same_landmass)} of demand elsewhere on the continent and {pct(s.slow_delivery_loss.cross_water)} across water</td></tr>
          </tbody>
        </table>
      </div>
      <HBars title="Total cost in the held-out year" format={money}
        bars={policyRows.map(([k, label]) => ({ label, value: P[k].total, color: k === d ? cat(0) : k.startsWith('cluster') ? cat(2) : k === 'true_type' || k === 'per_sku' ? '#6250d6' : '#8a8699', highlight: k === d }))} />
      <div className="table-wrap">
        <table className="grid">
          <thead><tr><th>Policy</th><th className="num">Rules</th><th className="num">Holding</th><th className="num">Orders</th><th className="num">Slots</th><th className="num">Outbound</th><th className="num">Lost margin</th><th className="num">Total</th><th className="num">Fill rate</th></tr></thead>
          <tbody>
            {policyRows.map(([k, label]) => (
              <tr key={k} className={k === d ? 'hl' : ''}><td>{label}</td><td className="num">{P[k].rules}</td>
                {(['holding', 'ordering', 'slots', 'outbound', 'lost_margin', 'total'] as const).map((c) => <td key={c} className="num">{money(P[k][c])}</td>)}
                <td className="num">{pct(P[k].fill_rate)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>4. Findings</h3>
      <ul className="tight">
        <li>Grouping beats a single rule by about {pct(1 - P[d].total / P.uniform.total)} in the held-out year: most of the gain is not spreading slow sellers across every FC.</li>
        <li>With all features, clusters are about as good as the ABC/XYZ grid for this cost model ({pct(Math.abs(P[d].total / P.abc_xyz.total - 1))} {P[d].total > P.abc_xyz.total ? 'more expensive' : 'cheaper'}). They differ in <i>which</i> products they group, and they give every product a probability rather than a hard class.</li>
        <li>What a cluster means depends on the features you choose: demand-pattern features recover the generating types; volume, size and economics make cheaper policies. Choose features for the decision, not for the score.</li>
        <li>Tuning every SKU individually is cheapest here ({money(P.per_sku.total)}), but it needs history for every product. Clusters give a rule for a brand-new SKU from its attributes on day one.</li>
      </ul>
      <Callout kind="caveat" title="Limitations">
        Synthetic world (by design); a simple two-mode network decision; static policies for a full year; one held-out year. The policy check
        prices segmentation, it does not optimise the network: that is model O.
      </Callout>
    </Section>
  )
}

function SubsetTable({ rows, metric, onLoad }: { rows: Overview['subset_leaderboard']; metric: 'ari' | 'policy'; onLoad: (r: Overview['subset_leaderboard'][number]) => void }) {
  return (
    <div className="table-wrap">
      <table className="grid">
        <thead><tr><th>Feature groups</th><th>Model</th><th className="num">{metric === 'ari' ? 'ARI' : 'vs ABC/XYZ'}</th><th /></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.mask}>
              <td className="small">{r.groups.join(', ')}</td>
              <td className="small mono">{metric === 'ari' ? r.best : r.best_policy}</td>
              <td className="num">{metric === 'ari' ? r.ari_truth.toFixed(2) : `${(r.policy_vs_abc_xyz * 100).toFixed(1)}%`}</td>
              <td><button className="btn small" onClick={() => onLoad(r)}>Explore</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
