import { useEffect, useMemo, useState } from 'react'
import LineChart from '../../components/LineChart'
import { cat } from '../charts/colors'
import Heatmap from '../charts/Heatmap'
import { modelFile, type Overview, type SkuRecord, type Variant } from '../types'
import Projector from './Projector'

export type ColourBy = 'cluster' | 'true_type' | 'category' | 'abc' | 'sbc'
export interface ExplorerState {
  mask: number
  model: string
  k: number
  embedding: 'umap3d' | 'pca3d' | 'umap2d'
  colourBy: ColourBy
}

const pct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

export default function Explorer({ overview, skus, state, setState }: {
  overview: Overview; skus: SkuRecord[]; state: ExplorerState; setState: (s: ExplorerState) => void
}) {
  const [variant, setVariant] = useState<Variant | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [isolate, setIsolate] = useState<string | null>(null)
  const [rotate, setRotate] = useState(true)
  const [query, setQuery] = useState('')

  useEffect(() => {
    let live = true
    modelFile<Variant>('sku-segmentation', `variants/m${state.mask}.json`).then((v) => { if (live) setVariant(v) })
    return () => { live = false }
  }, [state.mask])

  const clustering = variant?.clusterings[`${state.model}-${state.k}`]
  const groups = overview.groups

  // Colour key: one value per SKU, plus the legend entries.
  const { key, legend } = useMemo(() => {
    if (!clustering) return { key: [] as string[], legend: [] as { id: string; label: string; color: string; count: number }[] }
    let values: string[]
    let order: string[]
    let label = (v: string) => v
    switch (state.colourBy) {
      case 'cluster':
        values = clustering.labels.map(String)
        order = clustering.names.map((_, i) => String(i))
        label = (v) => clustering.names[Number(v)]
        break
      case 'true_type':
        values = skus.map((s) => s.true_type); order = overview.truth.types; break
      case 'category':
        values = skus.map((s) => s.category); order = overview.world.categories; break
      case 'abc':
        values = skus.map((s) => s.abc); order = ['A', 'B', 'C']; break
      case 'sbc':
        values = skus.map((s) => s.sbc); order = overview.baselines.sbc.classes; break
    }
    const legend = order.map((v, i) => ({ id: v, label: label(v), color: cat(i), count: values.filter((x) => x === v).length }))
    return { key: values, legend }
  }, [clustering, state.colourBy, skus, overview])

  const colourOf = useMemo(() => new Map(legend.map((l) => [l.id, l.color])), [legend])
  const colors = useMemo(() => key.map((v) => colourOf.get(v) ?? '#8a8699'), [key, colourOf])
  const labels = useMemo(() => skus.map((s, i) => {
    const c = clustering ? clustering.names[clustering.labels[i]] : ''
    return `${s.name} · ${s.category} · ${c}`
  }), [skus, clustering])
  const dimmed = useMemo(() => (isolate == null ? null : key.map((v) => v !== isolate)), [isolate, key])
  const coords = variant?.embeddings[state.embedding] ?? []
  const neighbours = selected != null && variant ? variant.neighbours[selected] : []

  const toggleGroup = (i: number) => {
    const mask = state.mask ^ (1 << i)
    if (mask) setState({ ...state, mask })
  }
  const search = (q: string) => {
    setQuery(q)
    const hit = skus.findIndex((s) => s.name.toLowerCase().includes(q.trim().toLowerCase()))
    if (q.trim() && hit >= 0) setSelected(hit)
  }

  const m = clustering?.metrics
  const sku = selected != null ? skus[selected] : null
  const weeksTrain = Math.floor(overview.world.train_days / 7)

  return (
    <div className="explorer">
      <div className="explorer-controls">
        <fieldset className="chip-group">
          <legend>Feature groups</legend>
          {groups.map((g, i) => (
            <button key={g.key} className="chip" aria-pressed={!!(state.mask >> i & 1)} onClick={() => toggleGroup(i)}
              title={g.features.join(', ')}>{g.label}</button>
          ))}
        </fieldset>
        <div className="control-row">
          <label className="field">Model
            <select className="select" value={state.model} onChange={(e) => setState({ ...state, model: e.target.value })}>
              <option value="gmm">Gaussian mixture</option><option value="kmeans">k-means</option>
            </select>
          </label>
          <label className="field">Clusters (k)
            <select className="select" value={state.k} onChange={(e) => setState({ ...state, k: Number(e.target.value) })}>
              {overview.k_range.map((k) => <option key={k} value={k}>{k}{k === overview.default.k && state.model === 'gmm' && state.mask === overview.default.mask ? ' (lowest BIC)' : ''}</option>)}
            </select>
          </label>
          <label className="field">Projection
            <select className="select" value={state.embedding} onChange={(e) => setState({ ...state, embedding: e.target.value as ExplorerState['embedding'] })}>
              <option value="umap3d">UMAP 3D</option><option value="pca3d">PCA 3D</option><option value="umap2d">UMAP 2D</option>
            </select>
          </label>
          <label className="field">Colour by
            <select className="select" value={state.colourBy} onChange={(e) => { setIsolate(null); setState({ ...state, colourBy: e.target.value as ColourBy }) }}>
              <option value="cluster">Cluster</option><option value="true_type">True type (hidden)</option>
              <option value="category">Category</option><option value="abc">ABC class</option><option value="sbc">Demand class (SBC)</option>
            </select>
          </label>
          <label className="field">Find a SKU
            <input className="input" placeholder="e.g. PET-0123" value={query} onChange={(e) => search(e.target.value)} />
          </label>
          <label className="check" style={{ alignSelf: 'flex-end' }}>
            <input type="checkbox" checked={rotate} onChange={() => setRotate(!rotate)} disabled={state.embedding === 'umap2d'} /> Rotate
          </label>
        </div>
      </div>

      {m && (
        <div className="metric-strip" aria-label="Scores for this configuration">
          <div><span>Silhouette</span><b>{m.silhouette.toFixed(3)}</b></div>
          <div><span>Bootstrap stability (ARI)</span><b>{m.stability.toFixed(2)} ± {m.stability_sd.toFixed(2)}</b></div>
          <div><span>Recovers true types (ARI)</span><b>{m.ari_truth.toFixed(2)}</b></div>
          <div><span>NMI vs true types</span><b>{m.nmi_truth.toFixed(2)}</b></div>
          {m.policy_vs_abc_xyz != null && <div><span>Policy cost vs ABC/XYZ (year 5)</span><b>{pct(m.policy_vs_abc_xyz)}</b></div>}
          {m.bic != null && <div><span>BIC</span><b>{Math.round(m.bic).toLocaleString('en-GB')}</b></div>}
        </div>
      )}

      <div className="explorer-main">
        <div className="projector-wrap">
          {variant ? (
            <Projector coords={coords} colors={colors} labels={labels} selected={selected} neighbours={neighbours} dimmed={dimmed}
              flat={state.embedding === 'umap2d'} autoRotate={rotate && selected == null} onSelect={setSelected} />
          ) : <div className="projector" />}
          <p className="note projector-hint">Drag to rotate, scroll to zoom, click a point to inspect it and draw lines to its 8 nearest neighbours in feature space. Very distant groups are drawn closer than they are.
            {variant && state.embedding === 'pca3d' && ` PCA keeps ${(variant.explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of the variance.`}</p>
        </div>
        <aside className="explorer-side">
          {sku && clustering && (
            <div className="section sku-card">
              <div className="panel-head">
                <div><div className="eyebrow">{sku.category}</div><h3 style={{ fontSize: 18 }}>{sku.name}</h3></div>
                <button className="btn small" onClick={() => setSelected(null)}>Clear</button>
              </div>
              <div className="kv">
                <span>Cluster</span><b>{clustering.names[clustering.labels[selected!]]}{clustering.prob ? ` (p = ${clustering.prob[selected!].toFixed(2)})` : ''}</b>
                <span>True type</span><b>{sku.true_type}{sku.bulky ? ', bulky' : ''}</b>
                <span>ABC / XYZ / SBC</span><b>{sku.abc} / {sku.xyz} / {sku.sbc}</b>
                <span>Price, margin</span><b>${sku.price.toFixed(2)}, {(sku.margin * 100).toFixed(0)}%</b>
                <span>Mean weekly units</span><b>{sku.features.mean_weekly_units.toFixed(1)}</b>
                <span>ADI, CV²</span><b>{sku.features.adi.toFixed(2)}, {sku.features.cv2.toFixed(2)}</b>
                <span>Seasonality, trend</span><b>{sku.features.seasonality.toFixed(2)}, {sku.features.trend.toFixed(2)}</b>
              </div>
              <LineChart title="Weekly units: training years and held-out year" x={sku.weekly.map((_, i) => i + 1)} xLabel="Week" height={130}
                series={[
                  { id: 'train', label: 'Training', color: '#2a78d6', values: sku.weekly.map((v, i) => (i < weeksTrain ? v : null)) },
                  { id: 'test', label: 'Held-out year', color: '#eb6834', values: sku.weekly.map((v, i) => (i >= weeksTrain - 1 ? v : null)) },
                ]} />
              <h3>Nearest neighbours</h3>
              <ol className="neighbours">
                {neighbours.map((j, r) => (
                  <li key={j}><button className="linklike" onClick={() => setSelected(j)}>{skus[j].name}</button>
                    <span className="muted"> {skus[j].true_type}, d = {variant!.neighbour_dist[selected!][r].toFixed(2)}</span></li>
                ))}
              </ol>
            </div>
          )}
          <div className="section">
            <h3>Legend</h3>
            <p className="note">Click an entry to isolate it.</p>
            <div className="legend-list">
              {legend.map((l) => (
                <button key={l.id} className="legend-item" aria-pressed={isolate === l.id} onClick={() => setIsolate(isolate === l.id ? null : l.id)}>
                  <i className="dot" style={{ background: l.color }} />
                  <span>{l.label}</span>
                  <span className="muted mono">{l.count}</span>
                  {state.colourBy === 'cluster' && clustering?.policy[Number(l.id)] && (
                    <span className="tag">{clustering.policy[Number(l.id)].mode === 'central' ? 'central' : 'every FC'} · {Math.round(clustering.policy[Number(l.id)].service * 1000) / 10}%</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </aside>
      </div>

      {clustering && variant && (
        <div className="two-col">
          <div className="section">
            <h3>Cluster profiles</h3>
            <p className="note">Mean standardised value of every feature per cluster (blue below average, red above). This is what the names are read from.</p>
            <Heatmap title="Cluster profiles" mode="diverging" max={2} cell={30} rowWidth={230} colHeight={110}
              rows={clustering.names.map((n, i) => `${n} (${clustering.sizes[i]})`)}
              cols={overview.features.map((f) => f.label)} values={clustering.profile} format={(v) => v.toFixed(1)} />
          </div>
          <div className="section">
            <h3>Clusters against the true types</h3>
            <p className="note">Counts of SKUs. A perfect recovery would put each true type in exactly one cluster.</p>
            <Heatmap title="Clusters against true types" mode="sequential" cell={40} rowWidth={230} colHeight={90}
              rows={clustering.names}
              cols={overview.truth.types}
              values={clustering.names.map((_, c) => overview.truth.types.map((t) => skus.filter((s, i) => clustering.labels[i] === c && s.true_type === t).length))}
              format={(v) => String(v)} />
          </div>
        </div>
      )}
    </div>
  )
}
