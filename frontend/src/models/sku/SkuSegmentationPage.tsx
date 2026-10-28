import { useEffect, useState } from 'react'
import { modelFile, type Overview, type SkuRecord } from '../types'
import Explorer, { type ExplorerState } from './Explorer'
import { Analysis, Architecture, Baselines, Data, Features, Problem, Section, Transformations } from './Sections'

const TOC = [
  ['problem', 'The problem'], ['data', 'Data'], ['transformations', 'Transformations'], ['features', 'Feature engineering'],
  ['baselines', 'Baselines'], ['architecture', 'Model and selection'], ['outputs', 'Outputs: explorer'], ['analysis', 'Analysis'],
]

export default function SkuSegmentationPage() {
  const [overview, setOverview] = useState<Overview | null>(null)
  const [skus, setSkus] = useState<SkuRecord[] | null>(null)
  const [error, setError] = useState('')
  const [explorer, setExplorer] = useState<ExplorerState | null>(null)
  const [active, setActive] = useState('problem')

  useEffect(() => {
    Promise.all([modelFile<Overview>('sku-segmentation', 'overview.json'), modelFile<SkuRecord[]>('sku-segmentation', 'skus.json')])
      .then(([o, s]) => {
        setOverview(o)
        setSkus(s)
        setExplorer({ mask: o.default.mask, model: o.default.model, k: o.default.k, embedding: 'umap3d', colourBy: 'cluster' })
      })
      .catch((e) => setError(e.message))
  }, [])

  // Highlight the section in view.
  useEffect(() => {
    if (!overview) return
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
      if (vis[0]) setActive(vis[0].target.id)
    }, { rootMargin: '-20% 0px -70% 0px' })
    TOC.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el) })
    return () => io.disconnect()
  }, [overview])

  if (error) return <div className="model-page"><div className="error">Could not load the model results: {error}</div></div>
  if (!overview || !skus || !explorer) return <div className="model-page"><p className="note">Loading model J...</p></div>

  const load = (s: Partial<ExplorerState>) => {
    setExplorer({ ...explorer, ...s })
    document.getElementById('outputs')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="model-page">
      <nav className="toc" aria-label="Sections">
        <a href="#/models" className="toc-back">All models</a>
        <div className="eyebrow">Model J</div>
        {TOC.map(([id, label], i) => (
          <a key={id} href={`#/models/sku-segmentation`} className={active === id ? 'on' : ''}
            onClick={(e) => { e.preventDefault(); document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' }) }}>
            <span className="mono">{i + 1}</span> {label}
          </a>
        ))}
      </nav>
      <article className="model-article">
        <header className="model-header">
          <div className="eyebrow">Model J · Unsupervised learning · Clustering</div>
          <h1>SKU segmentation</h1>
          <p className="lead">Which products behave alike, so that each group can get its own listing and stocking policy?
            Sixteen engineered features, k-means and Gaussian mixtures, checked against hidden ground truth and priced in a simulation of a year the model never saw.</p>
        </header>
        <Problem o={overview} />
        <Data o={overview} />
        <Transformations o={overview} />
        <Features o={overview} />
        <Baselines o={overview} skus={skus} />
        <Architecture o={overview} />
        <Section id="outputs" n={7} title="Outputs: the explorer">
          <p>Every combination of feature groups (127), model (2) and k (2 to 10) is precomputed. Change them and watch the clusters, the scores
            and the policy cost move. Colour by the hidden true type to see what the model could not see.</p>
          <Explorer overview={overview} skus={skus} state={explorer} setState={setExplorer} />
        </Section>
        <Analysis o={overview} load={load} />
      </article>
    </div>
  )
}
