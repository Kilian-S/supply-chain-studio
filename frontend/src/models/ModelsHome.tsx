import { useEffect, useState } from 'react'
import { listModels, type ModelCard } from './types'

// The selected models from ML_ROADMAP.md; cards for built models come from the backend.
const PLANNED: ModelCard[] = [
  { id: 'sku-segmentation', letter: 'J', title: 'SKU segmentation', status: 'planned', question: 'Which products behave alike, so each group gets its own stocking and listing policy?', method: 'Unsupervised: clustering' },
  { id: 'product-affinity', letter: 'K', title: 'Product affinity', status: 'planned', question: 'Which products are bought together, so they should be stored together?', method: 'Unsupervised: co-purchase graph, embeddings' },
  { id: 'demand-forecast', letter: 'B', title: 'Long-range demand forecasting', status: 'planned', question: 'What will demand be per region, category and SKU over five years, and how uncertain?', method: 'Supervised regression, neural network' },
  { id: 'new-region-demand', letter: 'C', title: 'New-region demand', status: 'planned', question: 'How much demand would a new region or site bring, before it has any history?', method: 'Supervised regression (cold start)' },
  { id: 'what-if-surrogate', letter: 'A', title: 'What-if surrogate', status: 'planned', question: 'Expand FC A or FC B: what happens to cost, service and split orders?', method: 'Supervised regression on simulator runs, neural network' },
  { id: 'risk-classifier', letter: 'H', title: 'Split and stockout risk', status: 'planned', question: 'Which orders, SKUs or regions are at risk under a planned network?', method: 'Supervised classification' },
  { id: 'network-design', letter: 'O', title: 'Network design', status: 'planned', question: 'Where to open or expand FCs, and by how much, over five years?', method: 'Operations research: multi-period MILP' },
]

export default function ModelsHome() {
  const [ready, setReady] = useState<ModelCard[]>([])
  useEffect(() => { listModels().then(setReady).catch(() => setReady([])) }, [])
  const cards = PLANNED.map((p) => ready.find((r) => r.id === p.id) ?? p)
  return (
    <div className="start">
      <div className="start-inner">
        <div>
          <div className="eyebrow">Machine learning and operations research</div>
          <h1>Models</h1>
          <p className="note" style={{ maxWidth: 720, marginTop: 8 }}>
            Long-term supply chain questions answered in the simulated world. Each model page walks through the full development:
            data, transformations, features, architecture, outputs and analysis, checked against the world's hidden ground truth.
          </p>
        </div>
        <div className="cards">
          {cards.map((c) => (
            <a key={c.id} className={`card model-card${c.status === 'ready' ? '' : ' planned'}`} href={c.status === 'ready' ? `#/models/${c.id}` : undefined}
              aria-disabled={c.status !== 'ready'}>
              <div className="panel-head">
                <span className="model-letter">{c.letter}</span>
                <span className={`pill ${c.status === 'ready' ? 'ok' : 'off'}`}>{c.status === 'ready' ? 'READY' : 'PLANNED'}</span>
              </div>
              <h3>{c.title}</h3>
              <span className="note">{c.question}</span>
              <span className="small muted">{c.method}</span>
              {c.headline && <span className="small"><b>{c.headline}</b></span>}
            </a>
          ))}
        </div>
      </div>
    </div>
  )
}
