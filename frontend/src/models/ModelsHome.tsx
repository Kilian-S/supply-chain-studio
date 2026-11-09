import { useEffect, useState } from 'react'
import './showcase.css'
import { listModels, type ModelCard } from './types'

// The models chosen in docs/ML_ROADMAP.md, in storyline order. Cards for built ones come from the backend.
// A card is a title and its question; models not built yet are greyed out and not clickable.
const PLANNED: ModelCard[] = [
  { id: 'sku-segmentation', title: 'SKU clustering', status: 'planned', question: 'Which SKUs sell alike, so each cluster can get its own stocking policy?', method: 'Clustering' },
  { id: 'product-affinity', title: 'Bought together', status: 'planned', question: 'Which SKUs are bought together, so they should be stored in the same FC?', method: 'Clustering' },
  { id: 'demand-forecast', title: 'Demand five years out', status: 'planned', question: 'How much will each region and product sell over five years, and how sure are we?', method: 'Forecasting' },
  { id: 'new-region-demand', title: 'Demand in a new region', status: 'planned', question: 'How much would a new region bring before it has any history?', method: 'Prediction' },
  { id: 'what-if-surrogate', title: 'Instant what-if', status: 'planned', question: 'Expand FC A or FC B: what happens to cost and service level?', method: 'Prediction from simulation' },
  { id: 'risk-classifier', title: 'Risk of split orders', status: 'planned', question: 'Which orders and SKUs are at risk under a planned network?', method: 'Classification' },
  { id: 'network-design', title: 'Network plan', status: 'planned', question: 'Where to open or expand FCs, and by how much, over five years?', method: 'Optimisation' },
]

export default function ModelsHome() {
  const [ready, setReady] = useState<ModelCard[]>([])
  useEffect(() => { listModels().then(setReady).catch(() => setReady([])) }, [])
  const cards = PLANNED.map((p) => ready.find((r) => r.id === p.id) ?? p)
  return (
    <div className="showcase">
      <div className="sc-page">
        <section className="sc-hero">
          <h1>Models</h1>
          <p className="sc-sub">Long-term supply chain questions, answered in a simulated shop where the truth is known.</p>
        </section>
        <div className="sc-grid cols-2" style={{ marginTop: 32 }}>
          {cards.map((c) => {
            const body = (
              <>
                <h3>{c.title}</h3>
                <p className="sc-card-text">{c.question}</p>
              </>
            )
            return c.status === 'ready'
              ? <a key={c.id} href={`#/models/${c.id}`} className="sc-model-card">{body}</a>
              : <div key={c.id} className="sc-model-card off">{body}</div>
          })}
        </div>
      </div>
    </div>
  )
}
