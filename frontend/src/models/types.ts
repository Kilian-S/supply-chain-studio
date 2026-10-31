// Shapes of the prepared model results (backend/src/scgame/ml/sku_segmentation.py).

export interface ModelCard {
  id: string
  title: string
  status: 'ready' | 'planned'
  question: string
  method: string
  headline?: string
}

export interface Hist { counts: number[]; edges: number[] }

export interface FeatureInfo {
  key: string
  group: string
  label: string
  formula: string
  description: string
  log: boolean
  raw_hist: Hist
  z_hist: Hist
  raw_summary: { min: number; median: number; max: number }
}

export interface Metrics {
  silhouette: number
  inertia: number | null
  bic: number | null
  stability: number
  stability_sd: number
  ari_truth: number
  nmi_truth: number
  val_cost?: number
  val_vs_abc_xyz?: number
  purity: number
}

export interface GroupStats {
  mean_weekly_units: number
  sells_in_weeks: number
  seasonality: number
  growth_per_year: number
  price: number
  volume_l: number
  weight_kg: number
  margin: number
  solo_share: number
  qty_per_line: number
}

export interface PolicyResult {
  holding: number
  ordering: number
  slots: number
  outbound: number
  lost_margin: number
  total: number
  served: number
  demand: number
  stock_value: number
  points: number
  fill_rate: number
  rules: number
  choices?: Record<string, { mode: string; service: number; skus: number }>
}

export interface Overview {
  model: { id: string; title: string }
  world: {
    version: number
    counts: Record<string, number>
    checks: Record<string, unknown>
    days: number
    train_days: number
    categories: string[]
  }
  data_sample: Record<string, string | number>[]
  examples: { sku_id: number; name: string; true_type: string; daily: number[]; weekly: number[] }[]
  groups: { key: string; label: string; features: string[] }[]
  features: FeatureInfo[]
  correlation: number[][]
  truth: { types: string[]; counts: number[] }
  baselines: {
    abc_xyz_grid: { rows: string[]; cols: string[]; counts: number[][] }
    sbc: { classes: string[]; counts: number[] }
    sbc_points: { adi: number[]; cv2: number[] }
    vs_truth: Record<string, { ari: number; nmi: number; purity: number; groups: number; crosstab: number[][]; rows: string[] }>
  }
  selection: Record<string, Metrics>
  default: { mask: number; model: string; k: number; embedding: string }
  policies: Record<string, PolicyResult>
  policy_settings: {
    lead_days: number
    review_days: number
    order_cost: number
    slot_cost: number
    slow_delivery_loss: Record<string, number>
    service_levels: number[]
    central_region: number
    train_year: number
    validation_year: number
    test_year: number
  }
  k_range: number[]
  models: string[]
  tradeoff: {
    settings: [number, number, number, number, string][] // validation cost, purity, ARI, mask, method-k
    chosen: [number, string]
    baselines: Record<string, { val_cost: number; purity: number; ari: number }>
  }
  subset_leaderboard: {
    mask: number
    groups: string[]
    best: string
    ari_truth: number
    silhouette: number
    stability: number
    purity: number
    best_policy: string
    val_vs_abc_xyz: number
  }[]
}

export interface Clustering {
  labels: number[]
  prob: number[] | null
  metrics: Metrics
  sizes: number[]
  profile: number[][]
  names: string[]
  policy: { mode: string; service: number }[]
  stats: GroupStats[]
}

export interface Variant {
  mask: number
  groups: string[]
  features: string[]
  explained_variance: number[]
  embeddings: Record<'umap3d' | 'pca3d' | 'umap2d', number[][]>
  clusterings: Record<string, Clustering>
  neighbours: number[][]
  neighbour_dist: number[][]
}

export interface SkuRecord {
  id: number
  name: string
  category: string
  true_type: string
  bulky: boolean
  price: number
  margin: number
  launch_day: number
  features: Record<string, number>
  abc: string
  xyz: string
  sbc: string
  weekly: number[]
}

const cache = new Map<string, Promise<unknown>>()

export function modelFile<T>(model: string, path: string): Promise<T> {
  const url = `/api/models/${model}/${path}`
  if (!cache.has(url)) {
    cache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} loading ${path}`)
      return r.json()
    }))
  }
  return cache.get(url) as Promise<T>
}

export const listModels = () => fetch('/api/models').then((r) => r.json() as Promise<ModelCard[]>)
