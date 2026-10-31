/* The single source of visible wording on the model pages. Rules (checked by lexicon.test.ts):
 * Kilian's approved vocabulary may appear on the surface (SKU, feature, clustering, k, centroid,
 * kNN, z-score, PCA, ground truth, purity, ARI, ABC/XYZ, ADI, CV², stocking policy, FC, service
 * level, fill rate...). Terms on the reading list (GLOSSARY.md: GMM, soft assignment, EM, UMAP, NMI,
 * silhouette, BIC, bootstrap stability, Syntetos-Boylan) stay inside "How it works" drawers.
 * British English, no em or en dashes. */

export const pct = (x: number, decimals = 0) => `${(x * 100).toFixed(decimals)}%`
export const money = (x: number) => (Math.abs(x) >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : `$${Math.round(x).toLocaleString('en-GB')}`)

// Hidden demand types as the generator defines them, in plain words.
export const TYPE_NAMES: Record<string, string> = {
  steady: 'Steady',
  seasonal: 'Seasonal',
  trending: 'Growing',
  declining: 'Fading',
  intermittent: 'Intermittent',
  lumpy: 'Bulk orders',
}

// The seven parts of a product's fingerprint.
// The seven groups of features.
// Feature groups, named plainly.
export const FINGERPRINT: Record<string, string> = {
  volume: 'Sales volume',
  variability: 'Variability',
  intermittency: 'Intermittency',
  pattern: 'Seasonality and trend',
  physical: 'Size and weight',
  economic: 'Price and margin',
  basket: 'Basket',
}

// Plain names for the 16 features, unit in brackets (formal names and formulas stay in the drawers).
export const FEATURE_NAMES: Record<string, string> = {
  mean_weekly_units: 'Units per week',
  revenue: 'Revenue ($k)',
  cv2: 'CV²',
  adi: 'ADI (weeks)',
  zero_share: 'Weeks without a sale (%)',
  seasonality: 'Seasonal strength',
  trend: 'Growth per year (%)',
  recent_share: 'Share sold in year 4 (%)',
  volume_l: 'Unit volume (L)',
  weight_kg: 'Unit weight (kg)',
  margin: 'Margin (%)',
  price: 'Price ($)',
  holding_to_margin: 'Holding cost to margin',
  solo_share: 'Single-item orders (%)',
  basket_size: 'Basket size (SKUs)',
  qty_per_line: 'Units per order line',
}

// Each feature's value as a bare number in the unit its name gives, fixed decimals per feature.
const num = (x: number, d: number) => (x < 0 ? '−' : '') + Math.abs(x).toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d })
export const FEATURE_FORMAT: Record<string, (v: number) => string> = {
  mean_weekly_units: (v) => num(v, 1),
  revenue: (v) => num(v / 1000, 0),
  cv2: (v) => num(v, 2),
  adi: (v) => num(v, 2),
  zero_share: (v) => num(v * 100, 0),
  seasonality: (v) => num(v, 2),
  trend: (v) => num((Math.exp(v) - 1) * 100, 0),
  recent_share: (v) => num(v * 100, 0),
  volume_l: (v) => num(v, 1),
  weight_kg: (v) => num(v, 1),
  margin: (v) => num(v * 100, 0),
  price: (v) => num(v, 2),
  holding_to_margin: (v) => num(v, 2),
  solo_share: (v) => num(v * 100, 0),
  basket_size: (v) => num(v, 1),
  qty_per_line: (v) => num(v, 1),
}

export const COLOUR_BY: Record<string, string> = {
  cluster: 'Cluster',
  true_type: 'Ground truth',
  category: 'Category',
}

export const STOCKING = (mode: string, service: number) =>
  `${mode === 'central' ? 'Central FC only' : 'Every regional FC'}, ${pct(service, service > 0.99 ? 1 : 0)} service level`

export const POLICY_NAMES: Record<string, string> = {
  uniform: 'One policy for every SKU',
  all_central: 'Every SKU in the central FC',
  abc: 'ABC analysis (3 classes)',
  abc_xyz: 'ABC/XYZ analysis',
  sbc: 'Demand classes by ADI and CV² (4)',
  true_type: 'Ground truth types (6, for reference)',
  per_sku: 'One policy per SKU',
}

// Words that must never appear on the visible surface outside a drawer (checked by the test).
// Reading-list terms: allowed in drawers and GLOSSARY.md, not on the surface.
export const BANNED_ON_SURFACE = ['gmm', 'gaussian mixture', 'soft assignment', 'expectation', 'umap', 'nmi', 'mutual information', 'silhouette', 'bic', 'bootstrap', 'syntetos', 'boylan']
export const BANNED_SPELLINGS = ['color', 'behavior', 'optimize', 'center', 'analyze', 'minimize', 'maximize', 'catalog', 'modeling', 'labeled']

/** Visible surface strings, collected for the test. */
export const SURFACE: string[] = [
  ...Object.values(TYPE_NAMES),
  ...Object.values(FINGERPRINT),
  ...Object.values(COLOUR_BY),
  ...Object.values(FEATURE_NAMES),
  ...Object.values(POLICY_NAMES),
]
