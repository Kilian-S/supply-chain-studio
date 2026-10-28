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
export const FINGERPRINT: Record<string, { name: string; text: string }> = {
  volume: { name: 'How much it sells', text: 'Units a week and revenue' },
  variability: { name: 'How steady it is', text: 'How much sales jump around' },
  intermittency: { name: 'How often it sells', text: 'Share of weeks with no sale' },
  pattern: { name: 'Its shape over time', text: 'Seasonal peaks, growth or decline' },
  physical: { name: 'Size and weight', text: 'Litres and kilos per unit' },
  economic: { name: 'Price and margin', text: 'What it earns and what it costs to hold' },
  basket: { name: 'How it is bought', text: 'Alone, with other products, or in bulk' },
}

// Plain names for the 16 numbers (formal names and formulas stay in the drawers).
export const FEATURE_NAMES: Record<string, string> = {
  mean_weekly_units: 'Units a week',
  revenue: 'Revenue',
  cv2: 'CV² (swings in sales)',
  adi: 'ADI (gap between sales)',
  zero_share: 'Weeks without a sale',
  seasonality: 'Seasonal peak',
  trend: 'Growth',
  recent_share: 'Sales in the last year',
  volume_l: 'Volume',
  weight_kg: 'Weight',
  margin: 'Margin',
  price: 'Price',
  holding_to_margin: 'Storage cost against margin',
  solo_share: 'Bought alone',
  basket_size: 'Basket size',
  qty_per_line: 'Units per order line',
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
  abc_xyz: 'ABC/XYZ analysis (9 classes)',
  sbc: 'Demand classes by ADI and CV² (4)',
  true_type: 'Ground truth types (6, for reference)',
  per_sku: 'One policy per SKU (300)',
}

// Words that must never appear on the visible surface outside a drawer (checked by the test).
// Reading-list terms: allowed in drawers and GLOSSARY.md, not on the surface.
export const BANNED_ON_SURFACE = ['gmm', 'gaussian mixture', 'soft assignment', 'expectation', 'umap', 'nmi', 'mutual information', 'silhouette', 'bic', 'bootstrap', 'syntetos', 'boylan']
export const BANNED_SPELLINGS = ['color', 'behavior', 'optimize', 'center', 'analyze', 'minimize', 'maximize', 'catalog', 'modeling', 'labeled']

/** Visible surface strings, collected for the test. */
export const SURFACE: string[] = [
  ...Object.values(TYPE_NAMES),
  ...Object.values(FINGERPRINT).flatMap((f) => [f.name, f.text]),
  ...Object.values(COLOUR_BY),
  ...Object.values(FEATURE_NAMES),
  ...Object.values(POLICY_NAMES),
]
