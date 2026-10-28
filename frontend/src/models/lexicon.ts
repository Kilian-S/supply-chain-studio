/* The single source of visible wording on the model pages. Rules (checked by lexicon.test.ts):
 * plain words on the surface (product, group, stocking rule, test year), percentages for people,
 * technical terms only inside "How it works" drawers, British English, no em or en dashes. */

export const pct = (x: number, decimals = 0) => `${(x * 100).toFixed(decimals)}%`
export const money = (x: number) => (Math.abs(x) >= 1e6 ? `$${(x / 1e6).toFixed(2)}M` : `$${Math.round(x).toLocaleString('en-GB')}`)

// Hidden demand types as the generator defines them, in plain words.
export const TYPE_NAMES: Record<string, string> = {
  steady: 'Steady',
  seasonal: 'Seasonal',
  trending: 'Growing',
  declining: 'Fading',
  intermittent: 'Rare',
  lumpy: 'Bulk orders',
}

// The seven parts of a product's fingerprint.
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
  cv2: 'Swings in sales',
  adi: 'Gap between sales',
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
  cluster: 'Group',
  true_type: 'True type',
  category: 'Category',
}

export const STOCKING = (mode: string, service: number) =>
  `${mode === 'central' ? 'Central warehouse only' : 'Every regional warehouse'}, ${pct(service, service > 0.99 ? 1 : 0)} available`

export const POLICY_NAMES: Record<string, string> = {
  uniform: 'One rule for every product',
  all_central: 'Everything in the central warehouse',
  abc: 'Textbook rule: ABC (3 classes)',
  abc_xyz: 'Textbook rule: ABC/XYZ (9 classes)',
  sbc: 'Demand classes (4 classes)',
  true_type: 'Hidden true types (6, for reference)',
  per_sku: 'One rule per product (300 rules)',
}

// Words that must never appear on the visible surface outside a drawer (checked by the test).
export const BANNED_ON_SURFACE = ['sku', 'cluster', 'ari', 'silhouette', 'bic', 'umap', 'held-out', 'policy', 'z-score']
export const BANNED_SPELLINGS = ['color', 'behavior', 'optimize', 'center', 'analyze', 'minimize', 'maximize', 'catalog', 'modeling', 'labeled']

/** Visible surface strings, collected for the test. */
export const SURFACE: string[] = [
  ...Object.values(TYPE_NAMES),
  ...Object.values(FINGERPRINT).flatMap((f) => [f.name, f.text]),
  ...Object.values(COLOUR_BY),
  ...Object.values(FEATURE_NAMES),
  ...Object.values(POLICY_NAMES),
]
