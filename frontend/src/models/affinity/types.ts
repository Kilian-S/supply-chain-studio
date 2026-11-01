// Shapes of the prepared product affinity results (backend/src/scgame/ml/product_affinity.py).
import type { RuleKey } from './copy'

export interface PriceRow {
  fcs: number
  orders: number
  multi: number
  split: number
  load_share: number[]
  parcels: number
  cost: number
  split_share: number
  multi_split_share: number
}

export interface TruthCheck {
  precision: number
  recall: number
  bundles_exact: number
  pairs_together: number
  pairs_true: number
  ari: number
}

export interface AffinityOverview {
  model: { id: string; title: string }
  facts: {
    orders: number
    multi_share: number
    skus_per_order: number
    pairs_seen: number
    pairs_possible: number
    top200_count_true_share: number
    top200_lift_true_share: number
    basket_size_counts: number[]
    orders_by_year: number[]
    skus_per_order_all: number
    basket_size_counts_all: number[]
  }
  settings: {
    fdr_levels: number[]
    resolutions: number[]
    fcs: number[]
    slack: number
    parcel_fee: number
    per_kg: number
    central_region: number
  }
  grid: { fdr: number; resolution: number; edges: number; communities: number; alone: number; val_cost: number; val_split_share: number; truth: TruthCheck }[]
  chosen: { fdr: number; resolution: number; count_baseline_resolution: number }
  workload_note: { added_after_test_comparison: boolean; validation_without_workload_limit: Record<RuleKey, PriceRow[]> }
  quarter_note: {
    added_after_truth_check: boolean
    with_quarter: { val_cost: number; truth: TruthCheck }
    without_quarter: { fdr: number; val_cost: number; truth: TruthCheck }
  }
  truth: { bundles: number; skus_in_bundles: number; check: Record<'communities' | 'count_communities' | 'categories', TruthCheck> }
  results: Record<RuleKey, PriceRow[]>
  results_validation: Record<RuleKey, PriceRow[]>
  resolution_invariant?: boolean
  assignments: Record<RuleKey, Record<string, number[]>>
}

export interface AffinityNode {
  id: number
  name: string
  category: string
  baskets: number // orders holding the SKU, training set
  community: number
  alone: boolean // an unlinked SKU
  bundle: number // hidden bundle, 0 for none
  xyz: number[]
}

export interface AffinityGraph {
  nodes: AffinityNode[]
  communities: { id: number; name: string; size: number }[]
  edges: { a: number; b: number; count: number; lift: number; p: number }[] // a < b, node indices
}

export interface AffinityMatrix {
  count: number[][]
  lift: number[][]
  expected: number[][]
}
