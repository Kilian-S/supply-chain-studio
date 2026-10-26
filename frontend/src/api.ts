// Typed client for the FastAPI backend (backend/src/scgame/api.py).

export type Point = [number, number]
export type Mode = 'truck' | 'mail'
export type Policy = 'nearest' | 'soonest'

export interface RegionMap {
  outline: Point[]
  label: Point
  factory: Point
  warehouse: Point
  headquarters: Point | null
}

export interface Region {
  id: number
  name: string
  landmass: string
  map: RegionMap | null
}

export interface LaneTerms {
  truck_cost: number
  truck_days: number
  mail_cost_per_drum: number
  mail_days: number
}

export interface Economics {
  price_per_drum: number
  batch_fixed_cost: number
  unit_cost: number
  holding_cost_per_drum_year: number
  interest_rate_annual: number
  capacity_cost_per_unit: number
  factory_build_cost: number
  factory_build_days: number
  capacity_build_days: number
  warehouse_build_cost: number
  warehouse_build_days: number
  truck_capacity: number
}

export interface Rules {
  lost_sale_window_days: number
  allowed_policies: Policy[]
  buildable_regions: number[] | null
  allow_capacity_additions: boolean
}

export interface ScenarioInfo {
  id: string
  name: string
  description: string
  regions: Region[]
  lanes: Record<'same_region' | 'same_landmass' | 'cross_water', LaneTerms>
  economics: Economics
  rules: Rules
  timeline: { start_day: number; end_day: number }
}

export interface Batch {
  warehouse: number
  quantity: number
  mode: Mode
  start: number
  end: number
}

export interface Factory {
  region: number
  capacity: number
  online_at: number | null
  operational: boolean
  pending_capacity: { online_at: number; amount: number }[]
  batch: Batch | null
}

export interface Shipment {
  factory: number
  quantity: number
  mode: Mode
  depart: number
  arrive: number
}

export interface Warehouse {
  region: number
  online_at: number | null
  operational: boolean
  on_hand: number
  en_route_truck: number
  en_route_mail: number
  shipments: Shipment[]
  serves: number[]
}

export interface Route {
  factory: number
  warehouse: number
  order_point: number
  quantity: number
  mode: Mode
  priority: number
}

// Keys of per-region maps arrive as strings in JSON.
export type ByRegion = Record<string, number>

export interface DayRecord {
  day: number
  cash: number
  revenue: number
  production_cost: number
  truck_cost: number
  mail_cost: number
  fulfilment_cost: number
  holding_cost: number
  capex: number
  interest: number
  demand: ByRegion
  fulfilled: ByRegion
  lost: ByRegion
  shipped_from: ByRegion
  on_hand: ByRegion
  en_route_truck: ByRegion
  en_route_mail: ByRegion
  wip: ByRegion
  capacity: ByRegion
  batches_started: number
}

export interface Summary {
  final_cash: number
  demand: number
  fulfilled: number
  lost: number
  fill_rate: number
  leftover_inventory: number
  days_played: number
}

export interface RunState {
  run_id: string
  scenario: ScenarioInfo
  day: number
  time: number
  finished: boolean
  cash: number
  starting_cash: number
  policy: Policy
  factories: Factory[]
  warehouses: Warehouse[]
  routes: Route[]
  last_record: DayRecord | null
  summary: Summary
}

export interface ScenarioListItem {
  id: string
  name: string
  description: string
  regions: number
  start_day: number
  end_day: number
}

export interface LoggedDecision {
  day: number
  time: number
  kind: string
  [key: string]: unknown
}

export interface DemandHistory {
  days: number[]
  demand: Record<string, number[]>
  lost: Record<string, (number | null)[]>
}

export interface CashStatement {
  starting_cash: number
  sources: Record<string, number>
  uses: Record<string, number>
  cash: number
}

export type Decision =
  | { kind: 'set_route'; factory: number; warehouse: number; order_point?: number; quantity?: number; mode?: Mode; priority?: number }
  | { kind: 'set_policy'; policy: Policy }
  | { kind: 'set_capacity'; region: number; total: number }
  | { kind: 'build_warehouse'; region: number }
  | { kind: 'set_service'; region: number; warehouse: number; allowed: boolean }

export class ApiError extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).detail ?? detail
    } catch {
      // not JSON
    }
    throw new ApiError(String(detail))
  }
  return res.json() as Promise<T>
}

const post = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const api = {
  scenarios: () => call<ScenarioListItem[]>('/api/scenarios'),
  newRun: (scenario_id: string, seed?: number) => call<RunState>('/api/runs', post({ scenario_id, seed })),
  step: (runId: string, days: number) => call<RunState>(`/api/runs/${runId}/step`, post({ days })),
  runTo: (runId: string, until_day: number) => call<RunState>(`/api/runs/${runId}/step`, post({ until_day })),
  decide: (runId: string, decision: Decision) => call<RunState>(`/api/runs/${runId}/decisions`, post(decision)),
  records: (runId: string, sinceDay = 0) => call<DayRecord[]>(`/api/runs/${runId}/records?since_day=${sinceDay}`),
  log: (runId: string) => call<LoggedDecision[]>(`/api/runs/${runId}/log`),
  demand: (runId: string) => call<DemandHistory>(`/api/runs/${runId}/demand`),
  cashStatement: (runId: string) => call<CashStatement>(`/api/runs/${runId}/cash-statement`),
}
