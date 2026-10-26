import type { Point, RunState } from '../api'
import { regionPastel } from '../format'
import { FactoryGlyph, HqGlyph, WarehouseGlyph } from './icons'

export type Selection =
  | { kind: 'hq' }
  | { kind: 'region'; region: number }
  | { kind: 'factory'; region: number }
  | { kind: 'warehouse'; region: number }
  | null

interface Props {
  state: RunState
  selection: Selection
  onSelect: (s: Selection) => void
}

// Gentle arc between two sites; trucks travel along it.
function control(a: Point, b: Point): Point {
  const mx = (a[0] + b[0]) / 2
  const my = (a[1] + b[1]) / 2
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  return [mx - dy * 0.18, my + dx * 0.18]
}
function along(a: Point, b: Point, t: number): Point {
  const c = control(a, b)
  const u = 1 - t
  return [u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]]
}
const arc = (a: Point, b: Point) => {
  const c = control(a, b)
  return `M${a[0]},${a[1]} Q${c[0]},${c[1]} ${b[0]},${b[1]}`
}
const polygon = (pts: Point[]) => 'M' + pts.map((p) => p.join(',')).join(' L') + ' Z'

function Badge({ at, fill, dashed, selected, label, onClick, children }: {
  at: Point; fill: string; dashed?: boolean; selected?: boolean; label: string; onClick: () => void; children: React.ReactNode
}) {
  return (
    <g
      className={`site${selected ? ' selected' : ''}`} transform={`translate(${at[0]},${at[1]})`}
      role="button" tabIndex={0} aria-label={label}
      onClick={(e) => { e.stopPropagation(); onClick() }}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick() } }}
    >
      <rect x={-18} y={-18} width={36} height={36} rx={10} fill="rgba(30,27,46,.18)" transform="translate(3,4)" />
      <rect
        className="site-ring" x={-18} y={-18} width={36} height={36} rx={10} fill={fill}
        stroke="#1e1b2e" strokeWidth={2.4} strokeDasharray={dashed ? '5 4' : undefined}
      />
      <g style={{ color: fill === '#ffffff' ? '#1e1b2e' : '#ffffff' }} opacity={dashed ? 0.45 : 1}>{children}</g>
    </g>
  )
}

export default function MapView({ state, selection, onSelect }: Props) {
  const { scenario, time } = state
  const regions = scenario.regions.filter((r) => r.map)
  const site = (region: number, kind: 'factory' | 'warehouse'): Point => {
    const m = scenario.regions.find((r) => r.id === region)!.map!
    return kind === 'factory' ? m.factory : m.warehouse
  }
  const factories = new Map(state.factories.map((f) => [f.region, f]))
  const warehouses = new Map(state.warehouses.map((w) => [w.region, w]))
  const buildable = (id: number) => scenario.rules.buildable_regions == null || scenario.rules.buildable_regions.includes(id)
  const activeRoutes = state.routes.filter((r) => r.quantity > 0 && factories.has(r.factory) && warehouses.has(r.warehouse))
  const sel = (kind: string, region?: number) =>
    selection != null && selection.kind === kind && (region == null || ('region' in selection && selection.region === region))

  return (
    <svg className="map" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid meet" onClick={() => onSelect(null)}
      role="img" aria-label={`Map of ${scenario.name}`}>
      <defs>
        <pattern id="waves" width="90" height="46" patternUnits="userSpaceOnUse">
          <path d="M8 22 q8 -6 16 0 t16 0" fill="none" stroke="#b9ddee" strokeWidth="2" strokeLinecap="round" />
          <path d="M53 42 q8 -6 16 0 t16 0" fill="none" stroke="#b9ddee" strokeWidth="2" strokeLinecap="round" />
        </pattern>
      </defs>
      <rect width="1000" height="700" fill="url(#waves)" />

      {/* soft offset shadow, then the regions */}
      <g transform="translate(5,7)" opacity=".16">
        {regions.map((r) => <path key={r.id} d={polygon(r.map!.outline)} fill="#1e1b2e" />)}
      </g>
      {regions.map((r) => (
        <path
          key={r.id} className="region-shape" d={polygon(r.map!.outline)} fill={regionPastel(r.id)}
          stroke="#1e1b2e" strokeWidth={sel('region', r.id) ? 4 : 2.6} strokeLinejoin="round"
          onClick={(e) => { e.stopPropagation(); onSelect({ kind: 'region', region: r.id }) }}
        />
      ))}
      {regions.map((r) => (
        <text key={r.id} className="region-label" x={r.map!.label[0]} y={r.map!.label[1]} textAnchor="middle">{r.name}</text>
      ))}

      {/* replenishment lanes in use */}
      {activeRoutes.map((r) => (
        <path key={`${r.factory}-${r.warehouse}`} d={arc(site(r.factory, 'factory'), site(r.warehouse, 'warehouse'))}
          fill="none" stroke="#1e1b2e" strokeWidth={2.2} strokeDasharray="1 7" strokeLinecap="round" opacity={0.7} />
      ))}

      {/* shipments on the road */}
      {state.warehouses.flatMap((w) => w.shipments.map((s, i) => {
        const t = Math.max(0, Math.min(1, (time - s.depart) / Math.max(s.arrive - s.depart, 1e-6)))
        const [x, y] = along(site(s.factory, 'factory'), site(w.region, 'warehouse'), t)
        return s.mode === 'truck' ? (
          <g key={`${w.region}-${i}`} transform={`translate(${x},${y})`} pointerEvents="none">
            <rect x={-13} y={-9} width={26} height={14} rx={4} fill="#fff" stroke="#1e1b2e" strokeWidth={2} />
            <circle cx={-6} cy={7} r={2.8} fill="#1e1b2e" /><circle cx={6} cy={7} r={2.8} fill="#1e1b2e" />
          </g>
        ) : (
          <g key={`${w.region}-${i}`} transform={`translate(${x},${y})`} pointerEvents="none">
            <rect x={-9} y={-6} width={18} height={12} rx={2} fill="#ead77a" stroke="#1e1b2e" strokeWidth={1.8} />
            <path d="M-9 -6 L0 1 L9 -6" fill="none" stroke="#1e1b2e" strokeWidth={1.5} />
          </g>
        )
      }))}

      {/* headquarters */}
      {regions.filter((r) => r.map!.headquarters).map((r) => (
        <Badge key="hq" at={r.map!.headquarters!} fill="#ffffff" selected={sel('hq')} label="Headquarters" onClick={() => onSelect({ kind: 'hq' })}>
          <HqGlyph />
        </Badge>
      ))}

      {/* factories */}
      {state.factories.map((f) => {
        const at = site(f.region, 'factory')
        const progress = f.batch ? Math.max(0, Math.min(1, (time - f.batch.start) / (f.batch.end - f.batch.start))) : null
        return (
          <g key={`f${f.region}`}>
            <Badge at={at} fill="#ffffff" dashed={!f.operational} selected={sel('factory', f.region)}
              label={`Factory in ${scenario.regions.find((r) => r.id === f.region)?.name}`}
              onClick={() => onSelect({ kind: 'factory', region: f.region })}>
              <FactoryGlyph />
            </Badge>
            {progress != null && (
              <g transform={`translate(${at[0] - 18},${at[1] + 24})`} pointerEvents="none">
                <rect width={36} height={6} rx={3} fill="#fff" stroke="#1e1b2e" strokeWidth={1.2} />
                <rect width={36 * progress} height={6} rx={3} fill="#4338ca" />
              </g>
            )}
            {!f.operational && f.online_at != null && (
              <text className="map-badge" x={at[0]} y={at[1] + 34} textAnchor="middle">ready day {Math.ceil(f.online_at)}</text>
            )}
          </g>
        )
      })}

      {/* warehouses */}
      {state.warehouses.map((w) => {
        const at = site(w.region, 'warehouse')
        return (
          <g key={`w${w.region}`}>
            <Badge at={at} fill={w.operational ? '#4338ca' : '#ffffff'} dashed={!w.operational} selected={sel('warehouse', w.region)}
              label={`Warehouse in ${scenario.regions.find((r) => r.id === w.region)?.name}`}
              onClick={() => onSelect({ kind: 'warehouse', region: w.region })}>
              <WarehouseGlyph />
            </Badge>
            {w.operational ? (
              <g transform={`translate(${at[0]},${at[1] + 30})`} pointerEvents="none">
                <rect x={-24} y={-9} width={48} height={18} rx={9} fill="#fff" stroke="#1e1b2e" strokeWidth={1.5} />
                <text className="map-badge" y={4} textAnchor="middle">{w.on_hand.toLocaleString('en-GB')}</text>
              </g>
            ) : w.online_at != null && (
              <text className="map-badge" x={at[0]} y={at[1] + 34} textAnchor="middle">ready day {Math.ceil(w.online_at)}</text>
            )}
          </g>
        )
      })}

      {/* empty sites in the selected region */}
      {selection && selection.kind === 'region' && buildable(selection.region) && (['factory', 'warehouse'] as const).map((kind) => {
        const exists = kind === 'factory' ? factories.has(selection.region) : warehouses.has(selection.region)
        if (exists) return null
        const at = site(selection.region, kind)
        return (
          <g key={kind} className="ghost" transform={`translate(${at[0]},${at[1]})`} pointerEvents="none">
            <rect x={-18} y={-18} width={36} height={36} rx={10} fill="rgba(255,255,255,.7)" stroke="#1e1b2e" strokeWidth={2} strokeDasharray="4 4" />
            <path d="M0 -8v16M-8 0h16" stroke="#1e1b2e" strokeWidth={2.4} strokeLinecap="round" />
          </g>
        )
      })}
    </svg>
  )
}
