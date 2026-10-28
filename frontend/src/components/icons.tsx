// Inline stroke icons (no icon font, no emoji).

type P = { size?: number }

export const PlayIcon = ({ size = 16 }: P) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor" /></svg>
)
export const PauseIcon = ({ size = 16 }: P) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3v10M11 3v10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" /></svg>
)
export const StepIcon = ({ size = 16 }: P) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3v10l7-5z" fill="currentColor" /><path d="M12.5 3v10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
)
export const CloseIcon = ({ size = 16 }: P) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>
)
// Brand mark: one factory feeding two warehouses, on the coral brand tile.
export const LogoMark = ({ size = 30 }: P) => (
  <svg width={size} height={size} viewBox="0 0 30 30" aria-hidden="true">
    <rect x="1" y="1" width="28" height="28" rx="8" fill="#ff6b4a" />
    <path d="M10 20.5 L15 10.5 L20 20.5" fill="none" stroke="#1e1b2e" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    <rect x="12" y="7.5" width="6" height="6" rx="1.6" fill="#1e1b2e" />
    <circle cx="9.5" cy="21" r="3.2" fill="#fff" stroke="#1e1b2e" strokeWidth="2" />
    <circle cx="20.5" cy="21" r="3.2" fill="#fff" stroke="#1e1b2e" strokeWidth="2" />
  </svg>
)

// Map glyphs, drawn around (0, 0) inside a 32 x 32 badge.
export const FactoryGlyph = () => <path d="M-9 9V-3l6 4v-4l6 4V-7h5V9Z" fill="currentColor" />
export const WarehouseGlyph = () => <path d="M-9 9V-2L0-8l9 6V9H4V2h-8v7Z" fill="currentColor" />
export const HqGlyph = () => (
  <g fill="currentColor"><rect x="-9" y="-8" width="8" height="17" rx="1" /><rect x="1" y="-3" width="8" height="12" rx="1" /></g>
)
