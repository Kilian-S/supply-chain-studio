import { describe, expect, it } from 'vitest'
import { WORDING } from './affinity/copy'
import { BANNED_ON_SURFACE, BANNED_SPELLINGS, SURFACE } from './lexicon'
import { COPY } from './sku/copy'

const visible = [...SURFACE, ...COPY.surface]

describe('model page wording', () => {
  it('keeps jargon off the surface', () => {
    for (const s of visible) for (const w of BANNED_ON_SURFACE) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}`))
  })
  it('uses British spelling', () => {
    for (const s of visible) for (const w of BANNED_SPELLINGS) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}`))
  })
  it('has no em or en dashes', () => {
    for (const s of visible) { expect(s, s).not.toContain('—'); expect(s, s).not.toContain('–') }
  })
})

// The Bought together page: every visible string, surface and drawers. Formulas (\( \) and \[ \])
// are checked for dashes only; the prose around them for everything.
const prose = (s: string) => s.replace(/\\\(.+?\\\)|\\\[.+?\\\]/gs, ' ')
const surface = WORDING.surface.map(prose)
const everything = [...WORDING.surface, ...WORDING.drawer]

describe('bought together wording', () => {
  it('collects the strings', () => {
    expect(WORDING.surface.length).toBeGreaterThan(40)
    expect(WORDING.drawer.length).toBeGreaterThan(40)
  })
  it('keeps jargon off the surface', () => {
    for (const s of surface) for (const w of BANNED_ON_SURFACE) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}`))
  })
  it('uses British spelling', () => {
    for (const s of everything.map(prose)) for (const w of BANNED_SPELLINGS) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}`))
  })
  it('has no em or en dashes', () => {
    for (const s of everything) { expect(s, s).not.toContain('—'); expect(s, s).not.toContain('–') }
  })
  it('has no colons in prose', () => {
    for (const s of everything.map(prose)) expect(s, s).not.toContain(':')
  })
  it('says order, never basket', () => {
    for (const s of everything.map(prose)) expect(s.toLowerCase(), s).not.toMatch(/basket/)
  })
  it('has no formula left unclosed', () => {
    for (const s of everything) expect(prose(s), s).not.toMatch(/\\[([]/)
  })
})
