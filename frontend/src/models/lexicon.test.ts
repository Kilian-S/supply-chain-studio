import { describe, expect, it } from 'vitest'
import { BANNED_ON_SURFACE, BANNED_SPELLINGS, SURFACE } from './lexicon'
import { COPY } from './sku/copy'

const visible = [...SURFACE, ...COPY.surface]

describe('model page wording', () => {
  it('keeps jargon off the surface', () => {
    for (const s of visible) for (const w of BANNED_ON_SURFACE) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}\\b`))
  })
  it('uses British spelling', () => {
    for (const s of visible) for (const w of BANNED_SPELLINGS) expect(s.toLowerCase(), s).not.toMatch(new RegExp(`\\b${w}`))
  })
  it('has no em or en dashes', () => {
    for (const s of visible) { expect(s, s).not.toContain('—'); expect(s, s).not.toContain('–') }
  })
})
