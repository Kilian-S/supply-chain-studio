import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'

export default function Formula({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(tex, { throwOnError: false, displayMode: block }), [tex, block])
  return <span className={block ? 'formula block' : 'formula'} dangerouslySetInnerHTML={{ __html: html }} />
}
