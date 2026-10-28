import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/* A Projector-style 3D point cloud. Built to behave the same in Safari and Chrome:
 * the canvas is absolutely positioned (never feeds back into layout), points are drawn by a small
 * shader (no textures), picking is done in screen space, and a failed WebGL start shows a message. */

export interface ProjectorProps {
  coords: number[][]
  colors: string[]
  keys: string[] // colour key per point, used for the group labels
  labelFor: Record<string, string>
  tooltip: (i: number) => string
  selected: number | null
  neighbours: number[]
  isolate: string | null
  autoRotate: boolean
  onSelect: (i: number | null) => void
}

const VERT = `
attribute vec3 aColor;
attribute float aAlpha;
uniform float uSize;
uniform float uScale;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = max(2.0, uSize * uScale / -mv.z);
  gl_Position = projectionMatrix * mv;
  vColor = aColor;
  vAlpha = aAlpha;
}`
const FRAG = `
precision mediump float;
varying vec3 vColor;
varying float vAlpha;
void main() {
  vec2 c = gl_PointCoord - vec2(0.5);
  float d = length(c);
  if (d > 0.5) discard;
  vec3 col = d > 0.38 ? vColor * 0.72 : vColor;
  gl_FragColor = vec4(col, vAlpha * (1.0 - smoothstep(0.46, 0.5, d)));
}`

// Centre on the core of the cloud and scale so its 10-90% band spans about two units; points far
// outside are pulled in on a log curve so a distant group stays visible at the edge.
function normalise(coords: number[][]): Float32Array {
  const n = coords.length
  const out = new Float32Array(n * 3)
  const q = (xs: number[], p: number) => xs[Math.min(xs.length - 1, Math.max(0, Math.round(p * (xs.length - 1))))]
  const lo: number[] = [], hi: number[] = []
  let span = 1e-9
  for (let d = 0; d < 3; d++) {
    const xs = coords.map((c) => c[d] ?? 0).sort((a, b) => a - b)
    lo.push(q(xs, 0.1)); hi.push(q(xs, 0.9))
    span = Math.max(span, hi[d] - lo[d])
  }
  const core = coords.filter((c) => [0, 1, 2].every((d) => (c[d] ?? 0) >= lo[d] && (c[d] ?? 0) <= hi[d]))
  const base = core.length ? core : coords
  const centre = [0, 1, 2].map((d) => base.reduce((a, c) => a + (c[d] ?? 0), 0) / base.length)
  const squash = (u: number) => (Math.abs(u) <= 1.2 ? u : Math.sign(u) * (1.2 + 0.6 * Math.log1p((Math.abs(u) - 1.2) / 0.6)))
  coords.forEach((c, i) => { for (let d = 0; d < 3; d++) out[i * 3 + d] = squash((((c[d] ?? 0) - centre[d]) / span) * 2) })
  return out
}

const rgb = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
const HOME = new THREE.Vector3(2.1, 1.3, 2.5)

export default function Projector(props: ProjectorProps) {
  const host = useRef<HTMLDivElement>(null)
  const labelLayer = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [tip, setTip] = useState<{ i: number; x: number; y: number } | null>(null)
  const live = useRef(props)
  live.current = props
  const world = useRef<{
    renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; controls: OrbitControls
    points: THREE.Points; marker: THREE.Points; links: THREE.LineSegments
    pos: Float32Array; centroids: { key: string; p: THREE.Vector3; el: HTMLDivElement }[]
  } | null>(null)

  // Scene, once.
  useEffect(() => {
    const el = host.current!
    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    } catch (e) {
      setFailed(`This browser could not start 3D graphics (${(e as Error).message}).`)
      return
    }
    renderer.setClearColor('#fbfbfd', 1)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    el.prepend(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100)
    camera.position.copy(HOME)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.autoRotateSpeed = 0.7
    controls.minDistance = 0.6
    controls.maxDistance = 8

    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: false,
      uniforms: { uSize: { value: 0.06 }, uScale: { value: 300 } },
    })
    const points = new THREE.Points(new THREE.BufferGeometry(), mat)
    scene.add(points)
    const markerGeo = new THREE.BufferGeometry()
    markerGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
    markerGeo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array([0.12, 0.11, 0.18]), 3))
    markerGeo.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array([1]), 1))
    const markerMat = mat.clone()
    markerMat.uniforms.uSize.value = 0.11
    markerMat.depthTest = false
    const marker = new THREE.Points(markerGeo, markerMat)
    marker.visible = false
    marker.renderOrder = 2
    scene.add(marker)
    const links = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#1e1b2e', transparent: true, opacity: 0.6 }))
    scene.add(links)
    const grid = new THREE.GridHelper(4, 8, '#dcdbe6', '#ececf3')
    grid.position.y = -1.3
    scene.add(grid)

    const resize = () => {
      const w = Math.max(el.clientWidth, 1), h = Math.max(el.clientHeight, 1)
      renderer.setSize(w, h, false) // never touch the canvas CSS size: no layout feedback
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      mat.uniforms.uScale.value = h * renderer.getPixelRatio() * 0.5
      markerMat.uniforms.uScale.value = mat.uniforms.uScale.value
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    const onContextLost = (e: Event) => { e.preventDefault(); setFailed('The graphics context was lost. Reload the page to restore the map.') }
    renderer.domElement.addEventListener('webglcontextlost', onContextLost)

    // Screen-space picking: nearest projected point within 14 px.
    const v = new THREE.Vector3()
    const pick = (cx: number, cy: number) => {
      const w = world.current
      if (!w) return null
      const r = el.getBoundingClientRect()
      let best = -1, bestD = 14 * 14
      for (let i = 0; i < w.pos.length / 3; i++) {
        v.set(w.pos[i * 3], w.pos[i * 3 + 1], w.pos[i * 3 + 2]).project(camera)
        if (v.z > 1) continue
        const dx = (v.x * 0.5 + 0.5) * r.width - (cx - r.left)
        const dy = (-v.y * 0.5 + 0.5) * r.height - (cy - r.top)
        const d = dx * dx + dy * dy
        if (d < bestD) { bestD = d; best = i }
      }
      return best < 0 ? null : best
    }
    let down = { x: 0, y: 0 }
    const onMove = (e: PointerEvent) => {
      const i = pick(e.clientX, e.clientY)
      const r = el.getBoundingClientRect()
      setTip(i == null ? null : { i, x: e.clientX - r.left, y: e.clientY - r.top })
      renderer.domElement.style.cursor = i == null ? 'grab' : 'pointer'
    }
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY } }
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return
      live.current.onSelect(pick(e.clientX, e.clientY))
    }
    const onLeave = () => setTip(null)
    renderer.domElement.addEventListener('pointermove', onMove)
    renderer.domElement.addEventListener('pointerdown', onDown)
    renderer.domElement.addEventListener('pointerup', onUp)
    renderer.domElement.addEventListener('pointerleave', onLeave)

    world.current = { renderer, camera, controls, points, marker, links, pos: new Float32Array(0), centroids: [] }

    let raf = 0
    const loop = () => {
      controls.autoRotate = live.current.autoRotate && live.current.selected == null
      controls.update()
      renderer.render(scene, camera)
      const w = world.current
      if (w) {
        // Project each group label, then push overlapping labels apart vertically.
        const r = el.getBoundingClientRect()
        const placed: { x: number; y: number; w: number; h: number }[] = []
        const items = w.centroids.map((c) => {
          v.copy(c.p).project(camera)
          return { c, x: (v.x * 0.5 + 0.5) * r.width, y: (-v.y * 0.5 + 0.5) * r.height - 18, hidden: v.z > 1 || (live.current.isolate != null && live.current.isolate !== c.key) }
        }).sort((a, b) => a.y - b.y)
        for (const it of items) {
          it.c.el.style.display = it.hidden ? 'none' : 'block'
          if (it.hidden) continue
          const wdt = it.c.el.offsetWidth, hgt = it.c.el.offsetHeight
          let y = it.y
          for (let guard = 0; guard < 12; guard++) {
            const hit = placed.find((p) => Math.abs(p.x - it.x) < (p.w + wdt) / 2 + 4 && Math.abs(p.y - y) < (p.h + hgt) / 2 + 2)
            if (!hit) break
            y = hit.y + (hit.h + hgt) / 2 + 3
          }
          placed.push({ x: it.x, y, w: wdt, h: hgt })
          it.c.el.style.transform = `translate(${it.x - wdt / 2}px, ${y - hgt / 2}px)`
        }
      }
      raf = requestAnimationFrame(loop)
    }
    loop()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost)
      renderer.dispose()
      renderer.domElement.remove()
      world.current = null
    }
  }, [])

  // Points, colours, isolation and group labels.
  const { coords, colors, keys, labelFor, isolate } = props
  useEffect(() => {
    const w = world.current
    if (!w) return
    const pos = normalise(coords)
    w.pos = pos
    const col = new Float32Array(coords.length * 3)
    const alpha = new Float32Array(coords.length)
    colors.forEach((hex, i) => {
      col.set(rgb(hex), i * 3)
      alpha[i] = isolate == null || keys[i] === isolate ? 0.95 : 0.12
    })
    const g = w.points.geometry
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3))
    g.setAttribute('aAlpha', new THREE.BufferAttribute(alpha, 1))
    g.computeBoundingSphere()
    const layer = labelLayer.current!
    layer.replaceChildren()
    const sums = new Map<string, { s: THREE.Vector3; n: number }>()
    keys.forEach((k, i) => {
      const e = sums.get(k) ?? { s: new THREE.Vector3(), n: 0 }
      e.s.add(new THREE.Vector3(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]))
      e.n += 1
      sums.set(k, e)
    })
    w.centroids = [...sums.entries()].map(([key, { s, n }]) => {
      const div = document.createElement('div')
      div.className = 'proj-label'
      div.textContent = labelFor[key] ?? key
      layer.appendChild(div)
      return { key, p: s.divideScalar(n), el: div }
    })
  }, [coords, colors, keys, labelFor, isolate])

  // Selection marker and lines to the nearest neighbours.
  const { selected, neighbours } = props
  useEffect(() => {
    const w = world.current
    if (!w || !w.pos.length) return
    if (selected == null) {
      w.marker.visible = false
      w.links.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3))
      return
    }
    const p = w.pos.subarray(selected * 3, selected * 3 + 3)
    ;(w.marker.geometry.getAttribute('position') as THREE.BufferAttribute).set(p)
    w.marker.geometry.getAttribute('position').needsUpdate = true
    w.marker.visible = true
    const seg = new Float32Array(neighbours.length * 6)
    neighbours.forEach((j, k) => { seg.set(p, k * 6); seg.set(w.pos.subarray(j * 3, j * 3 + 3), k * 6 + 3) })
    w.links.geometry.setAttribute('position', new THREE.BufferAttribute(seg, 3))
  }, [selected, neighbours, coords])

  const zoom = (f: number) => {
    const w = world.current
    if (w) w.camera.position.sub(w.controls.target).multiplyScalar(f).add(w.controls.target)
  }
  const reset = () => {
    const w = world.current
    if (!w) return
    w.controls.target.set(0, 0, 0)
    w.camera.position.copy(HOME)
  }

  return (
    <div className="proj" ref={host}>
      <div ref={labelLayer} aria-hidden="true" />
      {failed && <div className="proj-fallback">{failed}</div>}
      {tip && !failed && (
        <div className="proj-tip" style={{ left: Math.min(tip.x + 14, (host.current?.clientWidth ?? 600) - 220), top: tip.y + 14 }}>{props.tooltip(tip.i)}</div>
      )}
      {!failed && (
        <div className="proj-tools">
          <button aria-label="Zoom in" onClick={() => zoom(0.8)}>+</button>
          <button aria-label="Zoom out" onClick={() => zoom(1.25)}>&minus;</button>
          <button aria-label="Reset view" title="Reset view" onClick={reset}>&#8634;</button>
        </div>
      )}
    </div>
  )
}
