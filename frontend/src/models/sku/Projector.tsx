import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface Props {
  coords: number[][] // n x 3 (or n x 2)
  colors: string[]
  labels: string[]
  selected: number | null
  neighbours: number[]
  dimmed: boolean[] | null // points to fade (e.g. other clusters when one is isolated)
  flat: boolean
  autoRotate: boolean
  onSelect: (i: number | null) => void
}

function dotTexture() {
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const g = c.getContext('2d')!
  g.beginPath()
  g.arc(32, 32, 28, 0, Math.PI * 2)
  g.fillStyle = '#ffffff'
  g.fill()
  g.lineWidth = 5
  g.strokeStyle = 'rgba(30,27,46,0.55)'
  g.stroke()
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

// Centre on the core of the cloud (points inside the 10-90% band on every axis) and scale so that
// band spans about 2 units on its widest axis. Points far outside are pulled in on a log curve, so a
// distant group stays visible at the edge instead of shrinking everything else to a dot.
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

export default function Projector({ coords, colors, labels, selected, neighbours, dimmed, flat, autoRotate, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const three = useRef<{
    renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls
    points: THREE.Points; marker: THREE.Points; links: THREE.LineSegments; positions: Float32Array; raf: number
  } | null>(null)
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null)
  const latest = useRef({ labels, onSelect })
  latest.current = { labels, onSelect }

  // Set up the scene once.
  useEffect(() => {
    const el = host.current!
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100)
    camera.position.set(2.0, 1.4, 2.4)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.autoRotateSpeed = 0.8

    const tex = dotTexture()
    const geo = new THREE.BufferGeometry()
    const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.085, map: tex, vertexColors: true, transparent: true, alphaTest: 0.3, sizeAttenuation: true }))
    scene.add(points)
    const markerGeo = new THREE.BufferGeometry()
    markerGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3), 3))
    const marker = new THREE.Points(markerGeo, new THREE.PointsMaterial({ size: 0.2, map: tex, color: '#1e1b2e', transparent: true, alphaTest: 0.3 }))
    marker.visible = false
    scene.add(marker)
    const links = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#1e1b2e', transparent: true, opacity: 0.55 }))
    scene.add(links)
    const axesGeo = new THREE.BufferGeometry()
    axesGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1.1, 0, 0, 1.1, 0, 0, 0, -1.1, 0, 0, 1.1, 0, 0, 0, -1.1, 0, 0, 1.1]), 3))
    const axes = new THREE.LineSegments(axesGeo, new THREE.LineBasicMaterial({ color: '#d6d3e2' }))
    scene.add(axes)

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(el)
    resize()

    const ray = new THREE.Raycaster()
    ray.params.Points = { threshold: 0.04 }
    const mouse = new THREE.Vector2()
    let down = { x: 0, y: 0 }
    const pick = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect()
      mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
      ray.setFromCamera(mouse, camera)
      const hit = ray.intersectObject(points)[0]
      return { hit: hit?.index ?? null, x: e.clientX - r.left, y: e.clientY - r.top }
    }
    const onMove = (e: PointerEvent) => {
      const p = pick(e)
      setHover(p.hit == null ? null : { i: p.hit, x: p.x, y: p.y })
      renderer.domElement.style.cursor = p.hit == null ? 'grab' : 'pointer'
    }
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY } }
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return // a drag, not a click
      latest.current.onSelect(pick(e).hit)
    }
    renderer.domElement.addEventListener('pointermove', onMove)
    renderer.domElement.addEventListener('pointerdown', onDown)
    renderer.domElement.addEventListener('pointerup', onUp)
    renderer.domElement.addEventListener('pointerleave', () => setHover(null))

    const state = { renderer, scene, camera, controls, points, marker, links, positions: new Float32Array(0), raf: 0 }
    const loop = () => {
      controls.update()
      renderer.render(scene, camera)
      state.raf = requestAnimationFrame(loop)
    }
    loop()
    three.current = state
    return () => {
      cancelAnimationFrame(state.raf)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      geo.dispose()
      tex.dispose()
      el.removeChild(renderer.domElement)
      three.current = null
    }
  }, [])

  // Points, colours and fading.
  useEffect(() => {
    const t = three.current
    if (!t) return
    const pos = normalise(flat ? coords.map((p) => [p[0], p[1], 0]) : coords)
    t.positions = pos
    const col = new Float32Array(coords.length * 3)
    const c = new THREE.Color()
    const paper = new THREE.Color('#f5f4f8')
    colors.forEach((hex, i) => {
      c.set(hex)
      if (dimmed && dimmed[i]) c.lerp(paper, 0.82)
      col.set([c.r, c.g, c.b], i * 3)
    })
    const g = t.points.geometry
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(col, 3))
    g.computeBoundingSphere()
  }, [coords, colors, dimmed, flat])

  // Camera mode.
  useEffect(() => {
    const t = three.current
    if (!t) return
    t.controls.enableRotate = !flat
    t.controls.autoRotate = autoRotate && !flat
    if (flat) {
      t.camera.position.set(0, 0, 3)
      t.controls.target.set(0, 0, 0)
    }
  }, [flat, autoRotate])

  // Selection marker and neighbour links.
  useEffect(() => {
    const t = three.current
    if (!t || !t.positions.length) return
    if (selected == null) {
      t.marker.visible = false
      t.links.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3))
      return
    }
    const p = t.positions.subarray(selected * 3, selected * 3 + 3)
    ;(t.marker.geometry.getAttribute('position') as THREE.BufferAttribute).set(p)
    t.marker.geometry.getAttribute('position').needsUpdate = true
    t.marker.visible = true
    const seg = new Float32Array(neighbours.length * 6)
    neighbours.forEach((j, k) => {
      seg.set(p, k * 6)
      seg.set(t.positions.subarray(j * 3, j * 3 + 3), k * 6 + 3)
    })
    t.links.geometry.setAttribute('position', new THREE.BufferAttribute(seg, 3))
  }, [selected, neighbours, coords, flat])

  return (
    <div className="projector" ref={host}>
      {hover && <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>{labels[hover.i]}</div>}
    </div>
  )
}
