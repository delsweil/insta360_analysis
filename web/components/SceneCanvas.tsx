'use client'

// A training sketch drawn as SVG in real metres. Used read-only for thumbnails and the shared view, and editable in the scene editor.
// Gestures (mouse, pen or touch, all through pointer events):
//   place tools (player, ball, material)  press = place, keep dragging to position, release = done
//   line tools, distance, field           press at the start, drag, release at the end (a plain tap makes nothing)
//   pen                                   press and draw
//   select                                press a shape to select and drag it; drag a round handle of the selected shape to change it;
//                                         press on empty space and drag to draw a selection rectangle; Shift/Ctrl-click adds or removes a shape
// Lines attach to the players at their ends, so moving a player carries its passes along.

import { memo, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  TEAM_COLORS, equipSize, ballRadius, symbolUnit, viewBoxOf, drawOrder, dimText, gridSizeText, textHeight,
  makeShape, isMeaningful, moveShapes, updateShape, shapesInRect, shapeBox, unionBox, scaleOf, equipDims, playerRadius, isLight, hitTest, handleAt, handlesOf, handlePatch, lineEndTarget, snapPt, snapStep, clampPt,
  type Scene, type SceneShape, type SceneBackground, type Tool, type Team, type EquipKind, type Pt,
} from '@/lib/scene'
import { arrowHead, blobPath, boxPath, curvePoints, decimate, hashStr, polyPath, smoothPath, trimPolyline, wavy, wobble, dist } from '@/lib/sketch'

export const INK = '#1b2a6b'
const PAPER = '#fbfbf8'
const DIM = '#475085'
const LINE_GREY = '#a9b3cf'
const HALO = '#e8780a'
export const POLE_COLORS = ['#f5c542', '#e94fae', '#49b05c', '#3b82e8', '#ef4444', '#ffffff']
const HAND = "var(--font-hand), 'Segoe Print', 'Bradley Hand', cursive"
const SANS = "var(--font-dm-sans), system-ui, sans-serif"

export type Look = 'sketch' | 'clean'

// ── Equipment symbols (local coordinates, opening of a goal faces up) ─────────

export function equipGlyph(kind: EquipKind, u: number, k: number, opts: { color?: string; num?: string; look?: Look } = {}) {
  const { w, d } = equipSize(kind, u)
  const sw = 2.4 * k
  const common = { stroke: INK, strokeWidth: sw, strokeLinejoin: 'round' as const, strokeLinecap: 'round' as const }
  switch (kind) {
    case 'cone':
      return <path d={`M0 ${-w / 2}L${w / 2} ${w / 2}L${-w / 2} ${w / 2}Z`} fill={opts.color ?? '#f59e0b'} {...common} />
    case 'colorpole':
      return (
        <g>
          <circle r={w / 2} fill={opts.color ?? '#f5c542'} {...common} />
          {opts.num ? <text y={0} textAnchor="middle" dominantBaseline="central" fontSize={w * 0.8} fontWeight={700} fill={!opts.color || isLight(opts.color) ? INK : '#fff'} style={{ fontFamily: SANS }}>{opts.num}</text> : null}
        </g>
      )
    case 'pole':
      return <g><circle r={w / 2} fill={opts.color ?? '#fff'} {...common} /><circle r={w * 0.12} fill={INK} /></g>
    case 'hurdle':
      return <g {...common} stroke={opts.color ?? INK} fill="none"><path d={`M${-w / 2} 0H${w / 2}`} strokeWidth={sw * 1.3} /><path d={`M${-w / 2} ${-d}V${d}M${w / 2} ${-d}V${d}`} /></g>
    case 'ring':
      return <circle r={w / 2} fill="none" {...common} stroke={opts.color ?? INK} strokeWidth={sw * 1.5} />
    case 'rebounder': {
      const hatch: string[] = []
      for (let i = -2; i <= 2; i++) hatch.push(`M${(i * w) / 5 - d / 2} ${d / 2}L${(i * w) / 5 + d / 2} ${-d / 2}`)
      return <g><rect x={-w / 2} y={-d / 2} width={w} height={d} fill={opts.color ?? '#fff'} {...common} /><path d={hatch.join('')} stroke={INK} strokeWidth={sw * 0.6} clipPath="none" opacity={0.5} fill="none" /></g>
    }
    case 'minigoal': case 'kleinfeldgoal': case 'goal': {
      const net: string[] = []
      const n = Math.max(2, Math.round(w / (kind === 'goal' ? 0.5 : kind === 'kleinfeldgoal' ? 0.45 : 0.3)))
      for (let i = 1; i < n; i++) net.push(`M${-w / 2 + (w * i) / n} ${-d / 2}V${d / 2}`)
      return (
        <g>
          <rect x={-w / 2} y={-d / 2} width={w} height={d} fill="#e9edf8" opacity={0.7} />
          <path d={net.join('')} stroke={LINE_GREY} strokeWidth={1 * k} fill="none" />
          <path d={`M${-w / 2} ${-d / 2}V${d / 2}H${w / 2}V${-d / 2}`} fill="none" {...common} stroke={opts.color ?? INK} strokeWidth={sw * 1.4} />
        </g>
      )
    }
  }
}

// ── Pitch markings (105 x 68, goals left and right) ───────────────────────────

const PITCH_D =
  'M0 0H105V68H0Z M52.5 0V68 M43.35 34a9.15 9.15 0 1 0 18.3 0a9.15 9.15 0 1 0-18.3 0 ' +
  'M0 13.84H16.5V54.16H0 M105 13.84H88.5V54.16H105 M0 24.84H5.5V43.16H0 M105 24.84H99.5V43.16H105 ' +
  'M16.5 26.69A9.15 9.15 0 0 1 16.5 41.31 M88.5 41.31A9.15 9.15 0 0 1 88.5 26.69 ' +
  'M0 30.34H-2V37.66H0 M105 30.34H107V37.66H105 M10.8 34a.2.2 0 1 0 .4 0a.2.2 0 1 0-.4 0 M93.8 34a.2.2 0 1 0 .4 0a.2.2 0 1 0-.4 0 M52.3 34a.2.2 0 1 0 .4 0a.2.2 0 1 0-.4 0'

const Background = memo(function Background({ bg, k, look, plain }: { bg: SceneBackground; k: number; look: Look; plain: boolean }) {
  const id = useId().replace(/:/g, '')
  const v = viewBoxOf(bg)
  const step = snapStep(bg)
  const seed = hashStr(bg.type + bg.width_m + bg.height_m)
  return (
    <g>
      <defs>
        <pattern id={`dots${id}`} x={-step / 2} y={-step / 2} width={step} height={step} patternUnits="userSpaceOnUse">
          <circle cx={step / 2} cy={step / 2} r={1.3 * k} fill="#d3d7e1" />
        </pattern>
        <clipPath id={`clip${id}`}><rect x={-3} y={0} width={55.5} height={68} /></clipPath>
      </defs>
      <rect x={v.x} y={v.y} width={v.w} height={v.h} fill={plain ? '#ffffff' : PAPER} />
      {!plain && <rect x={v.x} y={v.y} width={v.w} height={v.h} fill={`url(#dots${id})`} />}
      {bg.type === 'pitch' && <path d={PITCH_D} fill="none" stroke={LINE_GREY} strokeWidth={2 * k} strokeLinejoin="round" />}
      {bg.type === 'halfpitch' && (
        <g transform="translate(68 0) rotate(90)">
          <g clipPath={`url(#clip${id})`}><path d={PITCH_D} fill="none" stroke={LINE_GREY} strokeWidth={2 * k} strokeLinejoin="round" /></g>
        </g>
      )}
      {bg.type === 'free' && (
        <g>
          <path d={boxPath([0, 0], [bg.width_m, bg.height_m], seed, look === 'sketch' ? 1.4 * k : 0)} fill="none" stroke={INK} strokeWidth={2.6 * k} strokeDasharray={`${16 * k} ${10 * k}`} strokeLinecap="round" />
          <text x={bg.width_m / 2} y={-8 * k} textAnchor="middle" fontSize={15 * k} fill={DIM} style={{ fontFamily: look === 'sketch' ? HAND : SANS }} fontWeight={600}>{`${String(bg.width_m).replace('.', ',')} m`}</text>
          <text transform={`translate(${-8 * k} ${bg.height_m / 2}) rotate(-90)`} textAnchor="middle" fontSize={15 * k} fill={DIM} style={{ fontFamily: look === 'sketch' ? HAND : SANS }} fontWeight={600}>{`${String(bg.height_m).replace('.', ',')} m`}</text>
        </g>
      )}
    </g>
  )
})

// ── The canvas ────────────────────────────────────────────────────────────────

interface Props {
  scene: Scene
  look?: Look
  editable?: boolean
  tool?: Tool
  team?: Team
  equip?: EquipKind
  poleColor?: string
  /** No dot grid and a white page: for print. */
  plain?: boolean
  snap?: boolean
  selectedIds?: string[]
  onSelect?: (ids: string[]) => void
  /** Called once when a gesture ends with a change. */
  onCommit?: (shapes: SceneShape[]) => void
  /** The text tool was tapped: the editor asks for the text and adds the shape. */
  onRequestText?: (pos: Pt) => void
  className?: string
  style?: React.CSSProperties
  testId?: string
}

type Drag =
  | { kind: 'move'; ids: string[]; anchorId: string; start: Pt; before: SceneShape[]; created: boolean; moved: boolean }
  | { kind: 'marquee'; start: Pt; add: string[]; moved: boolean }
  | { kind: 'handle'; id: string; key: string; before: SceneShape[]; moved: boolean }
  | { kind: 'create'; id: string; before: SceneShape[]; tool: Tool }

const NONE: string[] = []

export default function SceneCanvas(props: Props) {
  const { scene, look = 'sketch', editable = false, tool = 'select', team = 'a', equip = 'cone', poleColor, plain = false, snap = false, selectedIds = NONE, onSelect, onCommit, onRequestText, className, style, testId } = props
  const bg = scene.background
  const vb = viewBoxOf(bg)
  const u = symbolUnit(bg)
  const svgRef = useRef<SVGSVGElement>(null)
  const [ppm, setPpm] = useState(10)
  const [draft, setDraft] = useState<SceneShape[] | null>(null)
  const [marquee, setMarquee] = useState<{ a: Pt; b: Pt } | null>(null)
  const dragRef = useRef<Drag | null>(null)
  const marqueeRef = useRef<{ a: Pt; b: Pt } | null>(null)
  const pointerId = useRef<number | null>(null)
  // k = metres per drawn pixel. Read-only miniatures scale their strokes and labels down with the picture; the editor never does.
  const kTouch = 1 / ppm
  const k = (editable || plain ? 1 : Math.min(1, Math.max(0.3, (vb.w * ppm) / 520))) / ppm
  const sketch = look === 'sketch'
  const amp = sketch ? 1.5 * k : 0

  useEffect(() => {
    const el = svgRef.current
    if (!el) return
    const measure = () => { const m = el.getScreenCTM(); const sc = m ? Math.abs(m.a) : 0; if (sc > 0) setPpm(sc) }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [vb.w])

  const shapes = draft ?? scene.shapes
  const players = useMemo(() => new Map(shapes.filter(s => s.type === 'player').map(s => [s.id, s])), [shapes])

  const toWorld = useCallback((e: { clientX: number; clientY: number }): Pt => {
    const el = svgRef.current!
    const m = el.getScreenCTM()
    if (!m) return [0, 0]
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse())
    return [p.x, p.y]
  }, [])

  function finish(commit: boolean, override?: SceneShape[]) {
    const d = dragRef.current
    dragRef.current = null
    pointerId.current = null
    if (!d) return
    if (d.kind === 'marquee') {
      const m = marqueeRef.current
      marqueeRef.current = null
      setMarquee(null)
      if (commit && d.moved && m) {
        const inside = shapesInRect(scene.shapes, m.a, m.b, bg)
        onSelect?.([...new Set([...d.add, ...inside])])
      }
      return
    }
    const cur = override ?? draft
    setDraft(null)
    if (!commit || !cur) return
    if (d.kind === 'create') {
      const s = cur.find(x => x.id === d.id)
      if (!s || !isMeaningful(s, 10 * kTouch)) { onSelect?.([]); return }
      onCommit?.(cur)
      onSelect?.([s.id])
      return
    }
    if ((d.kind === 'move' && (d.moved || d.created)) || (d.kind === 'handle' && d.moved)) onCommit?.(cur)
  }

  function onPointerDown(e: React.PointerEvent<SVGSVGElement>) {
    if (!editable) return
    if (pointerId.current !== null && pointerId.current !== e.pointerId) { finish(false); return } // a second finger cancels
    e.preventDefault()
    const touchy = e.pointerType === 'touch'
    const tol = (touchy ? 18 : 9) * kTouch
    const raw = toWorld(e)
    const p = clampPt(raw, bg)
    const sp = snap ? snapPt(p, snapStep(bg)) : p
    const base = scene.shapes

    if (tool === 'select') {
      const sel = selectedIds.length === 1 ? base.find(x => x.id === selectedIds[0]) : undefined
      if (sel) {
        const key = handleAt(sel, bg, p, tol * 1.2, base)
        if (key) {
          dragRef.current = { kind: 'handle', id: sel.id, key, before: base, moved: false }
          pointerId.current = e.pointerId; svgRef.current?.setPointerCapture(e.pointerId); return
        }
      }
      const id = hitTest(base, p, tol, bg)
      const extend = e.shiftKey || e.ctrlKey || e.metaKey
      if (id && extend) { onSelect?.(selectedIds.includes(id) ? selectedIds.filter(x => x !== id) : [...selectedIds, id]); return }
      if (id) {
        const ids = selectedIds.includes(id) ? selectedIds : [id]
        if (ids !== selectedIds) onSelect?.(ids)
        dragRef.current = { kind: 'move', ids, anchorId: id, start: p, before: base, created: false, moved: false }
      } else {
        const add = extend ? selectedIds : []
        if (!extend) onSelect?.([])
        dragRef.current = { kind: 'marquee', start: p, add, moved: false }
      }
      pointerId.current = e.pointerId; svgRef.current?.setPointerCapture(e.pointerId)
      return
    }
    if (tool === 'text') { onRequestText?.(sp); return }

    if (tool === 'player' || tool === 'ball' || tool === 'equip') {
      const s = makeShape(tool, sp, { team, equip, color: equip === 'colorpole' ? poleColor : undefined })
      if (!s) return
      setDraft([...base, s]); onSelect?.([s.id])
      dragRef.current = { kind: 'move', ids: [s.id], anchorId: s.id, start: p, before: [...base, s], created: true, moved: false }
      pointerId.current = e.pointerId; svgRef.current?.setPointerCapture(e.pointerId); return
    }

    // line tools, distance, field, pen
    const s = makeShape(tool, sp)
    if (!s) return
    if (s.type === 'line') { const t = lineEndTarget(base, p, bg); s.from = t.pos; s.to = t.pos; if (t.id) s.fromId = t.id }
    setDraft([...base, s]); onSelect?.([])
    dragRef.current = { kind: 'create', id: s.id, before: base, tool }
    pointerId.current = e.pointerId; svgRef.current?.setPointerCapture(e.pointerId)
  }

  function onPointerMove(e: React.PointerEvent<SVGSVGElement>) {
    const d = dragRef.current
    if (!d || e.pointerId !== pointerId.current) return
    const p = clampPt(toWorld(e), bg)
    const step = snap ? snapStep(bg) : undefined
    if (d.kind === 'marquee') {
      if (!d.moved && Math.hypot(p[0] - d.start[0], p[1] - d.start[1]) < 6 * kTouch) return
      d.moved = true
      const m = { a: d.start, b: p }
      marqueeRef.current = m; setMarquee(m)
      return
    }
    if (d.kind === 'move') {
      let dx = p[0] - d.start[0], dy = p[1] - d.start[1]
      if (!d.moved && Math.hypot(dx, dy) < 4 * kTouch && !d.created) return
      const s = d.before.find(x => x.id === d.anchorId)!
      if (step) {
        const anchor: Pt = 'pos' in s ? s.pos : 'a' in s ? s.a : 'from' in s ? s.from : s.points[0]
        const t = snapPt([anchor[0] + dx, anchor[1] + dy], step); dx = t[0] - anchor[0]; dy = t[1] - anchor[1]
      }
      d.moved = true
      setDraft(moveShapes(d.before, d.ids, dx, dy))
    } else if (d.kind === 'handle') {
      const s = d.before.find(x => x.id === d.id)!
      const patch = handlePatch(s, d.key, p, d.before, bg, step)
      if (patch) { d.moved = true; setDraft(updateShape(d.before, d.id, patch)) }
    } else {
      const cur = draft ?? d.before
      const s = cur.find(x => x.id === d.id)
      if (!s) return
      const sp = step ? snapPt(p, step) : p
      if (s.type === 'line') {
        const t = lineEndTarget(d.before, p, bg, s.fromId)
        setDraft(updateShape(cur, d.id, { to: t.pos, toId: t.id } as Partial<SceneShape>))
      } else if (s.type === 'dim') setDraft(updateShape(cur, d.id, { b: sp } as Partial<SceneShape>))
      else if (s.type === 'grid') setDraft(updateShape(cur, d.id, { b: sp } as Partial<SceneShape>))
      else if (s.type === 'pen') {
        const last = s.points[s.points.length - 1]
        if (dist(last, p) >= 2 * kTouch) setDraft(updateShape(cur, d.id, { points: [...s.points, p] } as Partial<SceneShape>))
      }
    }
  }

  function onPointerUp(e: React.PointerEvent<SVGSVGElement>) {
    if (e.pointerId !== pointerId.current) return
    const d = dragRef.current
    let cur = draft ?? undefined
    if (d && d.kind === 'create' && cur) {
      const s = cur.find(x => x.id === d.id)
      if (s && s.type === 'pen') cur = updateShape(cur, d.id, { points: decimate(s.points, 3 * kTouch) } as Partial<SceneShape>)
    }
    finish(true, cur)
  }

  // ── drawing ────────────────────────────────────────────────

  const sw = 2.6 * k
  const font = sketch ? HAND : SANS

  function renderShape(s: SceneShape) {
    const seed = hashStr(s.id)
    switch (s.type) {
      case 'player': {
        const r = playerRadius(u, s.scale)
        const fillC = s.color ?? TEAM_COLORS[s.team]
        return (
          <g key={s.id} data-shape-id={s.id} data-type="player">
            <path d={blobPath(s.pos, r, seed, sketch ? 0.11 : 0)} fill={fillC} stroke={INK} strokeWidth={sw} strokeLinejoin="round" />
            {s.num ? <text x={s.pos[0]} y={s.pos[1]} textAnchor="middle" dominantBaseline="central" fontSize={r * 1.15} fontWeight={700} fill={isLight(fillC) ? INK : '#fff'} style={{ fontFamily: font }}>{s.num}</text> : null}
          </g>
        )
      }
      case 'ball': {
        const r = ballRadius(u) * (s.scale ?? 1)
        return (
          <g key={s.id} data-shape-id={s.id} data-type="ball" transform={`translate(${s.pos[0]} ${s.pos[1]})`}>
            <circle r={r} fill="#fff" stroke={INK} strokeWidth={sw * 0.9} />
            <path d={`M0 ${-r * 0.5}L${r * 0.48} ${-r * 0.15}L${r * 0.3} ${r * 0.4}L${-r * 0.3} ${r * 0.4}L${-r * 0.48} ${-r * 0.15}Z`} fill={INK} />
          </g>
        )
      }
      case 'equip':
        return <g key={s.id} data-shape-id={s.id} data-type="equip" data-kind={s.kind} transform={`translate(${s.pos[0]} ${s.pos[1]}) rotate(${s.angle})`}><g transform={`scale(${s.scale ?? 1})`}>{equipGlyph(s.kind, u, k / (s.scale ?? 1), { color: s.color, num: s.num, look })}</g></g>
      case 'line': {
        let pts = curvePoints(s.from, s.to, s.bend, 28)
        const a = s.fromId && players.has(s.fromId) ? u * 1.25 * scaleOf(players.get(s.fromId)!) : 0
        const b = s.toId && players.has(s.toId) ? u * 1.25 * scaleOf(players.get(s.toId)!) + 4 * k : 4 * k
        pts = trimPolyline(pts, a, b)
        const head = 15 * k
        let body = pts
        if (s.kind === 'move') body = wavy(pts, 4.5 * k, 26 * k)
        else body = wobble(pts, seed, amp)
        const tip = body[body.length - 1]
        const from = body[Math.max(0, body.length - 1 - (s.kind === 'move' ? 6 : 3))]
        return (
          <g key={s.id} data-shape-id={s.id} data-type="line" data-kind={s.kind} fill="none" stroke={s.color ?? INK} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
            <path d={polyPath(body)} strokeDasharray={s.kind === 'return' ? `${11 * k} ${8 * k}` : undefined} />
            <path d={arrowHead(from, tip, head, seed, sketch ? 2 * k : 0)} />
          </g>
        )
      }
      case 'dim': {
        const L = dist(s.a, s.b)
        if (L < 1e-6) return null
        const ux = (s.b[0] - s.a[0]) / L, uy = (s.b[1] - s.a[1]) / L, nx = -uy, ny = ux
        const txt = dimText(s)
        // keep the label clear of its own line: push it out by half its width or height, whichever way the line runs
        const off = 8 * k + Math.abs(nx) * txt.length * 4.6 * k + Math.abs(ny) * 8 * k
        const t = 7 * k, mid: Pt = [(s.a[0] + s.b[0]) / 2 + nx * off, (s.a[1] + s.b[1]) / 2 + ny * off]
        return (
          <g key={s.id} data-shape-id={s.id} data-type="dim" stroke={s.color ?? DIM} strokeWidth={1.8 * k} fill="none" strokeLinecap="round">
            <path d={`M${s.a[0]} ${s.a[1]}L${s.b[0]} ${s.b[1]}`} />
            <path d={`M${s.a[0] + nx * t} ${s.a[1] + ny * t}L${s.a[0] - nx * t} ${s.a[1] - ny * t}M${s.b[0] + nx * t} ${s.b[1] + ny * t}L${s.b[0] - nx * t} ${s.b[1] - ny * t}`} />
            <path d={arrowHead(s.b, s.a, 9 * k) + arrowHead(s.a, s.b, 9 * k)} />
            <text x={mid[0]} y={mid[1]} textAnchor="middle" dominantBaseline="central" fontSize={16 * k} fontWeight={700} fill={s.color ?? DIM} stroke={PAPER} strokeWidth={4 * k} paintOrder="stroke" style={{ fontFamily: font }}>{txt}</text>
          </g>
        )
      }
      case 'grid': {
        const x0 = Math.min(s.a[0], s.b[0]), x1 = Math.max(s.a[0], s.b[0]), y1 = Math.max(s.a[1], s.b[1]), y0 = Math.min(s.a[1], s.b[1])
        return (
          <g key={s.id} data-shape-id={s.id} data-type="grid">
            <path d={boxPath(s.a, s.b, seed, amp)} fill="none" stroke={s.color ?? INK} strokeWidth={sw} strokeDasharray={`${16 * k} ${10 * k}`} strokeLinecap="round" />
            {s.label?.trim() ? <text x={(x0 + x1) / 2} y={y0 - 9 * k} textAnchor="middle" fontSize={20 * k} fontWeight={700} fill={s.color ?? INK} style={{ fontFamily: font }}>{s.label}</text> : null}
            {s.showSize ? <text x={(x0 + x1) / 2} y={y1 + 18 * k} textAnchor="middle" fontSize={15 * k} fontWeight={600} fill={s.color ?? DIM} style={{ fontFamily: font }}>{gridSizeText(s)}</text> : null}
          </g>
        )
      }
      case 'pen':
        return <path key={s.id} data-shape-id={s.id} data-type="pen" d={smoothPath(s.points)} fill="none" stroke={s.color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
      case 'text':
        return <text key={s.id} data-shape-id={s.id} data-type="text" x={s.pos[0]} y={s.pos[1]} textAnchor="middle" dominantBaseline="central" fontSize={textHeight(s.size, u) * (sketch ? 1.1 : 0.95)} fontWeight={700} fill={s.color ?? INK} stroke={PAPER} strokeWidth={3 * k} paintOrder="stroke" style={{ fontFamily: font }}>{s.text}</text>
    }
  }

  function halo(s: SceneShape) {
    const stroke = { fill: 'none', stroke: HALO, strokeOpacity: 0.45, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
    switch (s.type) {
      case 'player': return <circle cx={s.pos[0]} cy={s.pos[1]} r={playerRadius(u, s.scale) * 1.4} {...stroke} strokeWidth={4 * k} />
      case 'ball': return <circle cx={s.pos[0]} cy={s.pos[1]} r={ballRadius(u) * (s.scale ?? 1) + 5 * k} {...stroke} strokeWidth={4 * k} />
      case 'equip': { const z = equipDims(s, u), pad = 5 * k; return <rect x={-z.w / 2 - pad} y={-z.d / 2 - pad} width={z.w + 2 * pad} height={z.d + 2 * pad} rx={pad} transform={`translate(${s.pos[0]} ${s.pos[1]}) rotate(${s.angle})`} {...stroke} strokeWidth={3 * k} strokeDasharray={`${6 * k} ${5 * k}`} /> }
      case 'line': return <path d={polyPath(curvePoints(s.from, s.to, s.bend, 28))} {...stroke} strokeWidth={11 * k} />
      case 'dim': return <path d={`M${s.a[0]} ${s.a[1]}L${s.b[0]} ${s.b[1]}`} {...stroke} strokeWidth={11 * k} />
      case 'pen': return <path d={smoothPath(s.points)} {...stroke} strokeWidth={11 * k} />
      case 'grid': return <path d={`M${s.a[0]} ${s.a[1]}H${s.b[0]}V${s.b[1]}H${s.a[0]}Z`} {...stroke} strokeWidth={9 * k} />
      case 'text': { const h = textHeight(s.size, u), w = Math.max(1, s.text.length) * h * 0.55; return <rect x={s.pos[0] - w / 2 - 4 * k} y={s.pos[1] - h / 2 - 4 * k} width={w + 8 * k} height={h + 8 * k} rx={4 * k} {...stroke} strokeWidth={3 * k} strokeDasharray={`${6 * k} ${5 * k}`} /> }
    }
  }

  const selectedShapes = editable ? shapes.filter(s => selectedIds.includes(s.id)) : []
  const selected = selectedShapes.length === 1 ? selectedShapes[0] : undefined
  const groupBox = selectedShapes.length > 1 ? unionBox(selectedShapes.map(s => shapeBox(s, bg))) : null
  const ordered = drawOrder(shapes)
  const cursor = !editable ? 'default' : tool === 'select' ? 'default' : 'crosshair'

  return (
    <svg
      ref={svgRef}
      data-testid={testId ?? 'scene-svg'}
      viewBox={`${vb.x} ${vb.y} ${vb.w} ${vb.h}`}
      className={className}
      style={{ display: 'block', width: '100%', height: 'auto', touchAction: editable ? 'none' : 'auto', cursor, userSelect: 'none', WebkitUserSelect: 'none', ...style }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => finish(false)}
    >
      <Background bg={bg} k={k} look={look} plain={plain} />
      {selectedShapes.map(s => <g key={`h${s.id}`}>{halo(s)}</g>)}
      {groupBox && <rect data-testid="group-box" x={groupBox.x0 - 6 * k} y={groupBox.y0 - 6 * k} width={groupBox.x1 - groupBox.x0 + 12 * k} height={groupBox.y1 - groupBox.y0 + 12 * k} rx={6 * k} fill="none" stroke={HALO} strokeWidth={2 * k} strokeDasharray={`${8 * k} ${6 * k}`} />}
      {ordered.map(renderShape)}
      {selected && handlesOf(selected, bg, shapes).map(h => (
        <g key={h.key} data-handle={h.key}>
          <circle cx={h.pos[0]} cy={h.pos[1]} r={7 * k} fill="#fff" stroke={HALO} strokeWidth={2.2 * k} />
          {h.key === 'size' ? <path d={`M${h.pos[0] - 3 * k} ${h.pos[1] + 3 * k}L${h.pos[0] + 3 * k} ${h.pos[1] - 3 * k}M${h.pos[0] + 3 * k} ${h.pos[1] - 1 * k}V${h.pos[1] - 3 * k}H${h.pos[0] + 1 * k}M${h.pos[0] - 3 * k} ${h.pos[1] + 1 * k}V${h.pos[1] + 3 * k}H${h.pos[0] - 1 * k}`} fill="none" stroke={HALO} strokeWidth={1.6 * k} /> : null}
          {h.key === 'rot' ? <path d={`M${h.pos[0] - 3 * k} ${h.pos[1]}a${3 * k} ${3 * k} 0 1 1 ${3 * k} ${3 * k}`} fill="none" stroke={HALO} strokeWidth={1.6 * k} /> : null}
        </g>
      ))}
      {marquee && <rect data-testid="marquee" x={Math.min(marquee.a[0], marquee.b[0])} y={Math.min(marquee.a[1], marquee.b[1])} width={Math.abs(marquee.b[0] - marquee.a[0])} height={Math.abs(marquee.b[1] - marquee.a[1])} fill={HALO} fillOpacity={0.08} stroke={HALO} strokeWidth={1.6 * k} strokeDasharray={`${7 * k} ${5 * k}`} />}
    </svg>
  )
}

