'use client'

// components/AnnotationCanvas.tsx
// Hand-rolled Canvas 2D overlay (no drawing library) — normalized 0..1
// coords, resolution-independent between editor and playback.
//
// Shapes may be animated (see lib/annoAnim.ts). The canvas draws every shape as it looks at the drawing's clock τ,
// which the parent supplies as a getter (`time`) so the canvas can run its own animation loop without re-rendering the page.
// Without `time` a drawing is shown in its final state, exactly as drawings made before animation existed.
//
// Placing things takes as few gestures as possible:
//   highlight  one click (size comes from the coach's earlier corrections) — or press and drag to set the size right there
//   line, body shape, zone   press-drag-release, or the older click-click(-double-click)
// Resizing a highlight scales it proportionally (Shift = free), the mouse wheel resizes the highlight under the pointer,
// and every size the coach sets teaches the next markers.

import { useRef, useEffect, useCallback, useState, forwardRef, useImperativeHandle } from 'react'
import ShapeSettingsPanel from './ShapeSettingsPanel'
import { resolveAll, resolveShape, pinchPatch, type Animated, type Resolved } from '@/lib/annoAnim'
import { markerSize, type MarkerSample } from '@/lib/annoAnim'

export type NPoint = [number, number]
export type DashStyle = 'solid' | 'dashed' | 'dotted'
export type CurveStyle = 'pass' | 'dribble'
export type HighlightStyle = 'circle' | 'spotlight'

interface StyleProps {
  color: string
  opacity: number
  dash: DashStyle
}

export type Shape = (
  | ({ id: string; type: 'zone'; points: NPoint[] } & StyleProps) // freeform polygon, 3+ points
  | ({ id: string; type: 'curve'; from: NPoint; control: NPoint; to: NPoint; style: CurveStyle } & StyleProps) // bezier pass/dribble line
  | { id: string; type: 'label'; pos: NPoint; text: string; color: string }
  | ({ id: string; type: 'highlight'; pos: NPoint; radiusX: number; radiusY: number; style: HighlightStyle; playerName?: string } & StyleProps)
  | { id: string; type: 'number'; pos: NPoint; value: number; color: string }
  | ({ id: string; type: 'cone'; pos: NPoint; angle: number; length: number; width: number } & StyleProps) // body shape / vision / passing-lane wedge
  | ({ id: string; type: 'connector'; fromId: string; toId: string } & StyleProps) // line between two highlights, tracks their positions live
) & Animated

export type Tool = 'select' | 'zone' | 'curve' | 'label' | 'highlight' | 'number' | 'cone' | 'connector'

interface AnnotationCanvasProps {
  shapes: Shape[]
  editable: boolean
  tool: Tool
  onAddShape: (s: Shape) => void
  /** `tau` is the drawing clock at the moment of the edit (only given when the canvas has a clock), so the parent can turn a move into a keyframe. */
  onUpdateShape: (id: string, patch: Partial<Shape>, tau?: number) => void
  onRemoveShape: (id: string) => void
  onRequestLabelText?: (pos: NPoint) => void
  /** Called on click when not editable — lets a normal click on the video toggle play/pause via your own commands, instead of passing the click through to the iframe (which would re-trigger YouTube's own overlay). */
  onToggleVideo?: () => void
  /**
   * The four pitch calibration corners (normalized 0..1), if this game has been calibrated. Used as the depth scale for
   * marker sizes (bigger near the bottom of the calibrated area, smaller far away); omit it to use the picture's height.
   */
  pitchPolygon?: NPoint[] | null
  /** The drawing's clock in seconds. When given, the canvas animates by itself. */
  time?: () => number
  /** Selection can be owned by the parent (the timeline selects shapes too). */
  selectedId?: string | null
  onSelect?: (id: string | null) => void
  onDuplicateShape?: (id: string) => void
  /** Colour for new zones, lines, highlights and body shapes. */
  penColor?: string
  /** What the coach has taught us about marker size in this game, and where to report a new correction. */
  markerSamples?: MarkerSample[]
  onMarkerSample?: (s: MarkerSample) => void
  /** Show the style panel for the selected shape over the video (default). The phone HUD turns it off and has its own bar. */
  showSettings?: boolean
}

const DEFAULTS = {
  zone: { color: '#4ade80', opacity: 0.3, dash: 'solid' as DashStyle },
  curve: { color: '#ffffff', opacity: 1, dash: 'solid' as DashStyle, style: 'pass' as CurveStyle },
  highlight: { color: '#facc15', opacity: 0.85, dash: 'solid' as DashStyle, style: 'circle' as HighlightStyle },
  cone: { color: '#38bdf8', opacity: 0.35, dash: 'solid' as DashStyle },
  connector: { color: '#f472b6', opacity: 0.9, dash: 'dashed' as DashStyle },
}
const LABEL_COLOR = '#ffffff'
const HANDLE_RADIUS = 8
const MIN_RADIUS = 0.012
const DRAG_PX = 12            // a press that moves less than this is a click
const FINAL = 1e9             // τ of a drawing shown without a clock

function setDash(ctx: CanvasRenderingContext2D, dash: DashStyle) {
  if (dash === 'dashed') ctx.setLineDash([14, 10])
  else if (dash === 'dotted') ctx.setLineDash([3, 7])
  else ctx.setLineDash([])
}
/** The candidate with the smallest squared distance, if within `max`. */
function nearest<K extends string>(c: [K, number][], max: number): K | null {
  let best: K | null = null, bd = max
  for (const [k, d] of c) if (d <= bd) { best = k; bd = d }
  return best
}
function dist2(ax: number, ay: number, bx: number, by: number) { return (ax - bx) ** 2 + (ay - by) ** 2 }
function distToSegment2(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const l2 = dist2(ax, ay, bx, by)
  if (l2 === 0) return dist2(px, py, ax, ay)
  let t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2
  t = Math.max(0, Math.min(1, t))
  return dist2(px, py, ax + t * (bx - ax), ay + t * (by - ay))
}
function pointInPolygon(px: number, py: number, pts: [number, number][]) {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i]; const [xj, yj] = pts[j]
    const intersect = yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}
function bezierPoint(t: number, p0: [number, number], p1: [number, number], p2: [number, number]) {
  const x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
  const y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
  return [x, y] as [number, number]
}
const uid = () => Math.random().toString(36).slice(2, 10)

export interface AnnotationCanvasHandle {
  /** Commits any in-progress geometry (a zone that hasn't been double-clicked to close yet) as a real shape, or discards it if incomplete. Call right before saving. */
  finalizePending: () => void
}

type Drag =
  | { kind: 'handle'; shapeId: string; handle: string | number }
  | { kind: 'size'; shapeId: string; origin: [number, number]; ratio: number; moved: boolean }
  | { kind: 'gesture'; tool: 'zone' | 'curve' | 'cone'; origin: [number, number] }

const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, AnnotationCanvasProps>(function AnnotationCanvas(props, ref) {
  const {
    shapes, editable, tool,
    onAddShape, onUpdateShape, onRemoveShape, onRequestLabelText, onToggleVideo, pitchPolygon,
    time, onDuplicateShape, penColor, markerSamples, onMarkerSample, showSettings = true,
  } = props
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draftPoints = useRef<NPoint[]>([]) // in-progress zone/curve points
  const dragTarget = useRef<Drag | null>(null)
  const dragStartOffset = useRef<NPoint>([0, 0])
  const preview = useRef<{ from: NPoint; to: NPoint; tool: 'zone' | 'curve' | 'cone' } | null>(null) // rubber band while dragging out a line / zone / body shape
  const numberCounter = useRef(1)
  const [connectorFrom, setConnectorFrom] = useState<string | null>(null)
  // Touch: fingers are fat, so handles get a bigger target; two fingers pinch (and drag) the selected shape.
  const touchy = useRef(false)
  const pointers = useRef(new Map<number, [number, number]>())
  const pinch = useRef<{ id: string; base: Shape; d0: number; c0: [number, number] } | null>(null)

  const [selInternal, setSelInternal] = useState<string | null>(null)
  const selectedId = props.selectedId !== undefined ? props.selectedId : selInternal

  // Latest values for the animation loop and native listeners, which must not capture stale props.
  const live = useRef({ shapes, time, selectedId, editable, tool, markerSamples, onUpdateShape, onMarkerSample, pitchPolygon, onSelect: props.onSelect })
  live.current = { shapes, time, selectedId, editable, tool, markerSamples, onUpdateShape, onMarkerSample, pitchPolygon, onSelect: props.onSelect }
  const setSelectedId = useCallback((id: string | null) => { setSelInternal(id); live.current.onSelect?.(id) }, [])
  const penColorRef = useRef(penColor); penColorRef.current = penColor
  const connectorFromRef = useRef(connectorFrom); connectorFromRef.current = connectorFrom
  const getTau = useCallback(() => (live.current.time ? live.current.time() : FINAL), [])
  const stamped = !!time

  const toCanvas = useCallback((p: NPoint): [number, number] => {
    const c = canvasRef.current; if (!c) return [0, 0]
    return [p[0] * c.width, p[1] * c.height]
  }, [])
  const toNorm = useCallback((cx: number, cy: number): NPoint => {
    const c = canvasRef.current; if (!c) return [0, 0]
    return [cx / c.width, cy / c.height]
  }, [])
  const xy = (e: { clientX: number; clientY: number }): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect()
    const scale = canvasRef.current!.width / r.width
    return [(e.clientX - r.left) * scale, (e.clientY - r.top) * scale]
  }
  const pxScale = () => { const c = canvasRef.current!; return c.width / c.getBoundingClientRect().width }

  /** The shapes as they look at this moment — what the coach sees, so what they can grab. */
  const current = useCallback((): Shape[] => resolveAll(live.current.shapes, getTau()).map(r => r.s), [getTau])

  /** Every new shape is born at the playhead: drawn at 2 s, it appears at 2 s. */
  const add = useCallback((s: Shape) => {
    const tau = live.current.time ? live.current.time() : 0
    onAddShape(tau > 0.05 ? ({ ...s, t0: Math.round(tau * 100) / 100 } as Shape) : s)
  }, [onAddShape])
  const update = useCallback((id: string, patch: Partial<Shape>) => {
    live.current.onUpdateShape(id, patch, live.current.time ? live.current.time() : undefined)
  }, [])

  /** Depth of a point for marker sizing: 0 = far (top), 1 = near (bottom). */
  const depthOf = useCallback((p: NPoint): number => {
    const poly = live.current.pitchPolygon
    if (poly && poly.length >= 3) {
      const ys = poly.map(pt => pt[1]); const minY = Math.min(...ys), maxY = Math.max(...ys)
      return maxY > minY ? Math.max(0, Math.min(1, (p[1] - minY) / (maxY - minY))) : 0.5
    }
    return Math.max(0, Math.min(1, p[1]))
  }, [])
  const aspect = () => { const c = canvasRef.current; return c && c.height ? c.width / c.height : 16 / 9 }
  const reportSample = useCallback((s: Shape | undefined) => {
    if (!s || s.type !== 'highlight' || !live.current.onMarkerSample) return
    live.current.onMarkerSample({ y: depthOf(s.pos), rx: s.radiusX, ry: s.radiusY })
  }, [depthOf])

  function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
    ctx.beginPath()
    ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = '#fff'
    ctx.fill()
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.stroke()
  }

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const { editable, tool, selectedId } = live.current
    const tau = getTau()
    const resolved = resolveAll(live.current.shapes, tau)
    const byId = new Map<string, Resolved>(resolved.map(r => [r.s.id, r]))

    for (const { s, a, p: prog } of resolved) {
      const isSelected = editable && s.id === selectedId

      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (cpts.length < 2) continue
        ctx.save()
        ctx.beginPath()
        cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
        if (cpts.length >= 3) ctx.closePath()
        ctx.globalAlpha = s.opacity * a
        ctx.fillStyle = s.color
        if (cpts.length >= 3) ctx.fill()
        ctx.globalAlpha = a
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
        ctx.restore()
        if (isSelected) cpts.forEach(([x, y]) => drawHandle(ctx, x, y, s.color))
      }

      if (s.type === 'curve') {
        const p0 = toCanvas(s.from)
        let p1 = toCanvas(s.control), p2 = toCanvas(s.to)
        if (prog < 1) {
          // draw-on: the part of the curve from its start to `prog`
          const e = bezierPoint(prog, p0, p1, p2)
          p1 = [p0[0] + (p1[0] - p0[0]) * prog, p0[1] + (p1[1] - p0[1]) * prog]
          p2 = e
        }
        ctx.save()
        ctx.globalAlpha = s.opacity * a
        ctx.shadowColor = 'rgba(0,0,0,0.6)'
        ctx.shadowBlur = 4
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 6 : 4
        ctx.lineCap = 'round'
        setDash(ctx, s.dash)

        if (s.style === 'dribble') {
          // sample the bezier and draw a wavy line by offsetting perpendicular to the path
          const N = 40, amp = Math.max(4, canvas.width * 0.008)
          ctx.beginPath()
          for (let i = 0; i <= N; i++) {
            const t = i / N
            const [x, y] = bezierPoint(t, p0, p1, p2)
            const [xa, ya] = bezierPoint(Math.max(0, t - 0.01), p0, p1, p2)
            const [xb, yb] = bezierPoint(Math.min(1, t + 0.01), p0, p1, p2)
            const dx = xb - xa, dy = yb - ya
            const len = Math.hypot(dx, dy) || 1
            const nx = -dy / len, ny = dx / len
            const off = Math.sin(t * Math.PI * 8) * amp
            const px = x + nx * off, py = y + ny * off
            i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
          }
          ctx.stroke()
        } else {
          ctx.beginPath()
          ctx.moveTo(p0[0], p0[1])
          ctx.quadraticCurveTo(p1[0], p1[1], p2[0], p2[1])
          ctx.stroke()
        }
        ctx.setLineDash([])

        // arrowhead at the end, angled along the curve's final tangent
        if (prog > 0.03) {
          const [tx, ty] = p2
          const [nearEndX, nearEndY] = bezierPoint(0.92, p0, p1, p2)
          const angle = Math.atan2(ty - nearEndY, tx - nearEndX)
          const head = 14
          ctx.beginPath()
          ctx.moveTo(tx, ty)
          ctx.lineTo(tx - head * Math.cos(angle - Math.PI / 6), ty - head * Math.sin(angle - Math.PI / 6))
          ctx.lineTo(tx - head * Math.cos(angle + Math.PI / 6), ty - head * Math.sin(angle + Math.PI / 6))
          ctx.closePath()
          ctx.fillStyle = s.color
          ctx.fill()
        }
        ctx.restore()

        if (isSelected) {
          const q0 = toCanvas(s.from), q1 = toCanvas(s.control), q2 = toCanvas(s.to)
          drawHandle(ctx, q0[0], q0[1], s.color)
          drawHandle(ctx, q2[0], q2[1], s.color)
          drawHandle(ctx, q1[0], q1[1], '#a78bfa') // control point, distinct color
        }
      }

      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const rx = s.radiusX * canvas.width
        const ry = s.radiusY * canvas.height

        if (s.style === 'spotlight') {
          ctx.save()
          ctx.globalAlpha = a
          const grad = ctx.createLinearGradient(0, 0, 0, y)
          grad.addColorStop(0, `${s.color}00`)
          grad.addColorStop(1, `${s.color}55`)
          ctx.fillStyle = grad
          const topHalfWidth = rx * 0.35
          ctx.beginPath()
          ctx.moveTo(x - topHalfWidth, 0)
          ctx.lineTo(x + topHalfWidth, 0)
          ctx.lineTo(x + rx, y)
          ctx.lineTo(x - rx, y)
          ctx.closePath()
          ctx.fill()
          ctx.restore()
        }

        ctx.save()
        ctx.globalAlpha = s.opacity * a
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 4 : 3
        ctx.beginPath()
        ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2)
        ctx.stroke()
        ctx.restore()

        if (s.playerName) {
          ctx.save()
          ctx.globalAlpha = a
          const fontSize = Math.max(11, canvas.height * 0.022)
          ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
          ctx.fillStyle = 'rgba(0,0,0,0.6)'
          const w = ctx.measureText(s.playerName).width
          ctx.fillRect(x - w / 2 - 5, y + ry + 4, w + 10, fontSize + 6)
          ctx.fillStyle = '#fff'
          ctx.textAlign = 'center'; ctx.textBaseline = 'top'
          ctx.fillText(s.playerName, x, y + ry + 7)
          ctx.restore()
        }

        if (isSelected) {
          drawHandle(ctx, x, y, s.color) // move
          drawHandle(ctx, x + rx, y, s.color) // size
          drawHandle(ctx, x, y + ry, s.color) // size
        }
      }

      if (s.type === 'cone') {
        const [ox, oy] = toCanvas(s.pos)
        const r = s.length * canvas.width
        const a0 = s.angle - s.width / 2
        const a1 = s.angle + s.width / 2
        ctx.save()
        ctx.globalAlpha = s.opacity * a
        ctx.fillStyle = s.color
        ctx.beginPath()
        ctx.moveTo(ox, oy)
        ctx.arc(ox, oy, r, a0, a1)
        ctx.closePath()
        ctx.fill()
        ctx.globalAlpha = a
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
        ctx.restore()

        if (isSelected) {
          const tipX = ox + r * Math.cos(s.angle), tipY = oy + r * Math.sin(s.angle)
          const wX = ox + r * Math.cos(a1), wY = oy + r * Math.sin(a1)
          drawHandle(ctx, ox, oy, s.color) // move
          drawHandle(ctx, tipX, tipY, s.color) // direction + length
          drawHandle(ctx, wX, wY, '#a78bfa') // width, matches curve's control-point color
        }
      }

      if (s.type === 'connector') {
        const from = byId.get(s.fromId), to = byId.get(s.toId)
        if (from?.s.type === 'highlight' && to?.s.type === 'highlight') {
          const [fx, fy] = toCanvas(from.s.pos)
          const [tx, ty] = toCanvas(to.s.pos)
          ctx.save()
          ctx.globalAlpha = s.opacity * a * Math.min(from.a, to.a)
          setDash(ctx, s.dash)
          ctx.strokeStyle = s.color
          ctx.lineWidth = isSelected ? 4 : 2.5
          ctx.beginPath()
          ctx.moveTo(fx, fy)
          ctx.lineTo(tx, ty)
          ctx.stroke()
          ctx.restore()
          if (isSelected) drawHandle(ctx, (fx + tx) / 2, (fy + ty) / 2, s.color)
        }
      }

      if (s.type === 'label') {
        const [x, y] = toCanvas(s.pos)
        ctx.save()
        ctx.globalAlpha = a
        const fontSize = Math.max(14, canvas.height * 0.035)
        ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
        ctx.fillStyle = 'rgba(0,0,0,0.55)'
        const w = ctx.measureText(s.text).width
        ctx.fillRect(x - 6, y - fontSize, w + 12, fontSize + 10)
        ctx.fillStyle = s.color ?? LABEL_COLOR
        ctx.textBaseline = 'top'
        ctx.fillText(s.text, x, y - fontSize + 5)
        ctx.restore()
        if (isSelected) drawHandle(ctx, x, y, s.color)
      }

      if (s.type === 'number') {
        const [x, y] = toCanvas(s.pos)
        const r = Math.max(12, canvas.height * 0.022)
        ctx.save()
        ctx.globalAlpha = a
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = s.color
        ctx.fill()
        ctx.strokeStyle = isSelected ? '#facc15' : '#fff'
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
        ctx.fillStyle = '#fff'
        ctx.font = `700 ${r}px 'DM Sans', sans-serif`
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText(String(s.value), x, y + 1)
        ctx.restore()
      }

      // The path a moving shape takes, with a diamond at each keyframe, so the coach sees what they animated.
      if (isSelected && 'pos' in s && s.kf?.some(k => 'pos' in k.v)) {
        const src = live.current.shapes.find(x => x.id === s.id)
        if (src) {
          const kfs = (src.kf ?? []).filter(k => 'pos' in k.v)
          const tEnd = kfs[kfs.length - 1].t, tStart = src.t0 ?? 0
          ctx.save()
          ctx.strokeStyle = 'rgba(255,255,255,0.75)'
          ctx.lineWidth = 2
          ctx.setLineDash([2, 6])
          ctx.lineCap = 'round'
          ctx.beginPath()
          const N = 48
          for (let i = 0; i <= N; i++) {
            const r = resolveShape(src, tStart + ((tEnd - tStart) * i) / N)
            if (!r || !('pos' in r.s)) continue
            const [px, py] = toCanvas(r.s.pos)
            if (i === 0) ctx.moveTo(px, py)
            else ctx.lineTo(px, py)
          }
          ctx.stroke()
          ctx.setLineDash([])
          for (const k of kfs) {
            const [px, py] = toCanvas(k.v.pos as NPoint)
            const hit = Math.abs(k.t - tau) < 0.06
            ctx.beginPath()
            ctx.moveTo(px, py - 7); ctx.lineTo(px + 7, py); ctx.lineTo(px, py + 7); ctx.lineTo(px - 7, py); ctx.closePath()
            ctx.fillStyle = hit ? '#facc15' : '#ffffff'
            ctx.fill()
            ctx.strokeStyle = '#0f2972'; ctx.lineWidth = 1.5; ctx.stroke()
          }
          ctx.restore()
        }
      }
    }

    // in-progress previews
    const pv = preview.current
    if (editable && pv) {
      const [fx, fy] = toCanvas(pv.from), [tx, ty] = toCanvas(pv.to)
      ctx.save()
      ctx.setLineDash([6, 6]); ctx.lineWidth = 3
      const col = penColorRef.current ?? DEFAULTS[pv.tool === 'zone' ? 'zone' : pv.tool === 'curve' ? 'curve' : 'cone'].color
      ctx.strokeStyle = col; ctx.fillStyle = col
      if (pv.tool === 'zone') {
        ctx.globalAlpha = 0.25; ctx.fillRect(Math.min(fx, tx), Math.min(fy, ty), Math.abs(tx - fx), Math.abs(ty - fy))
        ctx.globalAlpha = 1; ctx.strokeRect(Math.min(fx, tx), Math.min(fy, ty), Math.abs(tx - fx), Math.abs(ty - fy))
      } else if (pv.tool === 'curve') {
        ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke()
      } else {
        const ang = Math.atan2(ty - fy, tx - fx), r = Math.hypot(tx - fx, ty - fy)
        ctx.globalAlpha = 0.3; ctx.beginPath(); ctx.moveTo(fx, fy); ctx.arc(fx, fy, r, ang - Math.PI / 6, ang + Math.PI / 6); ctx.closePath(); ctx.fill()
        ctx.globalAlpha = 1; ctx.stroke()
      }
      ctx.restore()
    }
    if (editable && tool === 'zone' && draftPoints.current.length > 0) {
      const cpts = draftPoints.current.map(toCanvas)
      ctx.beginPath()
      cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
      ctx.strokeStyle = penColorRef.current ?? DEFAULTS.zone.color
      ctx.lineWidth = 2
      ctx.setLineDash([6, 6])
      ctx.stroke()
      ctx.setLineDash([])
      cpts.forEach(([x, y]) => drawHandle(ctx, x, y, DEFAULTS.zone.color))
    }
    if (editable && (tool === 'curve' || tool === 'cone') && draftPoints.current.length === 1) {
      const [x, y] = toCanvas(draftPoints.current[0])
      drawHandle(ctx, x, y, DEFAULTS.curve.color)
    }
    if (editable && tool === 'connector' && connectorFromRef.current) {
      const fromShape = byId.get(connectorFromRef.current)?.s
      if (fromShape?.type === 'highlight') {
        const [x, y] = toCanvas(fromShape.pos)
        const r = Math.max(fromShape.radiusX * canvas.width, fromShape.radiusY * canvas.height) + 8
        ctx.save()
        ctx.strokeStyle = DEFAULTS.connector.color
        ctx.lineWidth = 3
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.stroke()
        ctx.restore()
      }
    }
  }, [getTau, toCanvas])

  const drawRef = useRef(draw); drawRef.current = draw

  // Resize with the container; animate when there is a clock; otherwise redraw whenever something changes.
  useEffect(() => {
    const container = containerRef.current, canvas = canvasRef.current
    if (!container || !canvas) return
    const resize = () => { canvas.width = container.clientWidth; canvas.height = container.clientHeight; drawRef.current() }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [])
  useEffect(() => {
    if (!stamped) return
    let raf = 0
    const loop = () => { drawRef.current(); raf = requestAnimationFrame(loop) }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [stamped])
  useEffect(() => { if (!stamped) draw() }, [draw, stamped, shapes, editable, tool, selectedId, connectorFrom, penColor])

  useEffect(() => {
    if (draftPoints.current.length >= 3) {
      // A zone was mid-draw (never double-clicked to close yet) and the tool
      // just changed out from under it — commit it rather than losing it.
      add({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone, color: penColorRef.current ?? DEFAULTS.zone.color })
    }
    draftPoints.current = []
    preview.current = null
    setConnectorFrom(null)
    if (tool !== 'select') setSelectedId(null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool])
  useEffect(() => { if (selectedId && !shapes.find(s => s.id === selectedId)) setSelectedId(null) }, [shapes, selectedId, setSelectedId])
  useEffect(() => {
    const maxNum = shapes.filter((s): s is Extract<Shape, { type: 'number' }> => s.type === 'number')
      .reduce((m, s) => Math.max(m, s.value), 0)
    numberCounter.current = maxNum + 1
  }, [shapes])

  useImperativeHandle(ref, () => ({
    finalizePending: () => {
      if (tool === 'zone' && draftPoints.current.length >= 3) {
        add({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone, color: penColorRef.current ?? DEFAULTS.zone.color })
      }
      // A single placed curve-start point with no endpoint isn't a valid
      // shape (needs 2 clicks minimum) — nothing meaningful to commit there,
      // just discard it below.
      draftPoints.current = []
      preview.current = null
    },
  }), [tool, add])

  // The wheel resizes the highlight under the pointer (or the selected one). Native listener: React's onWheel cannot stop the page scrolling.
  useEffect(() => {
    const c = canvasRef.current
    if (!c || !editable) return
    let timer: ReturnType<typeof setTimeout> | null = null
    let sizedId: string | null = null
    const onWheel = (e: WheelEvent) => {
      const cur = resolveAll(live.current.shapes, getTau()).map(r => r.s)
      const [cx, cy] = xy(e)
      let target: Extract<Shape, { type: 'highlight' }> | undefined
      const sel = cur.find(s => s.id === live.current.selectedId)
      if (sel?.type === 'highlight') target = sel
      else {
        for (let i = cur.length - 1; i >= 0 && !target; i--) {
          const s = cur[i]
          if (s.type !== 'highlight') continue
          const [x, y] = toCanvas(s.pos)
          const nx = (cx - x) / (s.radiusX * c.width * 1.3), ny = (cy - y) / (s.radiusY * c.height * 1.3)
          if (nx * nx + ny * ny <= 1) target = s
        }
      }
      if (!target) return
      e.preventDefault()
      const f = e.deltaY < 0 ? 1.07 : 1 / 1.07
      live.current.onUpdateShape(target.id, {
        radiusX: Math.max(MIN_RADIUS, Math.min(0.12, target.radiusX * f)),
        radiusY: Math.max(MIN_RADIUS, Math.min(0.2, target.radiusY * f)),
      } as Partial<Shape>, live.current.time ? live.current.time() : undefined)
      sizedId = target.id
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {   // the wheel has settled: that size is a correction worth learning from
        const s = live.current.shapes.find(x => x.id === sizedId)
        const r = s ? resolveShape(s, getTau())?.s : undefined
        reportSample(r)
      }, 450)
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => { c.removeEventListener('wheel', onWheel); if (timer) clearTimeout(timer) }
  }, [editable, getTau, toCanvas, reportSample])

  const hitHandle = useCallback((cx: number, cy: number, only: string | null) => {
    const c = canvasRef.current!
    const t2 = ((HANDLE_RADIUS + 5) * (touchy.current ? 2 : 1) * pxScale()) ** 2
    const byId = new Map(current().map(s => [s.id, s]))
    for (const s of current()) {
      if (only !== null && s.id !== only) continue
      if (s.type === 'zone') {
        for (let i = 0; i < s.points.length; i++) {
          const [x, y] = toCanvas(s.points[i])
          if (dist2(x, y, cx, cy) <= t2) return { shapeId: s.id, handle: i }
        }
      }
      if (s.type === 'curve') {
        const [fx, fy] = toCanvas(s.from), [tx, ty] = toCanvas(s.to), [cx2, cy2] = toCanvas(s.control)
        if (dist2(cx2, cy2, cx, cy) <= t2) return { shapeId: s.id, handle: 'control' }
        if (dist2(fx, fy, cx, cy) <= t2) return { shapeId: s.id, handle: 'from' }
        if (dist2(tx, ty, cx, cy) <= t2) return { shapeId: s.id, handle: 'to' }
      }
      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const rx = s.radiusX * c.width, ry = s.radiusY * c.height
        // The nearest handle wins: on a small marker a finger's reach covers all three, and the centre must still drag it.
        const best = nearest<'radiusX' | 'radiusY' | 'move'>([
          ['radiusX', dist2(x + rx, y, cx, cy)], ['radiusY', dist2(x, y + ry, cx, cy)], ['move', dist2(x, y, cx, cy) * 0.8],
        ], t2)
        if (best) return { shapeId: s.id, handle: best }
      }
      if (s.type === 'cone') {
        const [ox, oy] = toCanvas(s.pos)
        const r = s.length * c.width
        const tipX = ox + r * Math.cos(s.angle), tipY = oy + r * Math.sin(s.angle)
        const a1 = s.angle + s.width / 2
        const wX = ox + r * Math.cos(a1), wY = oy + r * Math.sin(a1)
        const best = nearest<'width' | 'direction' | 'move'>([
          ['width', dist2(wX, wY, cx, cy)], ['direction', dist2(tipX, tipY, cx, cy)], ['move', dist2(ox, oy, cx, cy) * 0.8],
        ], t2)
        if (best) return { shapeId: s.id, handle: best }
      }
      if (s.type === 'connector') {
        const from = byId.get(s.fromId), to = byId.get(s.toId)
        if (from?.type === 'highlight' && to?.type === 'highlight') {
          const [fx, fy] = toCanvas(from.pos), [tx, ty] = toCanvas(to.pos)
          if (dist2((fx + tx) / 2, (fy + ty) / 2, cx, cy) <= t2) return { shapeId: s.id, handle: 'select' }
        }
      }
    }
    return null
  }, [current, toCanvas])

  const hitBody = useCallback((cx: number, cy: number): string | null => {
    const c = canvasRef.current!
    const cur = current()
    const byId = new Map(cur.map(s => [s.id, s]))
    const scale = pxScale()
    for (let i = cur.length - 1; i >= 0; i--) {
      const s = cur[i]
      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (pointInPolygon(cx, cy, cpts)) return s.id
      }
      if (s.type === 'curve') {
        const p0 = toCanvas(s.from), p1 = toCanvas(s.control), p2 = toCanvas(s.to)
        for (let t = 0; t <= 1; t += 0.05) {
          const [x, y] = bezierPoint(t, p0, p1, p2)
          if (dist2(x, y, cx, cy) <= ((touchy.current ? 20 : 12) * scale) ** 2) return s.id
        }
      }
      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const rx = s.radiusX * c.width, ry = s.radiusY * c.height
        const nx = (cx - x) / rx, ny = (cy - y) / ry
        if (nx * nx + ny * ny <= 1) return s.id
      }
      if (s.type === 'cone') {
        const [ox, oy] = toCanvas(s.pos)
        const r = s.length * c.width
        const dx = cx - ox, dy = cy - oy
        const dist = Math.hypot(dx, dy)
        if (dist <= r) {
          let a = Math.atan2(dy, dx) - s.angle
          while (a > Math.PI) a -= 2 * Math.PI
          while (a < -Math.PI) a += 2 * Math.PI
          if (Math.abs(a) <= s.width / 2) return s.id
        }
      }
      if (s.type === 'label' || s.type === 'number') {
        const [x, y] = toCanvas(s.pos)
        if (dist2(x, y, cx, cy) <= ((touchy.current ? 34 : 25) * scale) ** 2) return s.id
      }
      if (s.type === 'connector') {
        const from = byId.get(s.fromId), to = byId.get(s.toId)
        if (from?.type === 'highlight' && to?.type === 'highlight') {
          const [fx, fy] = toCanvas(from.pos), [tx, ty] = toCanvas(to.pos)
          if (distToSegment2(cx, cy, fx, fy, tx, ty) <= (10 * scale) ** 2) return s.id
        }
      }
    }
    return null
  }, [current, toCanvas])

  const newCurve = (from: NPoint, to: NPoint): Shape => ({
    id: uid(), type: 'curve', from, control: [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2], to,
    ...DEFAULTS.curve, color: penColor ?? DEFAULTS.curve.color,
  })
  const newCone = (origin: NPoint, cx: number, cy: number): Shape => {
    const [ox, oy] = toCanvas(origin)
    return {
      id: uid(), type: 'cone', pos: origin, angle: Math.atan2(cy - oy, cx - ox),
      length: Math.max(0.03, Math.hypot(cx - ox, cy - oy) / canvasRef.current!.width), width: Math.PI / 3,
      ...DEFAULTS.cone, color: penColor ?? DEFAULTS.cone.color,
    }
  }
  const newRect = (a: NPoint, b: NPoint): Shape => ({
    id: uid(), type: 'zone', points: [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]],
    ...DEFAULTS.zone, color: penColor ?? DEFAULTS.zone.color,
  })

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!editable) { onToggleVideo?.(); return }
    const [cx, cy] = xy(e)
    touchy.current = e.pointerType === 'touch' || e.pointerType === 'pen'
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* not every browser/pointer supports capture */ }
    pointers.current.set(e.pointerId, [cx, cy])
    if (pointers.current.size >= 2) { startPinch(); return }

    if (tool === 'select') {
      const handleHit = hitHandle(cx, cy, selectedId)
      if (handleHit) {
        setSelectedId(handleHit.shapeId)
        if (handleHit.handle === 'select') return
        if (handleHit.handle === 'move') {
          const s = current().find(x => x.id === handleHit.shapeId)
          if (s && 'pos' in s) { const p = toNorm(cx, cy); dragStartOffset.current = [p[0] - s.pos[0], p[1] - s.pos[1]] }
        }
        dragTarget.current = { kind: 'handle', ...handleHit }
        return
      }
      const bodyHit = hitBody(cx, cy)
      setSelectedId(bodyHit)
      if (bodyHit) {
        const shape = current().find(s => s.id === bodyHit)!
        if (shape.type === 'connector') {
          // Selectable, but not draggable as a whole — its position
          // comes from the two highlights it links, not its own coords.
          return
        }
        const p = toNorm(cx, cy)
        const anchor = shape.type === 'highlight' || shape.type === 'label' || shape.type === 'number' || shape.type === 'cone' ? shape.pos
          : shape.type === 'curve' ? shape.from : shape.points[0]
        dragStartOffset.current = [p[0] - anchor[0], p[1] - anchor[1]]
        dragTarget.current = { kind: 'handle', shapeId: bodyHit, handle: 'move' }
      }
      return
    }

    if (tool === 'zone') {
      if (draftPoints.current.length === 0) {
        dragTarget.current = { kind: 'gesture', tool: 'zone', origin: [cx, cy] }   // press-drag = rectangle; a plain click starts a polygon
        return
      }
      draftPoints.current = [...draftPoints.current, toNorm(cx, cy)]
      drawRef.current()
      return
    }

    if (tool === 'curve') {
      if (draftPoints.current.length === 1) {
        add(newCurve(draftPoints.current[0], toNorm(cx, cy)))
        draftPoints.current = []; preview.current = null
        return
      }
      dragTarget.current = { kind: 'gesture', tool: 'curve', origin: [cx, cy] }
      return
    }

    if (tool === 'cone') {
      if (draftPoints.current.length === 1) {
        add(newCone(draftPoints.current[0], cx, cy))
        draftPoints.current = []; preview.current = null
        return
      }
      dragTarget.current = { kind: 'gesture', tool: 'cone', origin: [cx, cy] }
      return
    }

    if (tool === 'highlight') {
      const p = toNorm(cx, cy)
      const { radiusX, radiusY } = markerSize(markerSamples ?? [], depthOf(p), aspect())
      const id = uid()
      add({ id, type: 'highlight', pos: p, radiusX, radiusY, ...DEFAULTS.highlight, color: penColor ?? DEFAULTS.highlight.color })
      // Still holding the button? Dragging outward sets the size right now.
      dragTarget.current = { kind: 'size', shapeId: id, origin: [cx, cy], ratio: radiusY / radiusX, moved: false }
      return
    }

    if (tool === 'number') {
      const p = toNorm(cx, cy)
      add({ id: uid(), type: 'number', pos: p, value: numberCounter.current, color: '#0f2972' })
      return
    }

    if (tool === 'connector') {
      // Generous "nearest highlight" search rather than requiring an
      // exact hit inside the drawn ellipse — much easier to hit
      // reliably when several highlights are close together.
      const c = canvasRef.current!
      let nearest: string | null = null
      let nearestDist = Infinity
      for (const s of current()) {
        if (s.type !== 'highlight') continue
        const [hx, hy] = toCanvas(s.pos)
        const d = Math.hypot(hx - cx, hy - cy)
        const reach = Math.max(s.radiusX * c.width, s.radiusY * c.height) * 1.8 + 20
        if (d <= reach && d < nearestDist) { nearest = s.id; nearestDist = d }
      }

      if (!nearest) {
        setConnectorFrom(null) // clicked empty space — cancel rather than get stuck
        return
      }
      if (!connectorFrom) {
        setConnectorFrom(nearest)
      } else if (nearest !== connectorFrom) {
        const fromShape = current().find(s => s.id === connectorFrom)
        const color = fromShape?.type === 'highlight' ? fromShape.color : DEFAULTS.connector.color
        add({ id: uid(), type: 'connector', fromId: connectorFrom, toId: nearest, ...DEFAULTS.connector, color })
        setConnectorFrom(null)
      }
      return
    }

    if (tool === 'label') {
      const p = toNorm(cx, cy)
      if (onRequestLabelText) onRequestLabelText(p)
      else {
        const text = window.prompt('Label text')
        if (text) add({ id: uid(), type: 'label', pos: p, text, color: LABEL_COLOR })
      }
    }
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!editable) return
    const [cx, cy] = xy(e)
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, [cx, cy])
    if (pointers.current.size >= 2) { movePinch(); return }
    if (pinch.current) return   // one finger is still down after a pinch: ignore it until it lifts
    const p = toNorm(cx, cy)
    const canvas = canvasRef.current!
    const drag = dragTarget.current

    if (!drag) {
      // rubber band for the second click of a click-click line / body shape
      if ((tool === 'curve' || tool === 'cone') && draftPoints.current.length === 1) {
        preview.current = { from: draftPoints.current[0], to: p, tool }
        if (!stamped) drawRef.current()
      }
      // cursor feedback: what would a press do here?
      let cursor = tool !== 'select' ? 'crosshair' : 'default'
      if (tool === 'select') {
        const h = hitHandle(cx, cy, selectedId)
        cursor = h ? (h.handle === 'move' || h.handle === 'select' ? 'move' : 'nwse-resize') : hitBody(cx, cy) ? 'move' : 'default'
      }
      canvas.style.cursor = cursor
      return
    }

    if (drag.kind === 'gesture') {
      const moved = Math.hypot(cx - drag.origin[0], cy - drag.origin[1]) / pxScale()
      if (moved >= DRAG_PX) {
        preview.current = { from: toNorm(drag.origin[0], drag.origin[1]), to: p, tool: drag.tool }
        if (!stamped) drawRef.current()
      }
      return
    }

    const shape = current().find(s => s.id === drag.shapeId)
    if (!shape) return

    if (drag.kind === 'size' && shape.type === 'highlight') {
      const d = Math.hypot(cx - drag.origin[0], cy - drag.origin[1])
      if (!drag.moved && d / pxScale() < DRAG_PX) return
      drag.moved = true
      const radiusX = Math.max(MIN_RADIUS, Math.min(0.12, d / canvas.width))
      update(shape.id, { radiusX, radiusY: Math.max(MIN_RADIUS, radiusX * drag.ratio) } as Partial<Shape>)
      return
    }
    if (drag.kind !== 'handle') return
    const { shapeId, handle } = drag

    if (shape.type === 'zone' && typeof handle === 'number') {
      const points = [...shape.points]; points[handle] = p
      update(shapeId, { points } as Partial<Shape>)
    }
    if (shape.type === 'zone' && handle === 'move') {
      const dx = p[0] - dragStartOffset.current[0] - shape.points[0][0]
      const dy = p[1] - dragStartOffset.current[1] - shape.points[0][1]
      update(shapeId, { points: shape.points.map(([x, y]) => [x + dx, y + dy]) } as Partial<Shape>)
    }
    if (shape.type === 'curve') {
      if (handle === 'from') update(shapeId, { from: p } as Partial<Shape>)
      if (handle === 'to') update(shapeId, { to: p } as Partial<Shape>)
      if (handle === 'control') update(shapeId, { control: p } as Partial<Shape>)
      if (handle === 'move') {
        const dx = p[0] - dragStartOffset.current[0] - shape.from[0]
        const dy = p[1] - dragStartOffset.current[1] - shape.from[1]
        update(shapeId, {
          from: [shape.from[0] + dx, shape.from[1] + dy],
          to: [shape.to[0] + dx, shape.to[1] + dy],
          control: [shape.control[0] + dx, shape.control[1] + dy],
        } as Partial<Shape>)
      }
    }
    if (shape.type === 'highlight') {
      if (handle === 'move') update(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
      // Either size handle scales the whole marker (the shape stays a ground ellipse); Shift stretches one axis on its own.
      if (handle === 'radiusX') {
        const rx = Math.max(MIN_RADIUS, p[0] - shape.pos[0])
        update(shapeId, { radiusX: rx, radiusY: e.shiftKey ? shape.radiusY : Math.max(MIN_RADIUS, shape.radiusY * (rx / shape.radiusX)) } as Partial<Shape>)
      }
      if (handle === 'radiusY') {
        const ry = Math.max(MIN_RADIUS, p[1] - shape.pos[1])
        update(shapeId, { radiusY: ry, radiusX: e.shiftKey ? shape.radiusX : Math.max(MIN_RADIUS, shape.radiusX * (ry / shape.radiusY)) } as Partial<Shape>)
      }
    }
    if (shape.type === 'cone') {
      if (handle === 'move') {
        update(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
      }
      if (handle === 'direction') {
        const [ox, oy] = toCanvas(shape.pos)
        update(shapeId, { angle: Math.atan2(cy - oy, cx - ox), length: Math.max(0.03, Math.hypot(cx - ox, cy - oy) / canvas.width) } as Partial<Shape>)
      }
      if (handle === 'width') {
        const [ox, oy] = toCanvas(shape.pos)
        let a = Math.atan2(cy - oy, cx - ox) - shape.angle
        while (a > Math.PI) a -= 2 * Math.PI
        while (a < -Math.PI) a += 2 * Math.PI
        update(shapeId, { width: Math.max(0.15, Math.min(Math.PI * 0.9, Math.abs(a) * 2)) } as Partial<Shape>)
      }
    }
    if ((shape.type === 'label' || shape.type === 'number') && handle === 'move') {
      update(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
    }
  }

  /** Two fingers down: whatever one finger had started is dropped, and the selected (else the touched) shape is pinched. */
  function startPinch() {
    dragTarget.current = null; preview.current = null; draftPoints.current = draftPoints.current.length === 1 ? [] : draftPoints.current
    const pts = [...pointers.current.values()]
    const [a, b] = pts
    const c0: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
    const cur = current()
    const id = (selectedId && cur.find(s => s.id === selectedId)?.id) || hitBody(c0[0], c0[1])
    const base = id ? cur.find(s => s.id === id) : undefined
    if (!id || !base) { pinch.current = null; return }
    setSelectedId(id)
    pinch.current = { id, base, d0: Math.max(1, Math.hypot(a[0] - b[0], a[1] - b[1])), c0 }
  }
  function movePinch() {
    const pz = pinch.current; if (!pz) return
    const [a, b] = [...pointers.current.values()]
    const c = canvasRef.current!
    const f = Math.hypot(a[0] - b[0], a[1] - b[1]) / pz.d0
    const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
    const patch = pinchPatch(pz.base, f, (mid[0] - pz.c0[0]) / c.width, (mid[1] - pz.c0[1]) / c.height)
    if (patch) update(pz.id, patch)
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const wasPinching = pointers.current.size >= 2
    pointers.current.delete(e.pointerId)
    try { e.currentTarget.releasePointerCapture(e.pointerId) } catch { /* ignore */ }
    if (wasPinching || pinch.current) {
      const pz = pinch.current
      if (pointers.current.size < 2 && pz) {
        const s = live.current.shapes.find(x => x.id === pz.id)
        reportSample(s ? resolveShape(s, getTau())?.s : undefined)   // a pinched marker is a size correction worth learning from
      }
      if (pointers.current.size === 0) pinch.current = null
      dragTarget.current = null
      return
    }
    const drag = dragTarget.current
    dragTarget.current = null
    if (!editable || !drag) return
    const [cx, cy] = xy(e)

    if (drag.kind === 'gesture') {
      const moved = Math.hypot(cx - drag.origin[0], cy - drag.origin[1]) / pxScale()
      const from = toNorm(drag.origin[0], drag.origin[1]), to = toNorm(cx, cy)
      preview.current = null
      if (moved >= DRAG_PX) {
        if (drag.tool === 'curve') add(newCurve(from, to))
        else if (drag.tool === 'cone') add(newCone(from, cx, cy))
        else add(newRect(from, to))
      } else if (drag.tool === 'zone') {
        draftPoints.current = [from]       // a click: polygon mode, keep clicking corners, double-click to close
      } else {
        draftPoints.current = [from]       // a click: first end of a click-click line / body shape
      }
      drawRef.current()
      return
    }
    if (drag.kind === 'size') {
      if (drag.moved) reportSample(live.current.shapes.find(s => s.id === drag.shapeId) && resolveShape(live.current.shapes.find(s => s.id === drag.shapeId)!, getTau())?.s)
      return
    }
    if (drag.kind === 'handle' && (drag.handle === 'radiusX' || drag.handle === 'radiusY')) {
      const s = live.current.shapes.find(x => x.id === drag.shapeId)
      reportSample(s ? resolveShape(s, getTau())?.s : undefined)
    }
  }

  const selectedShape = (() => {
    const s = shapes.find(x => x.id === selectedId)
    return s ? (resolveShape(s, getTau())?.s ?? s) : null
  })()

  return (
    <div ref={containerRef} style={{ position: 'absolute', inset: 0, pointerEvents: editable ? 'auto' : 'none' }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: editable && tool !== 'select' ? 'crosshair' : 'default', touchAction: 'none' }}
        onDoubleClick={() => {
          if (!editable || tool !== 'zone' || draftPoints.current.length < 3) return
          add({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone, color: penColor ?? DEFAULTS.zone.color })
          draftPoints.current = []
          drawRef.current()
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={e => { pointers.current.delete(e.pointerId); if (pointers.current.size === 0) pinch.current = null; dragTarget.current = null; preview.current = null }}
      />

      {editable && showSettings && selectedShape && (
        <ShapeSettingsPanel
          shape={selectedShape}
          onChange={patch => update(selectedShape.id, patch)}
          onRemove={() => { onRemoveShape(selectedShape.id); setSelectedId(null) }}
          onDuplicate={onDuplicateShape ? () => onDuplicateShape(selectedShape.id) : undefined}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  )
})

export default AnnotationCanvas
