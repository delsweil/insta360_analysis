'use client'

// components/AnnotationCanvas.tsx
// Hand-rolled Canvas 2D overlay (no drawing library) — normalized 0..1
// coords, resolution-independent between editor and playback.

import { useRef, useEffect, useCallback, useState, forwardRef, useImperativeHandle } from 'react'
import ShapeSettingsPanel from './ShapeSettingsPanel'

export type NPoint = [number, number]
export type DashStyle = 'solid' | 'dashed' | 'dotted'
export type CurveStyle = 'pass' | 'dribble'
export type HighlightStyle = 'circle' | 'spotlight'

interface StyleProps {
  color: string
  opacity: number
  dash: DashStyle
}

export type Shape =
  | ({ id: string; type: 'zone'; points: NPoint[] } & StyleProps) // freeform polygon, 3+ points
  | ({ id: string; type: 'curve'; from: NPoint; control: NPoint; to: NPoint; style: CurveStyle } & StyleProps) // bezier pass/dribble line
  | { id: string; type: 'label'; pos: NPoint; text: string; color: string }
  | ({ id: string; type: 'highlight'; pos: NPoint; radiusX: number; radiusY: number; style: HighlightStyle; playerName?: string } & StyleProps)
  | { id: string; type: 'number'; pos: NPoint; value: number; color: string }
  | ({ id: string; type: 'cone'; pos: NPoint; angle: number; length: number; width: number } & StyleProps) // body shape / vision / passing-lane wedge

export type Tool = 'select' | 'zone' | 'curve' | 'label' | 'highlight' | 'number' | 'cone'

interface AnnotationCanvasProps {
  shapes: Shape[]
  editable: boolean
  tool: Tool
  onAddShape: (s: Shape) => void
  onUpdateShape: (id: string, patch: Partial<Shape>) => void
  onRemoveShape: (id: string) => void
  onRequestLabelText?: (pos: NPoint) => void
  /** Called on click when not editable — lets a normal click on the video toggle play/pause via your own commands, instead of passing the click through to the iframe (which would re-trigger YouTube's own overlay). */
  onToggleVideo?: () => void
}

const DEFAULTS = {
  zone: { color: '#4ade80', opacity: 0.3, dash: 'solid' as DashStyle },
  curve: { color: '#ffffff', opacity: 1, dash: 'solid' as DashStyle, style: 'pass' as CurveStyle },
  highlight: { color: '#facc15', opacity: 0.85, dash: 'solid' as DashStyle, style: 'circle' as HighlightStyle },
  cone: { color: '#38bdf8', opacity: 0.35, dash: 'solid' as DashStyle },
}
const LABEL_COLOR = '#ffffff'
const HANDLE_RADIUS = 7

function setDash(ctx: CanvasRenderingContext2D, dash: DashStyle) {
  if (dash === 'dashed') ctx.setLineDash([14, 10])
  else if (dash === 'dotted') ctx.setLineDash([3, 7])
  else ctx.setLineDash([])
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

export interface AnnotationCanvasHandle {
  /** Commits any in-progress geometry (a zone that hasn't been double-clicked to close yet) as a real shape, or discards it if incomplete. Call right before saving. */
  finalizePending: () => void
}

const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, AnnotationCanvasProps>(function AnnotationCanvas({
  shapes, editable, tool,
  onAddShape, onUpdateShape, onRemoveShape, onRequestLabelText, onToggleVideo,
}, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draftPoints = useRef<NPoint[]>([]) // in-progress zone/curve points
  const dragTarget = useRef<{ shapeId: string; handle: string | number } | null>(null)
  const dragStartOffset = useRef<NPoint>([0, 0])
  const numberCounter = useRef(1)

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selectedShape = shapes.find(s => s.id === selectedId) ?? null

  const toCanvas = useCallback((p: NPoint): [number, number] => {
    const c = canvasRef.current; if (!c) return [0, 0]
    return [p[0] * c.width, p[1] * c.height]
  }, [])
  const toNorm = useCallback((cx: number, cy: number): NPoint => {
    const c = canvasRef.current; if (!c) return [0, 0]
    return [cx / c.width, cy / c.height]
  }, [])
  const xy = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect()
    const scale = canvasRef.current!.width / r.width
    return [(e.clientX - r.left) * scale, (e.clientY - r.top) * scale]
  }

  const draw = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    for (const s of shapes) {
      const isSelected = editable && s.id === selectedId

      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (cpts.length < 2) continue
        ctx.save()
        ctx.beginPath()
        cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
        if (cpts.length >= 3) ctx.closePath()
        ctx.globalAlpha = s.opacity
        ctx.fillStyle = s.color
        if (cpts.length >= 3) ctx.fill()
        ctx.globalAlpha = 1
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
        ctx.restore()
        if (editable) cpts.forEach(([x, y]) => drawHandle(ctx, x, y, s.color))
      }

      if (s.type === 'curve') {
        const p0 = toCanvas(s.from), p1 = toCanvas(s.control), p2 = toCanvas(s.to)
        ctx.save()
        ctx.globalAlpha = s.opacity
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
        ctx.restore()

        if (editable) {
          drawHandle(ctx, p0[0], p0[1], s.color)
          drawHandle(ctx, p2[0], p2[1], s.color)
          drawHandle(ctx, p1[0], p1[1], '#a78bfa') // control point, distinct color
        }
      }

      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const rx = s.radiusX * canvas.width
        const ry = s.radiusY * canvas.height

        if (s.style === 'spotlight') {
          ctx.save()
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
        ctx.globalAlpha = s.opacity
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 4 : 3
        ctx.beginPath()
        ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2)
        ctx.stroke()
        ctx.restore()

        if (s.playerName) {
          const fontSize = Math.max(11, canvas.height * 0.022)
          ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
          ctx.fillStyle = 'rgba(0,0,0,0.6)'
          const w = ctx.measureText(s.playerName).width
          ctx.fillRect(x - w / 2 - 5, y + ry + 4, w + 10, fontSize + 6)
          ctx.fillStyle = '#fff'
          ctx.textAlign = 'center'; ctx.textBaseline = 'top'
          ctx.fillText(s.playerName, x, y + ry + 7)
          ctx.textAlign = 'left'
        }

        if (editable) {
          drawHandle(ctx, x, y, s.color) // move
          drawHandle(ctx, x + rx, y, s.color) // width
          drawHandle(ctx, x, y + ry, s.color) // height
        }
      }

      if (s.type === 'cone') {
        const [ox, oy] = toCanvas(s.pos)
        const r = s.length * canvas.width
        const a0 = s.angle - s.width / 2
        const a1 = s.angle + s.width / 2
        ctx.save()
        ctx.globalAlpha = s.opacity
        ctx.fillStyle = s.color
        ctx.beginPath()
        ctx.moveTo(ox, oy)
        ctx.arc(ox, oy, r, a0, a1)
        ctx.closePath()
        ctx.fill()
        ctx.globalAlpha = 1
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 3 : 2
        ctx.stroke()
        ctx.restore()

        if (editable) {
          const tipX = ox + r * Math.cos(s.angle), tipY = oy + r * Math.sin(s.angle)
          const wX = ox + r * Math.cos(a1), wY = oy + r * Math.sin(a1)
          drawHandle(ctx, ox, oy, s.color) // move
          drawHandle(ctx, tipX, tipY, s.color) // direction + length
          drawHandle(ctx, wX, wY, '#a78bfa') // width, matches curve's control-point color
        }
      }

      if (s.type === 'label') {
        const [x, y] = toCanvas(s.pos)
        const fontSize = Math.max(14, canvas.height * 0.035)
        ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
        ctx.fillStyle = 'rgba(0,0,0,0.55)'
        const w = ctx.measureText(s.text).width
        ctx.fillRect(x - 6, y - fontSize, w + 12, fontSize + 10)
        ctx.fillStyle = s.color ?? LABEL_COLOR
        ctx.textBaseline = 'top'
        ctx.fillText(s.text, x, y - fontSize + 5)
      }

      if (s.type === 'number') {
        const [x, y] = toCanvas(s.pos)
        const r = Math.max(12, canvas.height * 0.022)
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = s.color
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.fillStyle = '#fff'
        ctx.font = `700 ${r}px 'DM Sans', sans-serif`
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
        ctx.fillText(String(s.value), x, y + 1)
        ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'
      }
    }

    // in-progress previews
    if (editable && tool === 'zone' && draftPoints.current.length > 0) {
      const cpts = draftPoints.current.map(toCanvas)
      ctx.beginPath()
      cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
      ctx.strokeStyle = DEFAULTS.zone.color
      ctx.lineWidth = 2
      ctx.setLineDash([6, 6])
      ctx.stroke()
      ctx.setLineDash([])
      cpts.forEach(([x, y]) => drawHandle(ctx, x, y, DEFAULTS.zone.color))
    }
    if (editable && tool === 'curve' && draftPoints.current.length === 1) {
      const [x, y] = toCanvas(draftPoints.current[0])
      drawHandle(ctx, x, y, DEFAULTS.curve.color)
    }
  }, [shapes, editable, tool, selectedId, toCanvas])

  function drawHandle(ctx: CanvasRenderingContext2D, x: number, y: number, color: string) {
    ctx.beginPath()
    ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
    ctx.fillStyle = '#fff'
    ctx.fill()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.stroke()
  }

  useEffect(() => {
    const container = containerRef.current, canvas = canvasRef.current
    if (!container || !canvas) return
    const resize = () => { canvas.width = container.clientWidth; canvas.height = container.clientHeight; draw() }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])
  useEffect(() => { draw() }, [draw])
  useEffect(() => {
    if (draftPoints.current.length >= 3) {
      // A zone was mid-draw (never double-clicked to close) and the tool
      // just changed out from under it — commit it rather than losing it.
      onAddShape({ id: Math.random().toString(36).slice(2, 10), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone })
    }
    draftPoints.current = []
    if (tool !== 'select') setSelectedId(null)
  }, [tool, onAddShape])
  useEffect(() => { if (selectedId && !shapes.find(s => s.id === selectedId)) setSelectedId(null) }, [shapes, selectedId])
  useEffect(() => {
    const maxNum = shapes.filter((s): s is Extract<Shape, { type: 'number' }> => s.type === 'number')
      .reduce((m, s) => Math.max(m, s.value), 0)
    numberCounter.current = maxNum + 1
  }, [shapes])

  const uid = () => Math.random().toString(36).slice(2, 10)

  useImperativeHandle(ref, () => ({
    finalizePending: () => {
      if (tool === 'zone' && draftPoints.current.length >= 3) {
        onAddShape({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone })
      }
      // A single placed curve-start point with no endpoint isn't a valid
      // shape (needs 2 clicks minimum) — nothing meaningful to commit there,
      // just discard it below.
      draftPoints.current = []
    },
  }), [tool, onAddShape])

  const hitHandle = useCallback((cx: number, cy: number) => {
    const c = canvasRef.current!
    const scale = c.width / c.getBoundingClientRect().width
    const t2 = (HANDLE_RADIUS * scale) ** 2
    for (const s of shapes) {
      if (s.type === 'zone') {
        for (let i = 0; i < s.points.length; i++) {
          const [x, y] = toCanvas(s.points[i])
          if (dist2(x, y, cx, cy) <= t2) return { shapeId: s.id, handle: i }
        }
      }
      if (s.type === 'curve') {
        const [fx, fy] = toCanvas(s.from), [tx, ty] = toCanvas(s.to), [cx2, cy2] = toCanvas(s.control)
        if (dist2(fx, fy, cx, cy) <= t2) return { shapeId: s.id, handle: 'from' }
        if (dist2(tx, ty, cx, cy) <= t2) return { shapeId: s.id, handle: 'to' }
        if (dist2(cx2, cy2, cx, cy) <= t2) return { shapeId: s.id, handle: 'control' }
      }
      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const rx = s.radiusX * c.width, ry = s.radiusY * c.height
        if (dist2(x, y, cx, cy) <= t2) return { shapeId: s.id, handle: 'move' }
        if (dist2(x + rx, y, cx, cy) <= t2) return { shapeId: s.id, handle: 'radiusX' }
        if (dist2(x, y + ry, cx, cy) <= t2) return { shapeId: s.id, handle: 'radiusY' }
      }
      if (s.type === 'cone') {
        const [ox, oy] = toCanvas(s.pos)
        const r = s.length * c.width
        const tipX = ox + r * Math.cos(s.angle), tipY = oy + r * Math.sin(s.angle)
        const a1 = s.angle + s.width / 2
        const wX = ox + r * Math.cos(a1), wY = oy + r * Math.sin(a1)
        if (dist2(ox, oy, cx, cy) <= t2) return { shapeId: s.id, handle: 'move' }
        if (dist2(tipX, tipY, cx, cy) <= t2) return { shapeId: s.id, handle: 'direction' }
        if (dist2(wX, wY, cx, cy) <= t2) return { shapeId: s.id, handle: 'width' }
      }
    }
    return null
  }, [shapes, toCanvas])

  const hitBody = useCallback((cx: number, cy: number): string | null => {
    const c = canvasRef.current!
    for (let i = shapes.length - 1; i >= 0; i--) {
      const s = shapes[i]
      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (pointInPolygon(cx, cy, cpts)) return s.id
      }
      if (s.type === 'curve') {
        const p0 = toCanvas(s.from), p1 = toCanvas(s.control), p2 = toCanvas(s.to)
        const scale = c.width / c.getBoundingClientRect().width
        for (let t = 0; t <= 1; t += 0.05) {
          const [x, y] = bezierPoint(t, p0, p1, p2)
          if (dist2(x, y, cx, cy) <= (10 * scale) ** 2) return s.id
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
        if (dist2(x, y, cx, cy) <= (25 * (c.width / c.getBoundingClientRect().width)) ** 2) return s.id
      }
    }
    return null
  }, [shapes, toCanvas])

  return (
    <div ref={containerRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'auto' }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: editable && tool !== 'select' ? 'crosshair' : 'default' }}
        onDoubleClick={() => {
          if (!editable || tool !== 'zone' || draftPoints.current.length < 3) return
          onAddShape({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone })
          draftPoints.current = []
          draw()
        }}
        onMouseDown={e => {
          if (!editable) { onToggleVideo?.(); return }
          const [cx, cy] = xy(e)

          if (tool === 'select') {
            const handleHit = hitHandle(cx, cy)
            if (handleHit) { dragTarget.current = handleHit; setSelectedId(handleHit.shapeId); return }
            const bodyHit = hitBody(cx, cy)
            setSelectedId(bodyHit)
            if (bodyHit) {
              const shape = shapes.find(s => s.id === bodyHit)!
              const p = toNorm(cx, cy)
              const anchor = shape.type === 'highlight' || shape.type === 'label' || shape.type === 'number' || shape.type === 'cone' ? shape.pos
                : shape.type === 'curve' ? shape.from : shape.points[0]
              dragStartOffset.current = [p[0] - anchor[0], p[1] - anchor[1]]
              dragTarget.current = { shapeId: bodyHit, handle: 'move' }
            }
            return
          }

          if (tool === 'zone') {
            draftPoints.current = [...draftPoints.current, toNorm(cx, cy)]
            draw()
            return
          }

          if (tool === 'curve') {
            const p = toNorm(cx, cy)
            if (draftPoints.current.length === 0) { draftPoints.current = [p]; draw() }
            else {
              const from = draftPoints.current[0]
              const control: NPoint = [(from[0] + p[0]) / 2, (from[1] + p[1]) / 2]
              onAddShape({ id: uid(), type: 'curve', from, control, to: p, ...DEFAULTS.curve })
              draftPoints.current = []
            }
            return
          }

          if (tool === 'highlight') {
            const p = toNorm(cx, cy)
            onAddShape({ id: uid(), type: 'highlight', pos: p, radiusX: 0.05, radiusY: 0.07, ...DEFAULTS.highlight })
            return
          }

          if (tool === 'cone') {
            if (draftPoints.current.length === 0) {
              draftPoints.current = [toNorm(cx, cy)]
              draw()
            } else {
              const originNorm = draftPoints.current[0]
              const [ox, oy] = toCanvas(originNorm)
              const angle = Math.atan2(cy - oy, cx - ox)
              const length = Math.max(0.03, Math.hypot(cx - ox, cy - oy) / canvasRef.current!.width)
              onAddShape({ id: uid(), type: 'cone', pos: originNorm, angle, length, width: Math.PI / 3, ...DEFAULTS.cone })
              draftPoints.current = []
            }
            return
          }

          if (tool === 'number') {
            const p = toNorm(cx, cy)
            onAddShape({ id: uid(), type: 'number', pos: p, value: numberCounter.current, color: '#0f2972' })
            return
          }

          if (tool === 'label') {
            const p = toNorm(cx, cy)
            if (onRequestLabelText) onRequestLabelText(p)
            else {
              const text = window.prompt('Label text')
              if (text) onAddShape({ id: uid(), type: 'label', pos: p, text, color: LABEL_COLOR })
            }
          }
        }}
        onMouseMove={e => {
          if (!editable || !dragTarget.current) return
          const [cx, cy] = xy(e)
          const p = toNorm(cx, cy)
          const { shapeId, handle } = dragTarget.current
          const shape = shapes.find(s => s.id === shapeId)
          if (!shape) return

          if (shape.type === 'zone' && typeof handle === 'number') {
            const points = [...shape.points]; points[handle] = p
            onUpdateShape(shapeId, { points } as Partial<Shape>)
          }
          if (shape.type === 'zone' && handle === 'move') {
            const dx = p[0] - dragStartOffset.current[0] - shape.points[0][0]
            const dy = p[1] - dragStartOffset.current[1] - shape.points[0][1]
            onUpdateShape(shapeId, { points: shape.points.map(([x, y]) => [x + dx, y + dy]) } as Partial<Shape>)
          }
          if (shape.type === 'curve') {
            if (handle === 'from') onUpdateShape(shapeId, { from: p } as Partial<Shape>)
            if (handle === 'to') onUpdateShape(shapeId, { to: p } as Partial<Shape>)
            if (handle === 'control') onUpdateShape(shapeId, { control: p } as Partial<Shape>)
            if (handle === 'move') {
              const dx = p[0] - dragStartOffset.current[0] - shape.from[0]
              const dy = p[1] - dragStartOffset.current[1] - shape.from[1]
              onUpdateShape(shapeId, {
                from: [shape.from[0] + dx, shape.from[1] + dy],
                to: [shape.to[0] + dx, shape.to[1] + dy],
                control: [shape.control[0] + dx, shape.control[1] + dy],
              } as Partial<Shape>)
            }
          }
          if (shape.type === 'highlight') {
            if (handle === 'move') onUpdateShape(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
            if (handle === 'radiusX') onUpdateShape(shapeId, { radiusX: Math.max(0.015, p[0] - shape.pos[0]) } as Partial<Shape>)
            if (handle === 'radiusY') onUpdateShape(shapeId, { radiusY: Math.max(0.015, p[1] - shape.pos[1]) } as Partial<Shape>)
          }
          if (shape.type === 'cone') {
            if (handle === 'move') {
              onUpdateShape(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
            }
            if (handle === 'direction') {
              const dx = p[0] - shape.pos[0], dy = p[1] - shape.pos[1]
              const angle = Math.atan2(dy, dx)
              const length = Math.max(0.03, Math.hypot(dx, dy))
              onUpdateShape(shapeId, { angle, length } as Partial<Shape>)
            }
            if (handle === 'width') {
              const dx = p[0] - shape.pos[0], dy = p[1] - shape.pos[1]
              let a = Math.atan2(dy, dx) - shape.angle
              while (a > Math.PI) a -= 2 * Math.PI
              while (a < -Math.PI) a += 2 * Math.PI
              const width = Math.max(0.15, Math.min(Math.PI * 0.9, Math.abs(a) * 2))
              onUpdateShape(shapeId, { width } as Partial<Shape>)
            }
          }
          if ((shape.type === 'label' || shape.type === 'number') && handle === 'move') {
            onUpdateShape(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
          }
        }}
        onMouseUp={() => { dragTarget.current = null }}
        onMouseLeave={() => { dragTarget.current = null }}
      />

      {editable && selectedShape && (
        <ShapeSettingsPanel
          shape={selectedShape}
          onChange={patch => onUpdateShape(selectedShape.id, patch)}
          onRemove={() => { onRemoveShape(selectedShape.id); setSelectedId(null) }}
          onClose={() => setSelectedId(null)}
        />
      )}
    </div>
  )
})

export default AnnotationCanvas
