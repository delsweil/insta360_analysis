'use client'

// components/AnnotationCanvas.tsx
// Draws/plays back spatial tactical shapes (zones, arrows, labels, highlights)
// layered over the YouTube iframe. Hand-rolled Canvas 2D, matching
// PitchCanvas.tsx's conventions — normalized 0..1 coords instead of
// PitchCanvas's image-pixel coords (no "natural size" for a live video
// overlay watched on different screens).

import { useRef, useEffect, useCallback, useState } from 'react'
import ShapeSettingsPanel from './ShapeSettingsPanel'

export type NPoint = [number, number] // normalized 0..1, relative to video frame
export type DashStyle = 'solid' | 'dashed' | 'dotted'

interface StyleProps {
  color: string
  opacity: number
  dash: DashStyle
}

export type Shape =
  | ({ id: string; type: 'zone'; points: NPoint[] } & StyleProps) // 4 corners, filled quad
  | ({ id: string; type: 'arrow'; from: NPoint; to: NPoint } & StyleProps)
  | { id: string; type: 'label'; pos: NPoint; text: string; color: string }
  | ({ id: string; type: 'highlight'; pos: NPoint; radius: number; playerName?: string } & StyleProps) // spotlight circle on a player

export type Tool = 'select' | 'zone' | 'arrow' | 'label' | 'highlight'

interface AnnotationCanvasProps {
  shapes: Shape[]
  editable: boolean
  tool: Tool
  onAddShape: (s: Shape) => void
  onUpdateShape: (id: string, patch: Partial<Shape>) => void
  onRemoveShape: (id: string) => void
  /** Called instead of a browser prompt() when the label tool places a point — lets the page render its own inline input. */
  onRequestLabelText?: (pos: NPoint) => void
}

const DEFAULTS: Record<'zone' | 'arrow' | 'highlight', StyleProps> = {
  zone:      { color: '#4ade80', opacity: 0.3, dash: 'solid' },
  arrow:     { color: '#ffffff', opacity: 1,   dash: 'solid' },
  highlight: { color: '#facc15', opacity: 0.8, dash: 'dashed' },
}
const LABEL_COLOR = '#ffffff'
const HANDLE_RADIUS = 7

function setDash(ctx: CanvasRenderingContext2D, dash: DashStyle) {
  if (dash === 'dashed') ctx.setLineDash([14, 10])
  else if (dash === 'dotted') ctx.setLineDash([3, 7])
  else ctx.setLineDash([])
}

function dist2(ax: number, ay: number, bx: number, by: number) {
  return (ax - bx) ** 2 + (ay - by) ** 2
}

// distance from point p to segment ab, squared
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
    const [xi, yi] = pts[i]
    const [xj, yj] = pts[j]
    const intersect = yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    if (intersect) inside = !inside
  }
  return inside
}

export default function AnnotationCanvas({
  shapes, editable, tool,
  onAddShape, onUpdateShape, onRemoveShape, onRequestLabelText,
}: AnnotationCanvasProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draftPoints  = useRef<NPoint[]>([]) // in-progress zone/arrow points
  const dragTarget   = useRef<{ shapeId: string; handle: 'from' | 'to' | 'move' | 'resize' | number } | null>(null)
  const dragStartOffset = useRef<NPoint>([0, 0]) // for whole-shape moves

  const [selectedId, setSelectedId] = useState<string | null>(null)
  const selectedShape = shapes.find(s => s.id === selectedId) ?? null

  // ── coordinate helpers ──────────────────────────────────────────
  const toCanvas = useCallback((p: NPoint): [number, number] => {
    const c = canvasRef.current
    if (!c) return [0, 0]
    return [p[0] * c.width, p[1] * c.height]
  }, [])

  const toNorm = useCallback((cx: number, cy: number): NPoint => {
    const c = canvasRef.current
    if (!c) return [0, 0]
    return [cx / c.width, cy / c.height]
  }, [])

  const xy = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect()
    const scale = canvasRef.current!.width / r.width
    return [(e.clientX - r.left) * scale, (e.clientY - r.top) * scale]
  }

  // ── draw ─────────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
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

        if (editable) {
          cpts.forEach(([x, y]) => {
            ctx.beginPath()
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
            ctx.fillStyle = '#fff'
            ctx.fill()
            ctx.strokeStyle = s.color
            ctx.lineWidth = 1.5
            ctx.stroke()
          })
        }
      }

      if (s.type === 'arrow') {
        const [fx, fy] = toCanvas(s.from)
        const [tx, ty] = toCanvas(s.to)
        ctx.save()
        ctx.globalAlpha = s.opacity
        ctx.shadowColor = 'rgba(0,0,0,0.6)'
        ctx.shadowBlur = 4
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 6 : 4
        ctx.lineCap = 'round'
        setDash(ctx, s.dash)
        ctx.beginPath()
        ctx.moveTo(fx, fy)
        ctx.lineTo(tx, ty)
        ctx.stroke()
        ctx.setLineDash([])

        const angle = Math.atan2(ty - fy, tx - fx)
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
          for (const [x, y] of [[fx, fy], [tx, ty]] as [number, number][]) {
            ctx.beginPath()
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
            ctx.fillStyle = '#fff'
            ctx.fill()
            ctx.strokeStyle = '#0f2972'
            ctx.lineWidth = 1.5
            ctx.stroke()
          }
        }
      }

      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const r = s.radius * canvas.width
        ctx.save()
        ctx.globalAlpha = s.opacity
        setDash(ctx, s.dash)
        ctx.strokeStyle = s.color
        ctx.lineWidth = isSelected ? 4 : 3
        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.stroke()
        ctx.restore()

        if (s.playerName) {
          const fontSize = Math.max(11, canvas.height * 0.022)
          ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
          ctx.fillStyle = 'rgba(0,0,0,0.6)'
          const w = ctx.measureText(s.playerName).width
          ctx.fillRect(x - w / 2 - 5, y + r + 4, w + 10, fontSize + 6)
          ctx.fillStyle = '#fff'
          ctx.textAlign = 'center'
          ctx.textBaseline = 'top'
          ctx.fillText(s.playerName, x, y + r + 7)
          ctx.textAlign = 'left'
        }

        if (editable) {
          // move handle (center) + resize handle (right edge of circle)
          ctx.beginPath()
          ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
          ctx.fillStyle = '#fff'
          ctx.fill()
          ctx.strokeStyle = s.color
          ctx.lineWidth = 1.5
          ctx.stroke()

          ctx.beginPath()
          ctx.arc(x + r, y, HANDLE_RADIUS, 0, Math.PI * 2)
          ctx.fillStyle = '#fff'
          ctx.fill()
          ctx.strokeStyle = s.color
          ctx.lineWidth = 1.5
          ctx.stroke()
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
    }

    // in-progress zone preview
    if (editable && tool === 'zone' && draftPoints.current.length > 0) {
      draftPoints.current.forEach(p => {
        const [x, y] = toCanvas(p)
        ctx.beginPath()
        ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
        ctx.fillStyle = DEFAULTS.zone.color
        ctx.fill()
      })
    }
    // in-progress arrow start marker
    if (editable && tool === 'arrow' && draftPoints.current.length === 1) {
      const [x, y] = toCanvas(draftPoints.current[0])
      ctx.beginPath()
      ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = DEFAULTS.arrow.color
      ctx.fill()
    }
  }, [shapes, editable, tool, selectedId, toCanvas])

  // resize canvas to match container, same pattern as PitchCanvas
  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const resize = () => {
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
      draw()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  useEffect(() => { draw() }, [draw])

  // reset in-progress drawing whenever the active tool changes; deselect too
  useEffect(() => {
    draftPoints.current = []
    if (tool !== 'select') setSelectedId(null)
  }, [tool])

  // deselect if the shape we had selected got removed/changed lists
  useEffect(() => {
    if (selectedId && !shapes.find(s => s.id === selectedId)) setSelectedId(null)
  }, [shapes, selectedId])

  const uid = () => Math.random().toString(36).slice(2, 10)

  // hit-test handles first (for dragging), for select tool
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
      if (s.type === 'arrow') {
        const [fx, fy] = toCanvas(s.from)
        const [tx, ty] = toCanvas(s.to)
        if (dist2(fx, fy, cx, cy) <= t2) return { shapeId: s.id, handle: 'from' as const }
        if (dist2(tx, ty, cx, cy) <= t2) return { shapeId: s.id, handle: 'to' as const }
      }
      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const r = s.radius * c.width
        if (dist2(x, y, cx, cy) <= t2) return { shapeId: s.id, handle: 'move' as const }
        if (dist2(x + r, y, cx, cy) <= t2) return { shapeId: s.id, handle: 'resize' as const }
      }
    }
    return null
  }, [shapes, toCanvas])

  // hit-test shape bodies (for click-to-select, when not on a handle)
  const hitBody = useCallback((cx: number, cy: number): string | null => {
    const c = canvasRef.current!
    for (let i = shapes.length - 1; i >= 0; i--) {
      const s = shapes[i]
      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (pointInPolygon(cx, cy, cpts)) return s.id
      }
      if (s.type === 'arrow') {
        const [fx, fy] = toCanvas(s.from)
        const [tx, ty] = toCanvas(s.to)
        const scale = c.width / c.getBoundingClientRect().width
        if (distToSegment2(cx, cy, fx, fy, tx, ty) <= (10 * scale) ** 2) return s.id
      }
      if (s.type === 'highlight') {
        const [x, y] = toCanvas(s.pos)
        const r = s.radius * c.width
        const d2 = dist2(x, y, cx, cy)
        if (d2 <= r * r) return s.id
      }
      if (s.type === 'label') {
        const [x, y] = toCanvas(s.pos)
        const ctx = c.getContext('2d')!
        const fontSize = Math.max(14, c.height * 0.035)
        ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
        const w = ctx.measureText(s.text).width
        if (cx >= x - 6 && cx <= x + w + 6 && cy >= y - fontSize && cy <= y + 10) return s.id
      }
    }
    return null
  }, [shapes, toCanvas])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute', inset: 0,
        // Always 'auto', not just while editing: the mouse must never
        // actually reach the cross-origin iframe underneath, or YouTube's
        // own hover-triggered play/pause overlay activates regardless of
        // the controls=0 URL parameter (a real limitation of their embed,
        // not something fixable via config). Clicks are effectively
        // swallowed harmlessly here when !editable, since the mouse
        // handlers below all early-return in that case — the app's own
        // custom scrubber/controls remain the way to control playback.
        pointerEvents: 'auto',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: editable && tool !== 'select' ? 'crosshair' : 'default' }}
        onMouseDown={e => {
          if (!editable) return
          const [cx, cy] = xy(e)

          if (tool === 'select') {
            const handleHit = hitHandle(cx, cy)
            if (handleHit) {
              dragTarget.current = handleHit
              setSelectedId(handleHit.shapeId)
              return
            }
            const bodyHit = hitBody(cx, cy)
            setSelectedId(bodyHit)
            if (bodyHit) {
              const shape = shapes.find(s => s.id === bodyHit)!
              const p = toNorm(cx, cy)
              const anchor = shape.type === 'highlight' || shape.type === 'label' ? shape.pos
                : shape.type === 'arrow' ? shape.from
                : shape.points[0]
              dragStartOffset.current = [p[0] - anchor[0], p[1] - anchor[1]]
              dragTarget.current = { shapeId: bodyHit, handle: 'move' }
            }
            return
          }

          if (tool === 'zone') {
            const p = toNorm(cx, cy)
            draftPoints.current = [...draftPoints.current, p]
            if (draftPoints.current.length === 4) {
              onAddShape({ id: uid(), type: 'zone', points: draftPoints.current, ...DEFAULTS.zone })
              draftPoints.current = []
            }
            draw()
            return
          }

          if (tool === 'arrow') {
            const p = toNorm(cx, cy)
            if (draftPoints.current.length === 0) {
              draftPoints.current = [p]
            } else {
              onAddShape({ id: uid(), type: 'arrow', from: draftPoints.current[0], to: p, ...DEFAULTS.arrow })
              draftPoints.current = []
            }
            draw()
            return
          }

          if (tool === 'highlight') {
            const p = toNorm(cx, cy)
            onAddShape({ id: uid(), type: 'highlight', pos: p, radius: 0.05, ...DEFAULTS.highlight })
            return
          }

          if (tool === 'label') {
            const p = toNorm(cx, cy)
            if (onRequestLabelText) {
              onRequestLabelText(p)
            } else {
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
            const points = [...shape.points]
            points[handle] = p
            onUpdateShape(shapeId, { points } as Partial<Shape>)
          }
          if (shape.type === 'arrow' && (handle === 'from' || handle === 'to')) {
            onUpdateShape(shapeId, { [handle]: p } as Partial<Shape>)
          }
          if (shape.type === 'highlight' && handle === 'move') {
            onUpdateShape(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
          }
          if (shape.type === 'highlight' && handle === 'resize') {
            const r = Math.max(0.015, p[0] - shape.pos[0])
            onUpdateShape(shapeId, { radius: r } as Partial<Shape>)
          }
          if ((shape.type === 'label') && handle === 'move') {
            onUpdateShape(shapeId, { pos: [p[0] - dragStartOffset.current[0], p[1] - dragStartOffset.current[1]] } as Partial<Shape>)
          }
          if (shape.type === 'zone' && handle === 'move') {
            const dx = p[0] - dragStartOffset.current[0] - shape.points[0][0]
            const dy = p[1] - dragStartOffset.current[1] - shape.points[0][1]
            onUpdateShape(shapeId, { points: shape.points.map(([x, y]) => [x + dx, y + dy]) } as Partial<Shape>)
          }
          if (shape.type === 'arrow' && handle === 'move') {
            const dx = p[0] - dragStartOffset.current[0] - shape.from[0]
            const dy = p[1] - dragStartOffset.current[1] - shape.from[1]
            onUpdateShape(shapeId, { from: [shape.from[0] + dx, shape.from[1] + dy], to: [shape.to[0] + dx, shape.to[1] + dy] } as Partial<Shape>)
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
}
