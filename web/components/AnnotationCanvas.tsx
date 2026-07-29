'use client'

// components/AnnotationCanvas.tsx
// Draws/plays back spatial tactical shapes (zones, arrows, labels) layered
// over the YouTube iframe. Follows the same hand-rolled canvas pattern as
// PitchCanvas.tsx — no external drawing library, normalized 0..1 coords
// instead of PitchCanvas's image-pixel coords (there's no "natural size"
// for a live video overlay watched on different screens).

import { useRef, useEffect, useCallback } from 'react'

export type NPoint = [number, number] // normalized 0..1, relative to video frame

export type Shape =
  | { id: string; type: 'zone'; points: NPoint[] } // 4 corners, drawn as a filled quad
  | { id: string; type: 'arrow'; from: NPoint; to: NPoint; dashed?: boolean }
  | { id: string; type: 'label'; pos: NPoint; text: string }

export type Tool = 'select' | 'zone' | 'arrow' | 'label'

interface AnnotationCanvasProps {
  shapes: Shape[]
  editable: boolean
  tool: Tool
  onAddShape: (s: Shape) => void
  onUpdateShape: (id: string, patch: Partial<Shape>) => void
  /** Called instead of a browser prompt() when the label tool places a point — lets the page render its own inline input. */
  onRequestLabelText?: (pos: NPoint) => void
}

const ZONE_STROKE  = '#4ade80'
const ZONE_FILL    = 'rgba(74, 222, 128, 0.2)'
const ARROW_COLOR  = '#ffffff'
const ARROW_SHADOW = 'rgba(0,0,0,0.6)'
const LABEL_COLOR  = '#ffffff'
const HANDLE_RADIUS = 7

export default function AnnotationCanvas({
  shapes, editable, tool,
  onAddShape, onUpdateShape, onRequestLabelText,
}: AnnotationCanvasProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const draftPoints   = useRef<NPoint[]>([]) // in-progress zone/arrow points
  const dragTarget    = useRef<{ shapeId: string; handle: 'from' | 'to' | number } | null>(null)

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
      if (s.type === 'zone') {
        const cpts = s.points.map(toCanvas)
        if (cpts.length < 2) continue
        ctx.beginPath()
        cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
        if (cpts.length >= 3) ctx.closePath()
        ctx.fillStyle = ZONE_FILL
        if (cpts.length >= 3) ctx.fill()
        ctx.strokeStyle = ZONE_STROKE
        ctx.lineWidth = 2
        ctx.stroke()

        if (editable) {
          cpts.forEach(([x, y]) => {
            ctx.beginPath()
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
            ctx.fillStyle = '#fff'
            ctx.fill()
            ctx.strokeStyle = ZONE_STROKE
            ctx.lineWidth = 1.5
            ctx.stroke()
          })
        }
      }

      if (s.type === 'arrow') {
        const [fx, fy] = toCanvas(s.from)
        const [tx, ty] = toCanvas(s.to)
        ctx.save()
        ctx.shadowColor = ARROW_SHADOW
        ctx.shadowBlur = 4
        ctx.strokeStyle = ARROW_COLOR
        ctx.lineWidth = 4
        ctx.lineCap = 'round'
        if (s.dashed) ctx.setLineDash([14, 10])
        ctx.beginPath()
        ctx.moveTo(fx, fy)
        ctx.lineTo(tx, ty)
        ctx.stroke()
        ctx.setLineDash([])

        // arrowhead
        const angle = Math.atan2(ty - fy, tx - fx)
        const head = 14
        ctx.beginPath()
        ctx.moveTo(tx, ty)
        ctx.lineTo(tx - head * Math.cos(angle - Math.PI / 6), ty - head * Math.sin(angle - Math.PI / 6))
        ctx.lineTo(tx - head * Math.cos(angle + Math.PI / 6), ty - head * Math.sin(angle + Math.PI / 6))
        ctx.closePath()
        ctx.fillStyle = ARROW_COLOR
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

      if (s.type === 'label') {
        const [x, y] = toCanvas(s.pos)
        const fontSize = Math.max(14, canvas.height * 0.035)
        ctx.font = `700 ${fontSize}px 'DM Sans', sans-serif`
        ctx.fillStyle = 'rgba(0,0,0,0.55)'
        const w = ctx.measureText(s.text).width
        ctx.fillRect(x - 6, y - fontSize, w + 12, fontSize + 10)
        ctx.fillStyle = LABEL_COLOR
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
        ctx.fillStyle = ZONE_STROKE
        ctx.fill()
      })
    }
    // in-progress arrow start marker
    if (editable && tool === 'arrow' && draftPoints.current.length === 1) {
      const [x, y] = toCanvas(draftPoints.current[0])
      ctx.beginPath()
      ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2)
      ctx.fillStyle = ARROW_COLOR
      ctx.fill()
    }
  }, [shapes, editable, tool, toCanvas])

  // resize canvas to match container, same pattern as PitchCanvas
  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const resize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      canvas.width = w
      canvas.height = h
      draw()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [draw])

  useEffect(() => { draw() }, [draw])

  // reset in-progress drawing whenever the active tool changes
  useEffect(() => { draftPoints.current = [] }, [tool])

  const uid = () => Math.random().toString(36).slice(2, 10)

  // hit-test existing shape handles, for dragging
  const hitTest = useCallback((cx: number, cy: number) => {
    const c = canvasRef.current!
    const scale = c.width / c.getBoundingClientRect().width
    const t2 = (HANDLE_RADIUS * scale) ** 2
    for (const s of shapes) {
      if (s.type === 'zone') {
        for (let i = 0; i < s.points.length; i++) {
          const [x, y] = toCanvas(s.points[i])
          if ((x - cx) ** 2 + (y - cy) ** 2 <= t2) return { shapeId: s.id, handle: i }
        }
      }
      if (s.type === 'arrow') {
        const [fx, fy] = toCanvas(s.from)
        const [tx, ty] = toCanvas(s.to)
        if ((fx - cx) ** 2 + (fy - cy) ** 2 <= t2) return { shapeId: s.id, handle: 'from' as const }
        if ((tx - cx) ** 2 + (ty - cy) ** 2 <= t2) return { shapeId: s.id, handle: 'to' as const }
      }
    }
    return null
  }, [shapes, toCanvas])

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute', inset: 0,
        pointerEvents: editable ? 'auto' : 'none',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: editable && tool !== 'select' ? 'crosshair' : 'default' }}
        onMouseDown={e => {
          if (!editable) return
          const [cx, cy] = xy(e)

          if (tool === 'select') {
            const hit = hitTest(cx, cy)
            if (hit) dragTarget.current = hit
            return
          }

          if (tool === 'zone') {
            const p = toNorm(cx, cy)
            draftPoints.current = [...draftPoints.current, p]
            if (draftPoints.current.length === 4) {
              onAddShape({ id: uid(), type: 'zone', points: draftPoints.current })
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
              onAddShape({ id: uid(), type: 'arrow', from: draftPoints.current[0], to: p })
              draftPoints.current = []
            }
            draw()
            return
          }

          if (tool === 'label') {
            const p = toNorm(cx, cy)
            if (onRequestLabelText) {
              onRequestLabelText(p)
            } else {
              const text = window.prompt('Label text')
              if (text) onAddShape({ id: uid(), type: 'label', pos: p, text })
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
        }}
        onMouseUp={() => { dragTarget.current = null }}
        onMouseLeave={() => { dragTarget.current = null }}
      />
    </div>
  )
}
