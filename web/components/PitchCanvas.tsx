'use client'

import { useRef, useEffect, useCallback } from 'react'

export type Point = [number, number] // image pixel coords

interface PitchCanvasProps {
  imageUrl: string
  imageNaturalWidth: number
  imageNaturalHeight: number
  points: Point[]
  onAddPoint: (p: Point) => void
  onUpdatePoint: (index: number, p: Point) => void
  closed: boolean
}

const HANDLE_RADIUS = 10
const LINE_COLOR    = '#4ade80'
const FILL_COLOR    = 'rgba(74, 222, 128, 0.15)'
const POINT_COLOR   = '#4ade80'
const POINT_STROKE  = '#166534'

export default function PitchCanvas({
  imageUrl, imageNaturalWidth, imageNaturalHeight,
  points, onAddPoint, onUpdatePoint, closed,
}: PitchCanvasProps) {
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef       = useRef<HTMLImageElement | null>(null)
  const imgLoaded    = useRef(false)
  const dragIdx      = useRef(-1)

  const getScale = useCallback(() => {
    const c = canvasRef.current
    return c ? imageNaturalWidth / c.width : 1
  }, [imageNaturalWidth])

  const toCanvas = useCallback((p: Point): [number, number] => {
    const s = getScale()
    return [p[0] / s, p[1] / s]
  }, [getScale])

  const toImage = useCallback((cx: number, cy: number): Point => {
    const s = getScale()
    return [Math.round(cx * s), Math.round(cy * s)]
  }, [getScale])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (imgLoaded.current && imgRef.current)
      ctx.drawImage(imgRef.current, 0, 0, canvas.width, canvas.height)

    if (points.length === 0) return
    const cpts = points.map(toCanvas)

    if (closed && points.length >= 3) {
      ctx.beginPath()
      cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
      ctx.closePath()
      ctx.fillStyle = FILL_COLOR
      ctx.fill()
    }

    ctx.beginPath()
    cpts.forEach(([x, y], i) => i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y))
    if (closed && points.length >= 3) ctx.closePath()
    ctx.strokeStyle = LINE_COLOR
    ctx.lineWidth = 2
    ctx.stroke()

    cpts.forEach(([x, y], i) => {
      ctx.beginPath()
      ctx.arc(x, y, 7, 0, Math.PI * 2)
      ctx.fillStyle = dragIdx.current === i ? '#fff' : POINT_COLOR
      ctx.fill()
      ctx.strokeStyle = POINT_STROKE
      ctx.lineWidth = 1.5
      ctx.stroke()
      ctx.fillStyle = '#000'
      ctx.font = 'bold 9px DM Sans, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(String(i + 1), x, y)
    })
  }, [points, toCanvas, closed])

  useEffect(() => {
    if (!imageUrl) return
    imgLoaded.current = false
    const img = new Image()
    img.onload = () => { imgRef.current = img; imgLoaded.current = true; draw() }
    img.src = imageUrl
  }, [imageUrl, draw])

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const resize = () => {
      const w = container.clientWidth
      canvas.width = w
      canvas.height = Math.round(w * imageNaturalHeight / imageNaturalWidth)
      draw()
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    return () => ro.disconnect()
  }, [imageNaturalWidth, imageNaturalHeight, draw])

  useEffect(() => { draw() }, [draw])

  const hitTest = useCallback((cx: number, cy: number) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const scale = canvasRef.current!.width / rect.width
    const t2 = (HANDLE_RADIUS * scale) ** 2
    for (let i = points.length - 1; i >= 0; i--) {
      const [px, py] = toCanvas(points[i])
      if ((px - cx) ** 2 + (py - cy) ** 2 <= t2) return i
    }
    return -1
  }, [points, toCanvas])

  const xy = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const r = canvasRef.current!.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  return (
    <div ref={containerRef} style={{
      width: '100%', borderRadius: 8, overflow: 'hidden',
      background: '#111', cursor: 'crosshair',
    }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%' }}
        onMouseDown={e => {
          const [cx, cy] = xy(e)
          const hit = hitTest(cx, cy)
          if (hit >= 0) { dragIdx.current = hit; return }
          if (!closed) onAddPoint(toImage(cx, cy))
        }}
        onMouseMove={e => {
          if (dragIdx.current < 0) return
          const [cx, cy] = xy(e)
          onUpdatePoint(dragIdx.current, toImage(cx, cy))
        }}
        onMouseUp={() => { dragIdx.current = -1; draw() }}
        onMouseLeave={() => { dragIdx.current = -1; draw() }}
      />
    </div>
  )
}
