'use client'

// components/TrimStrip.tsx
// A small, zoomed-in horizontal strip (not the full match timeline) showing
// a short window before a tactical annotation's pinned frame. Drag the
// handle to set where the scene should actually start playing from when
// someone jumps to this moment — the pin itself (the annotation's frame)
// never moves.

import { useRef, useState, useCallback, useEffect } from 'react'

interface TrimStripProps {
  /** The annotation's fixed frame — where the shapes are drawn, never moves. */
  timestampSec: number
  /** Current scene-start time (absolute seconds into the video). */
  value: number
  /** Committed when the drag ends. */
  onChange: (sec: number) => void
  /** Called continuously while dragging, so the video preview can scrub live. */
  onScrub: (sec: number) => void
  /** How many seconds of run-up the strip shows before the pin. */
  windowSec?: number
}

const STRIP_WIDTH = 260
const TAIL_SEC = 2 // small buffer shown after the pin, so it isn't flush against the edge

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec))
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}:${String(r).padStart(2, '0')}`
}

export default function TrimStrip({ timestampSec, value, onChange, onScrub, windowSec = 20 }: TrimStripProps) {
  const stripRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState(false)
  const [localValue, setLocalValue] = useState(value)

  useEffect(() => { if (!dragging) setLocalValue(value) }, [value, dragging])

  const rangeStart = timestampSec - windowSec
  const rangeEnd = timestampSec + TAIL_SEC
  const rangeLen = rangeEnd - rangeStart

  const secToX = useCallback((sec: number) => {
    const clamped = Math.max(rangeStart, Math.min(rangeEnd, sec))
    return ((clamped - rangeStart) / rangeLen) * STRIP_WIDTH
  }, [rangeStart, rangeEnd, rangeLen])

  const xToSec = useCallback((x: number) => {
    const ratio = Math.max(0, Math.min(1, x / STRIP_WIDTH))
    return rangeStart + ratio * rangeLen
  }, [rangeStart, rangeLen])

  const handleDrag = useCallback((clientX: number) => {
    const rect = stripRef.current!.getBoundingClientRect()
    const x = clientX - rect.left
    // clamp so the handle can never be dragged past the pin itself
    const sec = Math.min(timestampSec, xToSec(x))
    setLocalValue(sec)
    onScrub(sec)
  }, [xToSec, timestampSec, onScrub])

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: PointerEvent) => handleDrag(e.clientX)
    const onUp = (e: PointerEvent) => { handleDrag(e.clientX); setDragging(false); onChange(localValue) }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
    return () => window.removeEventListener('pointermove', onMove)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging, handleDrag])

  const pinX = secToX(timestampSec)
  const handleX = secToX(localValue)
  const leadSec = Math.max(0, timestampSec - localValue)

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ fontSize: 11, color: '#8A8F9E', marginBottom: 4 }}>
        Scene starts{' '}
        <strong style={{ color: '#0f2972' }}>
          {leadSec < 0.5 ? 'right at this moment' : `${leadSec.toFixed(0)}s earlier`}
        </strong>
      </div>
      <div
        ref={stripRef}
        onPointerDown={e => { setDragging(true); handleDrag(e.clientX) }}
        style={{
          position: 'relative', width: STRIP_WIDTH, height: 28,
          background: '#F0F1F5', borderRadius: 6, cursor: 'grab', touchAction: 'none',
        }}
      >
        {/* filled range from handle to pin */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: Math.min(handleX, pinX), width: Math.abs(pinX - handleX),
          background: 'rgba(15,41,114,0.12)', borderRadius: 4,
        }} />

        {/* fixed pin — the annotation's frame, never moves */}
        <div style={{
          position: 'absolute', top: 2, bottom: 2, left: pinX - 1.5, width: 3,
          background: '#4ade80', borderRadius: 2,
        }} />
        <div style={{
          position: 'absolute', top: -14, left: pinX, transform: 'translateX(-50%)',
          fontSize: 9, color: '#166534', fontWeight: 700, whiteSpace: 'nowrap',
        }}>
          {fmt(timestampSec)}
        </div>

        {/* draggable scene-start handle */}
        <div style={{
          position: 'absolute', top: -3, left: handleX - 8, width: 16, height: 34,
          background: '#fff', border: '2px solid #0f2972', borderRadius: 5,
          cursor: dragging ? 'grabbing' : 'grab', boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
        }} />
      </div>
    </div>
  )
}
