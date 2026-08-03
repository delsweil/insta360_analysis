'use client'

// components/TimingStrip.tsx
// Replaces TrimStrip with a fuller timing model: four distinct, explicit
// points instead of tangled implicit logic (a fixed "hold seconds after
// resume" heuristic, a dual-purpose end_timestamp_sec, etc).
//
//   clip start ──────► [ANNOTATION SHOWN] ──────► annotation removed ──────► clip end
//   (draggable,          (fixed pin — where           (draggable,              (draggable,
//    ≤ pin)                shapes are drawn)            ≥ pin)                   ≥ removed)
//
// Dragging any handle live-scrubs the video preview via onScrub, and commits
// via onChange when released — same pattern as the old TrimStrip.

import { useRef, useState, useCallback, useEffect } from 'react'

interface TimingStripProps {
  /** The annotation's fixed frame — where the shapes are drawn, never moves. */
  timestampSec: number
  contextStart: number
  annotationEnd: number
  clipEnd: number
  onChangeContextStart: (sec: number) => void
  onChangeAnnotationEnd: (sec: number) => void
  onChangeClipEnd: (sec: number) => void
  /** Called continuously while dragging any handle, so the video preview can scrub live. */
  onScrub: (sec: number) => void
}

type HandleKey = 'contextStart' | 'annotationEnd' | 'clipEnd'

const STRIP_WIDTH = 320
const LEFT_PAD_SEC = 3   // minimum room shown before the earliest point
const RIGHT_PAD_SEC = 3  // minimum room shown after the latest point

function fmt(sec: number) {
  const s = Math.max(0, Math.round(sec))
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}:${String(r).padStart(2, '0')}`
}

export default function TimingStrip({
  timestampSec, contextStart, annotationEnd, clipEnd,
  onChangeContextStart, onChangeAnnotationEnd, onChangeClipEnd, onScrub,
}: TimingStripProps) {
  const stripRef = useRef<HTMLDivElement>(null)
  const [dragging, setDragging] = useState<HandleKey | null>(null)
  const [local, setLocal] = useState({ contextStart, annotationEnd, clipEnd })

  useEffect(() => {
    if (!dragging) setLocal({ contextStart, annotationEnd, clipEnd })
  }, [contextStart, annotationEnd, clipEnd, dragging])

  const rangeStart = Math.min(contextStart, local.contextStart) - LEFT_PAD_SEC
  const rangeEnd = Math.max(clipEnd, local.clipEnd) + RIGHT_PAD_SEC
  const rangeLen = Math.max(1, rangeEnd - rangeStart)

  const secToX = useCallback((sec: number) => {
    const clamped = Math.max(rangeStart, Math.min(rangeEnd, sec))
    return ((clamped - rangeStart) / rangeLen) * STRIP_WIDTH
  }, [rangeStart, rangeEnd, rangeLen])

  const xToSec = useCallback((x: number) => {
    const ratio = Math.max(0, Math.min(1, x / STRIP_WIDTH))
    return rangeStart + ratio * rangeLen
  }, [rangeStart, rangeLen])

  const handleDrag = useCallback((key: HandleKey, clientX: number) => {
    const rect = stripRef.current!.getBoundingClientRect()
    const raw = xToSec(clientX - rect.left)
    setLocal(prev => {
      let next = { ...prev }
      if (key === 'contextStart') next.contextStart = Math.min(raw, timestampSec)
      if (key === 'annotationEnd') next.annotationEnd = Math.max(raw, timestampSec, prev.contextStart)
      if (key === 'clipEnd') next.clipEnd = Math.max(raw, next.annotationEnd)
      // keep dependent order sane if a middle handle got dragged past a neighbor
      next.annotationEnd = Math.max(next.annotationEnd, timestampSec)
      next.clipEnd = Math.max(next.clipEnd, next.annotationEnd)
      onScrub(raw)
      return next
    })
  }, [xToSec, timestampSec, onScrub])

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: PointerEvent) => handleDrag(dragging, e.clientX)
    const onUp = (e: PointerEvent) => {
      handleDrag(dragging, e.clientX)
      setDragging(null)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
    return () => window.removeEventListener('pointermove', onMove)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging, handleDrag])

  // commit on release
  useEffect(() => {
    if (dragging) return
    if (local.contextStart !== contextStart) onChangeContextStart(local.contextStart)
    if (local.annotationEnd !== annotationEnd) onChangeAnnotationEnd(local.annotationEnd)
    if (local.clipEnd !== clipEnd) onChangeClipEnd(local.clipEnd)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging])

  const pinX = secToX(timestampSec)
  const startX = secToX(local.contextStart)
  const removedX = secToX(local.annotationEnd)
  const endX = secToX(local.clipEnd)

  const Handle = ({ x, color, keyName }: { x: number; color: string; keyName: HandleKey }) => (
    <div
      onPointerDown={e => { e.stopPropagation(); setDragging(keyName) }}
      style={{
        position: 'absolute', top: -3, left: x - 8, width: 16, height: 30,
        background: '#fff', border: `2px solid ${color}`, borderRadius: 5,
        cursor: dragging === keyName ? 'grabbing' : 'grab',
        boxShadow: '0 1px 3px rgba(0,0,0,0.2)', zIndex: 3,
      }}
    />
  )

  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', gap: 10, fontSize: 10, color: '#8A8F9E', marginBottom: 4, flexWrap: 'wrap' }}>
        <span><span style={{ color: '#0f2972', fontWeight: 700 }}>●</span> Start {fmt(local.contextStart)}</span>
        <span><span style={{ color: '#4ade80', fontWeight: 700 }}>●</span> Shown {fmt(timestampSec)}</span>
        <span><span style={{ color: '#f97316', fontWeight: 700 }}>●</span> Removed {fmt(local.annotationEnd)}</span>
        <span><span style={{ color: '#b91c1c', fontWeight: 700 }}>●</span> Clip end {fmt(local.clipEnd)}</span>
      </div>
      <div
        ref={stripRef}
        style={{
          position: 'relative', width: STRIP_WIDTH, height: 26,
          background: '#F0F1F5', borderRadius: 6, touchAction: 'none',
        }}
      >
        {/* shown window: pin → removed */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: pinX, width: Math.max(0, removedX - pinX),
          background: 'rgba(74, 222, 128, 0.18)',
        }} />
        {/* lead-in: start → pin */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: startX, width: Math.max(0, pinX - startX),
          background: 'rgba(15, 41, 114, 0.10)',
        }} />
        {/* tail: removed → clip end */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: removedX, width: Math.max(0, endX - removedX),
          background: 'rgba(185, 28, 28, 0.08)',
        }} />

        {/* fixed pin */}
        <div style={{
          position: 'absolute', top: 1, bottom: 1, left: pinX - 1.5, width: 3,
          background: '#4ade80', borderRadius: 2, zIndex: 2,
        }} />

        <Handle x={startX} color="#0f2972" keyName="contextStart" />
        <Handle x={removedX} color="#f97316" keyName="annotationEnd" />
        <Handle x={endX} color="#b91c1c" keyName="clipEnd" />
      </div>
    </div>
  )
}
