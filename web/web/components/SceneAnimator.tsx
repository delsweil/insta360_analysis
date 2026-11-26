'use client'

// Plays the sketches of a form as an animation: scene 1 is shown, everything moves into scene 2, and so on.
// What moves where comes from the sketches; see lib/anim.ts.

import { useEffect, useMemo, useRef, useState } from 'react'
import SceneCanvas from './SceneCanvas'
import { frameAt, plan, formatTime, type Step } from '@/lib/anim'
import { canExportMp4, exportAnimation, type ExportFormat } from '@/lib/animExport'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'
const SPEEDS = [0.5, 1, 2]

export default function SceneAnimator({ steps }: { steps: Step[] }) {
  const pl = useMemo(() => plan(steps), [steps])
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loop, setLoop] = useState(true)
  const [full, setFull] = useState(false)
  const [busy, setBusy] = useState<{ format: ExportFormat; p: number } | null>(null)
  const [exportErr, setExportErr] = useState('')
  const cancel = useRef({ cancelled: false })
  const raf = useRef<number | null>(null)
  const last = useRef(0)
  const total = pl.total

  useEffect(() => {
    if (!playing) return
    last.current = performance.now()
    const tick = (now: number) => {
      const dt = ((now - last.current) / 1000) * speed
      last.current = now
      setT(prev => {
        const next = prev + dt
        if (next < total) return next
        if (loop) return next % Math.max(total, 0.001)
        setPlaying(false)
        return total
      })
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => { if (raf.current !== null) cancelAnimationFrame(raf.current) }
  }, [playing, speed, loop, total])

  useEffect(() => {
    if (!full) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setFull(false); if (e.key === ' ') { e.preventDefault(); setPlaying(p => !p) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [full])

  async function runExport(format: ExportFormat) {
    setPlaying(false); setExportErr(''); cancel.current = { cancelled: false }
    setBusy({ format, p: 0 })
    try {
      const blob = await exportAnimation(steps, format, p => setBusy({ format, p }), cancel.current)
      const file = new File([blob], `animation.${format}`, { type: blob.type })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(file); a.download = file.name; document.body.appendChild(a); a.click(); a.remove()
      setTimeout(() => URL.revokeObjectURL(a.href), 10000)
    } catch (e) {
      if (!cancel.current.cancelled) setExportErr(`Export nicht möglich: ${e instanceof Error ? e.message : 'unbekannter Fehler'}`)
    } finally { setBusy(null) }
  }

  if (steps.length === 0) return null
  const frame = frameAt(steps, t)
  const atEnd = t >= pl.total - 1e-6

  const btn = (active = false): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '7px 12px', minHeight: 38, minWidth: 40, borderRadius: 9, cursor: 'pointer', whiteSpace: 'nowrap',
    border: `1.5px solid ${active ? NAVY : '#E4E6EE'}`, background: active ? '#e8edf8' : '#fff', color: active ? NAVY : '#4A4F5C', fontFamily: 'DM Sans, sans-serif',
  })

  const wrap: React.CSSProperties = full
    ? { position: 'fixed', inset: 0, zIndex: 70, background: '#F8F8F6', display: 'flex', flexDirection: 'column', padding: 12, overflow: 'auto' }
    : { display: 'flex', flexDirection: 'column' }

  return (
    <div data-testid="animator" style={wrap}>
      <div style={{ border: '1px solid #E4E6EE', borderRadius: 12, overflow: 'hidden', background: '#fbfbf8', display: 'flex', justifyContent: 'center', flex: full ? 1 : undefined, minHeight: 0 }}>
        <SceneCanvas scene={frame.scene} testId="anim-canvas" style={{ maxHeight: full ? 'calc(100dvh - 190px)' : 'max(320px, calc(100dvh - 420px))' }} />
      </div>
      <div data-testid="anim-caption" style={{ minHeight: 22, marginTop: 6, fontSize: 13, color: '#33384a', fontWeight: 500 }}>
        {steps.length > 1 ? <b style={{ color: NAVY }}>{frame.index + 1} / {steps.length}  </b> : null}{frame.scene.caption}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
        <button data-testid="anim-play" aria-label={playing ? 'Pause' : 'Abspielen'} onClick={() => { if (!playing && atEnd) setT(0); setPlaying(p => !p) }} style={{ ...btn(true), minWidth: 52 }}>{playing ? '❚❚' : '▶'}</button>
        <button data-testid="anim-restart" aria-label="Von vorn" onClick={() => { setT(0) }} style={btn()}>⏮</button>
        <input
          data-testid="anim-scrub" aria-label="Zeitleiste" type="range" min={0} max={Math.max(pl.total, 0.01)} step={0.02} value={Math.min(t, pl.total)}
          onChange={e => { setPlaying(false); setT(parseFloat(e.target.value)) }}
          style={{ flex: 1, minWidth: 140, accentColor: NAVY }}
        />
        <span data-testid="anim-time" style={{ fontSize: 12, color: MUTED, minWidth: 70, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{formatTime(t)} / {formatTime(pl.total)}</span>
        {SPEEDS.map(sp => <button key={sp} data-testid={`anim-speed-${sp}`} onClick={() => setSpeed(sp)} style={btn(speed === sp)}>{sp === 0.5 ? '½×' : `${sp}×`}</button>)}
        <button data-testid="anim-loop" onClick={() => setLoop(!loop)} style={btn(loop)} title="Wiederholen">↻</button>
        <button data-testid="anim-full" onClick={() => setFull(!full)} style={btn(full)}>{full ? 'Schließen' : 'Vollbild'}</button>
      </div>
      <div data-testid="anim-export" style={{ display: 'flex', gap: 6, marginTop: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        {busy ? (
          <>
            <span data-testid="export-progress" style={{ fontSize: 12, color: NAVY, fontWeight: 600 }}>{busy.format.toUpperCase()} wird erstellt … {Math.round(busy.p * 100)} %</span>
            <button data-testid="export-cancel" onClick={() => { cancel.current.cancelled = true }} style={btn()}>Abbrechen</button>
          </>
        ) : (
          <>
            <span style={{ fontSize: 12, color: MUTED }}>Speichern als</span>
            {canExportMp4() && <button data-testid="export-mp4" onClick={() => runExport('mp4')} style={btn()} title="Video für WhatsApp, Vereinsheim-Bildschirm usw.">Video (MP4)</button>}
            <button data-testid="export-gif" onClick={() => runExport('gif')} style={btn()} title="Animiertes Bild, funktioniert überall">GIF</button>
          </>
        )}
        {exportErr && <span data-testid="export-error" style={{ fontSize: 12, color: '#991b1b' }}>{exportErr}</span>}
      </div>
      {steps.length > 1 && (
        <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
          {steps.map((_, i) => (
            <button key={i} data-testid="anim-jump" onClick={() => { setPlaying(false); setT(pl.starts[i]) }} style={btn(frame.index === i)} aria-label={`Zu Skizze ${i + 1}`}>{i + 1}</button>
          ))}
        </div>
      )}
    </div>
  )
}
