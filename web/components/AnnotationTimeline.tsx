'use client'

// components/AnnotationTimeline.tsx
// The coach's timeline for a drawing: one row per shape, a playhead, and a freeze control.
//
//   • drag on the ruler (or empty track) to move the playhead — the video shows the matching frame
//   • with Auto-key on, move a shape while the playhead is later than its start and it gets a keyframe (◆) there
//   • drag a bar's ends to set when the shape appears / disappears; drag the bar to slide all of its timing
//   • drag a ◆ to retime it; Alt-click a ◆ to delete it
//   • the blue zone is the freeze: the video holds on the anchor frame while the drawing plays; drag its edge to set the length
import { useEffect, useReducer, useRef, useSyncExternalStore } from 'react'
import type { Shape } from './AnnotationCanvas'
import { EditorClock, type Hold } from '@/lib/annoPlayback'
import { setWindow, shiftShape, moveKeyframe, removeKeyframe, keyframeAt, baseTime, type Animated } from '@/lib/annoAnim'

interface Props {
  shapes: Shape[]
  selectedId: string | null
  onSelect: (id: string | null) => void
  onChangeShape: (id: string, next: Shape) => void
  onReplaceAll: (shapes: Shape[]) => void
  clock: EditorClock
  total: number          // where the drawing ends
  range: number          // the length shown (≥ total): room to drag the playhead past the end
  F: number
  hold: Hold
  onHold: (h: Hold) => void
  autoKey: boolean
  onAutoKey: (v: boolean) => void
  onToggleKeyframe: (id: string) => void
  onReveal: () => void
  onShowAll: () => void
  animated: boolean
  onSave?: () => void
  saving?: boolean
  saveLabel?: string
}

const LABEL_W = 132
const ROW_H = 24
const NAVY = '#0f2972'

function rowLabel(s: Shape): { icon: string; text: string } {
  switch (s.type) {
    case 'highlight': return { icon: '◎', text: s.playerName || 'Player' }
    case 'curve': return { icon: '↝', text: s.style === 'dribble' ? 'Dribble' : 'Line' }
    case 'zone': return { icon: '▭', text: 'Zone' }
    case 'cone': return { icon: '◣', text: 'Body shape' }
    case 'label': return { icon: 'T', text: s.text }
    case 'number': return { icon: '①', text: `Number ${s.value}` }
    case 'connector': return { icon: '⌇', text: 'Link' }
  }
}
const colorOf = (s: Shape) => ('color' in s ? s.color : '#ffffff')
const snap = (x: number, free: boolean) => (free ? Math.round(x * 100) / 100 : Math.round(x * 20) / 20)
const fmt = (x: number) => `${x.toFixed(1)}s`

function drag(onMove: (e: PointerEvent) => void, onEnd?: () => void) {
  const move = (e: PointerEvent) => onMove(e)
  const up = (e: PointerEvent) => { onMove(e); window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); onEnd?.() }
  window.addEventListener('pointermove', move)
  window.addEventListener('pointerup', up)
}

export default function AnnotationTimeline(p: Props) {
  const { shapes, selectedId, clock, total, range, F, hold } = p
  useSyncExternalStore(clock.subscribe, clock.getVersion, clock.getVersion)
  const [, frame] = useReducer((n: number) => n + 1, 0)
  const playing = clock.isPlaying()
  useEffect(() => {
    if (!playing) return
    let raf = 0
    const loop = () => { frame(); raf = requestAnimationFrame(loop) }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  const tau = clock.get()
  const trackRef = useRef<HTMLDivElement>(null)
  const pct = (t: number) => `${Math.max(0, Math.min(1, t / range)) * 100}%`
  const tAt = (clientX: number, free = false) => {
    const r = trackRef.current!.getBoundingClientRect()
    return snap(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * range, free)
  }
  const secPerPx = () => range / trackRef.current!.getBoundingClientRect().width

  const step = range <= 8 ? 1 : range <= 16 ? 2 : 5
  const ticks: number[] = []
  for (let t = 0; t <= range + 1e-6; t += step) ticks.push(t)

  const scrubFrom = (e: React.PointerEvent) => {
    e.preventDefault()
    clock.scrub(tAt(e.clientX, e.shiftKey))
    drag(ev => clock.scrub(tAt(ev.clientX, ev.shiftKey)))
  }

  const selected = shapes.find(s => s.id === selectedId) ?? null
  const onKey = selected ? !!keyframeAt(selected, tau) : false
  const canKey = !!selected && tau > baseTime(selected) + 0.05

  const btn = (on = false): React.CSSProperties => ({
    fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 7, cursor: 'pointer', whiteSpace: 'nowrap',
    border: `1.5px solid ${on ? NAVY : '#E4E6EE'}`, background: on ? NAVY : '#fff', color: on ? '#fff' : '#4b5163',
  })

  return (
    <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '10px 12px', fontFamily: 'DM Sans, sans-serif', userSelect: 'none' }}>
      {/* controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        <button onClick={() => clock.toggle()} style={{ ...btn(true), minWidth: 74 }} title="Preview the drawing (Space)">
          {playing ? '❚❚ Pause' : '▶ Preview'}
        </button>
        <button onClick={() => clock.scrub(0)} style={btn()} title="Back to the start">⏮</button>
        <span style={{ fontSize: 12, fontWeight: 700, color: NAVY, fontVariantNumeric: 'tabular-nums', minWidth: 92 }}>
          {fmt(tau)} <span style={{ fontWeight: 400, color: '#8A8F9E' }}>/ {fmt(total)}</span>
        </span>

        <span style={{ width: 1, height: 18, background: '#E4E6EE' }} />

        <span style={{ fontSize: 11, color: '#8A8F9E' }}>Video</span>
        <div style={{ display: 'flex', gap: 0 }}>
          <button
            onClick={() => p.onHold({ mode: 'freeze', sec: hold.mode === 'freeze' ? hold.sec : 0 })}
            style={{ ...btn(hold.mode === 'freeze'), borderRadius: '7px 0 0 7px' }}
            title="The video stops on this frame while the drawing plays"
          >Freezes</button>
          <button
            onClick={() => p.onHold({ mode: 'live' })}
            style={{ ...btn(hold.mode === 'live'), borderRadius: '0 7px 7px 0', marginLeft: -1.5 }}
            title="The drawing animates over the playing video, e.g. a marker following a runner"
          >Keeps playing</button>
        </div>
        {hold.mode === 'freeze' && (
          <label style={{ fontSize: 11, color: '#8A8F9E', display: 'flex', alignItems: 'center', gap: 4 }}>
            for
            <input
              type="number" min={0} max={60} step={1} value={hold.sec}
              onChange={e => p.onHold({ mode: 'freeze', sec: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
              style={{ width: 52, fontSize: 11, padding: '3px 5px', border: '1px solid #E4E6EE', borderRadius: 5, outline: 'none' }}
            />
            s <span style={{ opacity: 0.7 }}>{hold.sec === 0 ? '(until play is pressed)' : '(then plays on)'}</span>
          </label>
        )}

        <span style={{ flex: 1 }} />

        <label style={{ fontSize: 11, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer', color: '#4b5163' }}
          title="Move a shape while the playhead is later than its start and a keyframe is created automatically">
          <input type="checkbox" checked={p.autoKey} onChange={e => p.onAutoKey(e.target.checked)} /> Auto-key
        </label>
        <button
          disabled={!selected || (!canKey && !onKey)}
          onClick={() => selected && p.onToggleKeyframe(selected.id)}
          style={{ ...btn(onKey), opacity: selected && (canKey || onKey) ? 1 : 0.4 }}
          title="Add or remove a keyframe for the selected shape at the playhead"
        >◆ {onKey ? 'Remove keyframe' : 'Keyframe here'}</button>
        <button onClick={p.onReveal} disabled={shapes.length < 2} style={{ ...btn(), opacity: shapes.length < 2 ? 0.4 : 1 }}
          title="Shapes appear one after another in the order you drew them">One by one</button>
        {p.animated && <button onClick={p.onShowAll} style={btn()} title="Show everything from the start, without timing">All at once</button>}
        {p.onSave && (
          <button onClick={p.onSave} disabled={p.saving} style={{ ...btn(true), background: '#16a34a', borderColor: '#16a34a', minWidth: 110, opacity: p.saving ? 0.6 : 1 }}>
            {p.saving ? 'Saving…' : `✓ ${p.saveLabel ?? 'Save'}`}
          </button>
        )}
      </div>

      {/* tracks */}
      <div style={{ position: 'relative' }}>
        <div style={{ display: 'grid', gridTemplateColumns: `${LABEL_W}px 1fr` }}>
          {/* ruler */}
          <div />
          <div ref={trackRef} onPointerDown={scrubFrom} style={{ position: 'relative', height: 22, cursor: 'ew-resize', borderBottom: '1px solid #E4E6EE' }}>
            {F > 0 && (
              <div title="Freeze: the video holds on this frame while the drawing plays" style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: pct(F), background: 'rgba(15,41,114,0.10)', borderRight: `2px solid ${NAVY}` }}>
                <span style={{ position: 'absolute', left: 4, top: 4, fontSize: 9, fontWeight: 700, color: NAVY, letterSpacing: '0.04em' }}>FROZEN</span>
              </div>
            )}
            {hold.mode === 'freeze' && F > 0 && (
              <div
                title="Drag to change the freeze length"
                onPointerDown={e => {
                  e.stopPropagation(); e.preventDefault()
                  drag(ev => p.onHold({ mode: 'freeze', sec: Math.max(1, Math.round(tAt(ev.clientX, true))) }))
                }}
                style={{ position: 'absolute', top: 0, bottom: 0, left: `calc(${pct(F)} - 5px)`, width: 10, cursor: 'col-resize', zIndex: 3 }}
              />
            )}
            {total < range - 0.05 && (
              <div title="After the drawing has ended. Moving a shape here extends the drawing." style={{ position: 'absolute', left: pct(total), right: 0, top: 0, bottom: 0, background: 'repeating-linear-gradient(135deg, rgba(0,0,0,0.05) 0 6px, transparent 6px 12px)', borderLeft: '2px dashed #8A8F9E', pointerEvents: 'none' }}>
                <span style={{ position: 'absolute', left: 4, top: 4, fontSize: 9, fontWeight: 700, color: '#8A8F9E', letterSpacing: '0.04em' }}>ENDS</span>
              </div>
            )}
            {ticks.map(t => (
              <div key={t} style={{ position: 'absolute', left: pct(t), top: 0, bottom: 0, borderLeft: '1px solid #E4E6EE', paddingLeft: 3, fontSize: 9, color: '#8A8F9E', lineHeight: '22px', pointerEvents: 'none' }}>
                {t === 0 ? '' : `${t}s`}
              </div>
            ))}
          </div>

          {/* shape rows */}
          {shapes.length === 0 && (
            <>
              <div />
              <div style={{ fontSize: 11, color: '#8A8F9E', padding: '10px 0' }}>Draw something on the video: each shape gets a row here.</div>
            </>
          )}
          {shapes.map(s => {
            const an = s as unknown as Animated
            const t0 = an.t0 ?? 0
            const t1 = Math.min(an.t1 ?? total, range)
            const sel = s.id === selectedId
            const { icon, text } = rowLabel(s)
            const col = colorOf(s)
            return (
              <RowFragment key={s.id}>
                <div
                  onClick={() => p.onSelect(sel ? null : s.id)}
                  title={text}
                  style={{
                    height: ROW_H, display: 'flex', alignItems: 'center', gap: 6, padding: '0 6px', cursor: 'pointer', fontSize: 11,
                    fontWeight: sel ? 700 : 500, color: sel ? NAVY : '#4b5163', background: sel ? 'rgba(15,41,114,0.07)' : 'transparent',
                    overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis', borderRadius: 4,
                  }}
                >
                  <span style={{ color: col, fontSize: 13, width: 14, textAlign: 'center', textShadow: '0 0 1px #0006' }}>{icon}</span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{text}</span>
                </div>
                <div
                  style={{ position: 'relative', height: ROW_H, background: sel ? 'rgba(15,41,114,0.04)' : 'transparent' }}
                  onPointerDown={e => { if (e.target === e.currentTarget) scrubFrom(e) }}
                >
                  {/* when it is on screen */}
                  <div
                    onPointerDown={e => {
                      e.stopPropagation(); e.preventDefault()
                      p.onSelect(s.id)
                      const x0 = e.clientX, k = secPerPx(); let last = 0
                      drag(ev => {
                        const d = snap((ev.clientX - x0) * k, ev.shiftKey)
                        if (d !== last) { p.onChangeShape(s.id, shiftShape(s, d)); last = d }
                      })
                    }}
                    title="Drag to slide this shape's timing"
                    style={{
                      position: 'absolute', top: 7, height: 10, left: pct(t0), width: `calc(${pct(t1)} - ${pct(t0)})`,
                      background: col, opacity: sel ? 0.85 : 0.5, borderRadius: 5, cursor: 'grab', boxShadow: '0 0 0 1px rgba(0,0,0,0.25)',
                    }}
                  />
                  {sel && (
                    <>
                      <Grip left={pct(t0)} title="Appears" onDown={e => {
                        e.stopPropagation(); e.preventDefault()
                        drag(ev => p.onChangeShape(s.id, setWindow(s, tAt(ev.clientX, ev.shiftKey), undefined)))
                      }} />
                      <Grip left={pct(t1)} title="Disappears (drag to the end for 'stays')" onDown={e => {
                        e.stopPropagation(); e.preventDefault()
                        drag(ev => {
                          const t = tAt(ev.clientX, ev.shiftKey)
                          p.onChangeShape(s.id, setWindow(s, undefined, t >= total - 0.05 ? null : t))
                        })
                      }} />
                    </>
                  )}
                  {/* keyframes */}
                  {(an.kf ?? []).map(k => (
                    <div
                      key={k.t}
                      title={`Keyframe at ${fmt(k.t)} — drag to retime, Alt-click to delete`}
                      onPointerDown={e => {
                        e.stopPropagation(); e.preventDefault()
                        p.onSelect(s.id)
                        if (e.altKey) { p.onChangeShape(s.id, removeKeyframe(s, k.t)); return }
                        clock.scrub(k.t)
                        let cur = s; let from = k.t
                        drag(ev => {
                          const to = tAt(ev.clientX, ev.shiftKey)
                          if (Math.abs(to - from) < 1e-6) return
                          cur = moveKeyframe(cur, from, to); from = Math.max(baseTime(cur) + 0.1, to)
                          p.onChangeShape(s.id, cur)
                        })
                      }}
                      style={{
                        position: 'absolute', top: 5, left: `calc(${pct(k.t)} - 7px)`, width: 14, height: 14, cursor: 'ew-resize', zIndex: 2,
                        transform: 'rotate(45deg) scale(0.72)', background: Math.abs(k.t - tau) < 0.06 ? '#facc15' : '#fff',
                        border: `2px solid ${NAVY}`, borderRadius: 2,
                      }}
                    />
                  ))}
                </div>
              </RowFragment>
            )
          })}
        </div>

        {/* playhead */}
        <div style={{ position: 'absolute', top: 0, bottom: 0, left: LABEL_W, right: 0, pointerEvents: 'none' }}>
          <div style={{ position: 'absolute', top: 0, bottom: 0, left: pct(tau), width: 0, borderLeft: '2px solid #E8780A' }}>
            <div style={{ position: 'absolute', top: -2, left: -6, width: 10, height: 10, background: '#E8780A', borderRadius: '50% 50% 50% 0', transform: 'rotate(-45deg)' }} />
          </div>
        </div>
      </div>
    </div>
  )
}

function RowFragment({ children }: { children: React.ReactNode }) { return <>{children}</> }

function Grip({ left, title, onDown }: { left: string; title: string; onDown: (e: React.PointerEvent) => void }) {
  return (
    <div
      title={title} onPointerDown={onDown}
      style={{ position: 'absolute', top: 4, height: 16, width: 8, left: `calc(${left} - 4px)`, background: '#fff', border: `2px solid ${NAVY}`, borderRadius: 3, cursor: 'col-resize', zIndex: 3 }}
    />
  )
}
