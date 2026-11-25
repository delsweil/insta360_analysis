'use client'

// components/MobileDrawLayer.tsx
// The coach's drawing controls on the phone HUD (landscape, over the video):
//
//   left       a narrow tool rail (two columns) with the pen colours — thumb reach, never over the middle of the picture
//   top right  Undo · Timeline show/hide · Save · Cancel
//   under it   what you can do with the selected shape: Duplicate · Delete · Keyframe
//   bottom     the timeline sheet (children), hidden until asked for so it never takes picture space by default
//
// Fingers: tap a shape to select, drag to move, pinch with two fingers to resize (the canvas does that).
import type { Tool } from './AnnotationCanvas'
import { DRAW_TOOLS, PEN_COLORS } from './FloatingDrawTools'

interface Props {
  tool: Tool
  onTool: (t: Tool) => void
  penColor: string
  onPenColor: (c: string) => void
  canUndo: boolean
  onUndo: () => void
  shapeCount: number
  onCancel: () => void
  onSave: () => void
  saving?: boolean
  saveLabel: string
  hasSelection: boolean
  onDuplicate: () => void
  onDelete: () => void
  onKeyframe: () => void
  timelineOpen: boolean
  onToggleTimeline: () => void
  /** The timeline sheet's content; rendered only while the timeline is open. */
  children?: React.ReactNode
}

const HINTS: Record<Tool, string> = {
  select: 'Tippen: auswählen · Ziehen: bewegen · Zwei Finger: Größe ändern',
  zone: 'Ziehen: Rechteck · oder Ecken antippen, Doppeltipp schließt',
  curve: 'Von Start nach Ziel ziehen',
  connector: 'Erst einen Spieler-Marker antippen, dann den zweiten',
  cone: 'Vom Spieler in Blickrichtung ziehen',
  highlight: 'Füße des Spielers antippen · Ziehen stellt die Größe ein',
  label: 'Tippen, wo der Text stehen soll',
  number: 'Tippen setzt die Nummern 1, 2, 3 …',
}

const safeL = 'max(8px, env(safe-area-inset-left))'
const safeR = 'max(8px, env(safe-area-inset-right))'
const safeB = 'max(8px, env(safe-area-inset-bottom))'
const TOP = 'max(50px, calc(env(safe-area-inset-top) + 40px))'

const glass: React.CSSProperties = {
  background: 'rgba(9,29,82,0.82)', backdropFilter: 'blur(6px)', border: '1px solid rgba(255,255,255,0.18)',
}
const pill = (over: React.CSSProperties = {}): React.CSSProperties => ({
  height: 38, minWidth: 38, padding: '0 11px', borderRadius: 10, fontSize: 12, fontWeight: 700, cursor: 'pointer',
  fontFamily: 'DM Sans, sans-serif', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.1)',
  color: 'rgba(255,255,255,0.92)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, ...over,
})

export default function MobileDrawLayer(p: Props) {
  const stop = { onPointerDown: (e: React.PointerEvent) => e.stopPropagation(), onClick: (e: React.MouseEvent) => e.stopPropagation() }
  return (
    <div style={{ position: 'absolute', inset: 0, zIndex: 12, pointerEvents: 'none', fontFamily: 'DM Sans, sans-serif' }}>
      {/* tool rail */}
      <div {...stop} style={{ ...glass, position: 'absolute', left: safeL, top: TOP, padding: 5, borderRadius: 12, display: 'flex', flexDirection: 'column', gap: 4, pointerEvents: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
          {DRAW_TOOLS.map(t => {
            const on = p.tool === t.key
            return (
              <button
                key={t.key} onClick={() => p.onTool(t.key)} aria-label={t.label} aria-pressed={on} data-tool={t.key}
                style={{
                  width: 42, height: 36, borderRadius: 9, fontSize: 17, lineHeight: 1, cursor: 'pointer', padding: 0,
                  border: `1.5px solid ${on ? '#4ade80' : 'rgba(255,255,255,0.2)'}`,
                  background: on ? 'rgba(74,222,128,0.2)' : 'rgba(255,255,255,0.06)',
                  color: on ? '#4ade80' : 'rgba(255,255,255,0.88)',
                }}
              >{t.icon}</button>
            )
          })}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4, padding: '2px 1px' }}>
          {PEN_COLORS.map(c => (
            <button
              key={c} onClick={() => p.onPenColor(c)} aria-label={`Farbe ${c}`}
              style={{
                height: 24, borderRadius: 6, background: c, cursor: 'pointer', padding: 0,
                border: `2px solid ${p.penColor === c ? '#fff' : 'rgba(255,255,255,0.25)'}`,
                boxShadow: p.penColor === c ? '0 0 0 1.5px #0f2972' : 'none',
              }}
            />
          ))}
        </div>
      </div>

      {/* actions */}
      <div style={{ position: 'absolute', right: safeR, top: TOP, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6, pointerEvents: 'none' }}>
        <div {...stop} style={{ display: 'flex', gap: 6, pointerEvents: 'auto' }}>
          <button onClick={p.onUndo} disabled={!p.canUndo} aria-label="Rückgängig" style={pill({ opacity: p.canUndo ? 1 : 0.4 })}>↩</button>
          <button
            onClick={p.onToggleTimeline} aria-pressed={p.timelineOpen} data-testid="timeline-toggle"
            style={pill(p.timelineOpen ? { background: '#E8780A', borderColor: '#E8780A', color: '#fff' } : {})}
          >⏱ Zeitachse {p.timelineOpen ? '▾' : '▴'}</button>
          <button onClick={p.onSave} disabled={p.saving || p.shapeCount === 0} data-testid="mobile-save"
            style={pill({ background: '#16a34a', borderColor: '#16a34a', color: '#fff', opacity: p.saving || p.shapeCount === 0 ? 0.5 : 1 })}
          >{p.saving ? '…' : `✓ ${p.saveLabel}`}</button>
          <button onClick={p.onCancel} aria-label="Abbrechen" style={pill({ color: '#fca5a5', borderColor: 'rgba(239,68,68,0.45)' })}>✕</button>
        </div>
        {p.hasSelection && (
          <div {...stop} style={{ ...glass, display: 'flex', gap: 6, padding: 5, borderRadius: 12, pointerEvents: 'auto' }}>
            <button onClick={p.onDuplicate} style={pill()}>⧉ Kopie</button>
            <button onClick={p.onKeyframe} style={pill()} title="Position oder Größe an dieser Stelle der Zeitachse festhalten">◆ Keyframe</button>
            <button onClick={p.onDelete} style={pill({ color: '#fca5a5', borderColor: 'rgba(239,68,68,0.45)' })} aria-label="Löschen">🗑</button>
          </div>
        )}
      </div>

      {/* hint — only while the timeline sheet is closed */}
      {!p.timelineOpen && !p.hasSelection && (
        <div style={{
          position: 'absolute', left: '50%', transform: 'translateX(-50%)', bottom: safeB, maxWidth: 'min(520px, 70%)',
          ...glass, color: '#fff', fontSize: 12, lineHeight: 1.35, padding: '6px 12px', borderRadius: 9, textAlign: 'center',
        }}>{HINTS[p.tool]}</div>
      )}

      {/* timeline sheet */}
      {p.timelineOpen && (
        <div
          {...stop}
          data-testid="timeline-sheet"
          style={{
            position: 'absolute', left: `calc(${safeL} + 102px)`, right: safeR, bottom: safeB, maxHeight: '52%', overflowY: 'auto',
            borderRadius: 12, pointerEvents: 'auto', boxShadow: '0 6px 24px rgba(0,0,0,0.45)',
          }}
        >{p.children}</div>
      )}
    </div>
  )
}
