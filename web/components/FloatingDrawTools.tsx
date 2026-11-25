'use client'

// components/FloatingDrawTools.tsx
// Drawing tools over the video itself (desktop): a narrow column at the left edge, one icon per tool, with the number key
// that selects it. Undo / clear / done sit underneath. Style and variants stay in ShapeSettingsPanel.

import type { Tool } from './AnnotationCanvas'

interface Props {
  tool: Tool
  onChange: (t: Tool) => void
  onUndo: () => void
  onClear: () => void
  onDone: () => void
  canUndo: boolean
  shapeCount: number
  penColor: string
  onPenColor: (c: string) => void
}

export const DRAW_TOOLS: { key: Tool; label: string; icon: string; hint: string }[] = [
  { key: 'select', label: 'Select / move', icon: '↖', hint: 'Click a shape to select it, drag to move. Arrow keys nudge, Delete removes, Ctrl/⌘+D duplicates.' },
  { key: 'zone', label: 'Zone', icon: '▭', hint: 'Drag out a rectangle — or click corners and double-click to close a polygon.' },
  { key: 'curve', label: 'Line', icon: '↝', hint: 'Drag from start to end. Pick the pass / dribble style after selecting it.' },
  { key: 'connector', label: 'Connect', icon: '⌇', hint: 'Click one player marker, then another, to link them.' },
  { key: 'cone', label: 'Body shape', icon: '◣', hint: 'Drag from the player in the direction they face; the length is how far they see.' },
  { key: 'highlight', label: 'Player marker', icon: '◎', hint: 'Click a player\'s feet to mark them. Drag instead of clicking to size it; the mouse wheel resizes. Sizes you set are remembered.' },
  { key: 'label', label: 'Label', icon: 'T', hint: 'Click where the text should go.' },
  { key: 'number', label: 'Number', icon: '①', hint: 'Click to drop numbered markers 1, 2, 3 … in order.' },
]

export const PEN_COLORS = ['#facc15', '#ffffff', '#4ade80', '#38bdf8', '#f472b6', '#ef4444']

const small: React.CSSProperties = {
  width: '100%', height: 28, borderRadius: 8, fontSize: 11, fontWeight: 700, cursor: 'pointer',
  fontFamily: 'DM Sans, sans-serif', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.08)',
  color: 'rgba(255,255,255,0.85)',
}

export default function FloatingDrawTools({ tool, onChange, onUndo, onClear, onDone, canUndo, shapeCount, penColor, onPenColor }: Props) {
  return (
    <div
      onPointerDown={e => e.stopPropagation()}
      onClick={e => e.stopPropagation()}
      style={{
        position: 'absolute', left: 10, top: 10, zIndex: 30, display: 'flex', flexDirection: 'column', gap: 4,
        padding: 6, borderRadius: 12, background: 'rgba(9,29,82,0.78)', backdropFilter: 'blur(6px)',
        border: '1px solid rgba(255,255,255,0.18)',
      }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
      {DRAW_TOOLS.map((t, i) => {
        const on = tool === t.key
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            title={`${t.label} (${i + 1})`}
            style={{
              position: 'relative', width: 44, height: 36, borderRadius: 9, cursor: 'pointer', fontSize: 17, lineHeight: 1,
              border: `1.5px solid ${on ? '#4ade80' : 'rgba(255,255,255,0.2)'}`,
              background: on ? 'rgba(74,222,128,0.2)' : 'rgba(255,255,255,0.06)',
              color: on ? '#4ade80' : 'rgba(255,255,255,0.85)',
            }}
          >
            {t.icon}
            <span style={{ position: 'absolute', right: 3, bottom: 1, fontSize: 8, opacity: 0.6, fontFamily: 'DM Sans, sans-serif' }}>{i + 1}</span>
          </button>
        )
      })}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 4, padding: '2px 2px' }} title="Colour for new shapes — and for the selected one">
        {PEN_COLORS.map(c => (
          <button
            key={c} onClick={() => onPenColor(c)} aria-label={`Colour ${c}`}
            style={{
              height: 22, borderRadius: 6, background: c, cursor: 'pointer', padding: 0,
              border: `2px solid ${penColor === c ? '#fff' : 'rgba(255,255,255,0.25)'}`,
              boxShadow: penColor === c ? '0 0 0 1.5px #0f2972' : 'none',
            }}
          />
        ))}
      </div>
      <div style={{ height: 1, background: 'rgba(255,255,255,0.18)', margin: '2px 0' }} />
      <button onClick={onUndo} disabled={!canUndo} title="Undo (Ctrl/⌘+Z)" style={{ ...small, opacity: canUndo ? 1 : 0.4, cursor: canUndo ? 'pointer' : 'default' }}>↩ Undo</button>
      <button onClick={onClear} disabled={shapeCount === 0} title="Remove everything" style={{ ...small, color: '#fca5a5', borderColor: 'rgba(239,68,68,0.4)', opacity: shapeCount ? 1 : 0.4 }}>✕ Clear</button>
      <button onClick={onDone} title="Finish drawing" style={{ ...small, background: '#0f2972', borderColor: '#0f2972', color: '#fff' }}>✓ Done</button>
    </div>
  )
}

/** One line under the toolbar saying what the active tool does with the mouse. Rendered by the parent over the video. */
export function DrawHint({ tool }: { tool: Tool }) {
  const hint = DRAW_TOOLS.find(t => t.key === tool)?.hint
  if (!hint) return null
  return (
    <div style={{
      position: 'absolute', left: 74, bottom: 10, maxWidth: 'min(520px, calc(100% - 90px))', zIndex: 29, pointerEvents: 'none',
      fontFamily: 'DM Sans, sans-serif', fontSize: 12, lineHeight: 1.35, color: '#fff',
      background: 'rgba(9,29,82,0.82)', padding: '6px 10px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.18)',
    }}>{hint}</div>
  )
}
