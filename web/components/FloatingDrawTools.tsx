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
}

export const DRAW_TOOLS: { key: Tool; label: string; icon: string }[] = [
  { key: 'select', label: 'Select / move', icon: '↖' },
  { key: 'zone', label: 'Zone', icon: '▭' },
  { key: 'curve', label: 'Line', icon: '↝' },
  { key: 'connector', label: 'Connect', icon: '⌇' },
  { key: 'cone', label: 'Body shape', icon: '◣' },
  { key: 'highlight', label: 'Highlight', icon: '◎' },
  { key: 'label', label: 'Label', icon: 'T' },
  { key: 'number', label: 'Number', icon: '①' },
]

const small: React.CSSProperties = {
  width: '100%', height: 28, borderRadius: 8, fontSize: 11, fontWeight: 700, cursor: 'pointer',
  fontFamily: 'DM Sans, sans-serif', border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.08)',
  color: 'rgba(255,255,255,0.85)',
}

export default function FloatingDrawTools({ tool, onChange, onUndo, onClear, onDone, canUndo, shapeCount }: Props) {
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
      <div style={{ height: 1, background: 'rgba(255,255,255,0.18)', margin: '2px 0' }} />
      <button onClick={onUndo} disabled={!canUndo} title="Undo (Ctrl/⌘+Z)" style={{ ...small, opacity: canUndo ? 1 : 0.4, cursor: canUndo ? 'pointer' : 'default' }}>↩ Undo</button>
      <button onClick={onClear} disabled={shapeCount === 0} title="Remove everything" style={{ ...small, color: '#fca5a5', borderColor: 'rgba(239,68,68,0.4)', opacity: shapeCount ? 1 : 0.4 }}>✕ Clear</button>
      <button onClick={onDone} title="Finish drawing" style={{ ...small, background: '#0f2972', borderColor: '#0f2972', color: '#fff' }}>✓ Done</button>
    </div>
  )
}
