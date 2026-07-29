'use client'

// components/ShapeSettingsPanel.tsx
// Inline "Object settings" style panel — appears when a placed shape is
// selected in AnnotationCanvas, letting the coach adjust color, opacity,
// dash style, and (for highlights) the player name tag.

import type { Shape, DashStyle } from './AnnotationCanvas'

interface Props {
  shape: Shape
  onChange: (patch: Partial<Shape>) => void
  onRemove: () => void
  onClose: () => void
}

const DASH_OPTIONS: { key: DashStyle; label: string }[] = [
  { key: 'solid',  label: '───' },
  { key: 'dashed', label: '╌╌╌' },
  { key: 'dotted', label: '····' },
]

export default function ShapeSettingsPanel({ shape, onChange, onRemove, onClose }: Props) {
  const hasStyle = shape.type === 'zone' || shape.type === 'arrow' || shape.type === 'highlight'

  return (
    <div style={{
      position: 'absolute', top: 12, right: 12, width: 200,
      background: '#1a1d2b', borderRadius: 10,
      border: '1px solid rgba(255,255,255,0.12)',
      boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
      padding: '10px 12px',
      fontFamily: 'DM Sans, sans-serif',
      color: '#fff',
      zIndex: 5,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', opacity: 0.6 }}>
          {shape.type === 'highlight' ? 'Player highlight' : shape.type} settings
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#8A8F9E', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>✕</button>
      </div>

      {shape.type === 'highlight' && (
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>Player name (optional)</label>
          <input
            type="text"
            value={shape.playerName ?? ''}
            onChange={e => onChange({ playerName: e.target.value || undefined })}
            placeholder="e.g. Müller"
            style={{
              width: '100%', fontSize: 12, padding: '5px 8px',
              borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)',
              background: '#111318', color: '#fff', outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>
      )}

      {shape.type === 'label' && (
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>Text</label>
          <input
            type="text"
            value={shape.text}
            onChange={e => onChange({ text: e.target.value })}
            style={{
              width: '100%', fontSize: 12, padding: '5px 8px',
              borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)',
              background: '#111318', color: '#fff', outline: 'none',
              boxSizing: 'border-box',
            }}
          />
        </div>
      )}

      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>Color</label>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="color"
            value={shape.type === 'label' ? shape.color : (shape as any).color}
            onChange={e => onChange({ color: e.target.value } as Partial<Shape>)}
            style={{ width: 30, height: 26, border: 'none', borderRadius: 4, background: 'none', cursor: 'pointer' }}
          />
          <span style={{ fontSize: 11, opacity: 0.7, fontFamily: 'monospace' }}>
            {(shape.type === 'label' ? shape.color : (shape as any).color)?.toUpperCase()}
          </span>
        </div>
      </div>

      {hasStyle && 'opacity' in shape && (
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>
            Opacity — {Math.round((shape as any).opacity * 100)}%
          </label>
          <input
            type="range" min={0.1} max={1} step={0.05}
            value={(shape as any).opacity}
            onChange={e => onChange({ opacity: parseFloat(e.target.value) } as Partial<Shape>)}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {hasStyle && 'dash' in shape && (
        <div style={{ marginBottom: 10 }}>
          <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>Line style</label>
          <div style={{ display: 'flex', gap: 4 }}>
            {DASH_OPTIONS.map(opt => (
              <button
                key={opt.key}
                onClick={() => onChange({ dash: opt.key } as Partial<Shape>)}
                style={{
                  flex: 1, padding: '5px 0', fontSize: 12,
                  borderRadius: 5,
                  border: `1.5px solid ${(shape as any).dash === opt.key ? '#4ade80' : 'rgba(255,255,255,0.15)'}`,
                  background: (shape as any).dash === opt.key ? 'rgba(74,222,128,0.15)' : 'transparent',
                  color: '#fff', cursor: 'pointer',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={onRemove}
        style={{
          width: '100%', fontSize: 12, fontWeight: 600,
          padding: '7px 0', borderRadius: 6, border: 'none',
          background: '#b91c1c', color: '#fff', cursor: 'pointer',
        }}
      >
        Remove
      </button>
    </div>
  )
}
