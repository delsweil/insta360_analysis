'use client'

// components/ShapeSettingsPanel.tsx
// Appears when a placed shape is selected. Houses all the "occasional" controls
// (color, opacity, dash, style variants) so the always-visible DrawTools bar
// can stay small — this panel only exists when something's actually selected.

import type { Shape, DashStyle, CurveStyle, HighlightStyle } from './AnnotationCanvas'

interface Props {
  shape: Shape
  onChange: (patch: Partial<Shape>) => void
  onRemove: () => void
  onClose: () => void
}

const DASH_OPTIONS: { key: DashStyle; label: string }[] = [
  { key: 'solid', label: '───' },
  { key: 'dashed', label: '╌╌╌' },
  { key: 'dotted', label: '····' },
]

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <label style={{ fontSize: 10, opacity: 0.6, display: 'block', marginBottom: 3 }}>{label}</label>
      {children}
    </div>
  )
}

export default function ShapeSettingsPanel({ shape, onChange, onRemove, onClose }: Props) {
  const hasStyle = shape.type === 'zone' || shape.type === 'curve' || shape.type === 'highlight' || shape.type === 'cone' || shape.type === 'connector'

  return (
    <div style={{
      position: 'absolute', top: 12, right: 12, width: 200,
      background: '#1a1d2b', borderRadius: 10,
      border: '1px solid rgba(255,255,255,0.12)',
      boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
      padding: '10px 12px', fontFamily: 'DM Sans, sans-serif', color: '#fff', zIndex: 5,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', opacity: 0.6 }}>
          {shape.type} settings
        </span>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#8A8F9E', cursor: 'pointer', fontSize: 14, lineHeight: 1 }}>✕</button>
      </div>

      {shape.type === 'highlight' && (
        <>
          <Field label="Style">
            <div style={{ display: 'flex', gap: 4 }}>
              {(['circle', 'spotlight'] as HighlightStyle[]).map(opt => (
                <button
                  key={opt}
                  onClick={() => onChange({ style: opt } as Partial<Shape>)}
                  style={{
                    flex: 1, padding: '5px 0', fontSize: 11, borderRadius: 5, textTransform: 'capitalize',
                    border: `1.5px solid ${shape.style === opt ? '#4ade80' : 'rgba(255,255,255,0.15)'}`,
                    background: shape.style === opt ? 'rgba(74,222,128,0.15)' : 'transparent',
                    color: '#fff', cursor: 'pointer',
                  }}
                >{opt}</button>
              ))}
            </div>
          </Field>
          <Field label="Player name (optional)">
            <input
              type="text" value={shape.playerName ?? ''}
              onChange={e => onChange({ playerName: e.target.value || undefined })}
              placeholder="e.g. Müller"
              style={{ width: '100%', fontSize: 12, padding: '5px 8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', background: '#111318', color: '#fff', outline: 'none', boxSizing: 'border-box' }}
            />
          </Field>
          <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 8 }}>
            Drag the right handle to resize width, bottom handle for height — makes it elliptical rather than a perfect circle.
          </div>
        </>
      )}

      {shape.type === 'curve' && (
        <Field label="Line type">
          <div style={{ display: 'flex', gap: 4 }}>
            {(['pass', 'dribble'] as CurveStyle[]).map(opt => (
              <button
                key={opt}
                onClick={() => onChange({ style: opt } as Partial<Shape>)}
                style={{
                  flex: 1, padding: '5px 0', fontSize: 11, borderRadius: 5, textTransform: 'capitalize',
                  border: `1.5px solid ${shape.style === opt ? '#4ade80' : 'rgba(255,255,255,0.15)'}`,
                  background: shape.style === opt ? 'rgba(74,222,128,0.15)' : 'transparent',
                  color: '#fff', cursor: 'pointer',
                }}
              >{opt}</button>
            ))}
          </div>
        </Field>
      )}

      {shape.type === 'number' && (
        <Field label="Sequence number">
          <input
            type="number" min={1} value={shape.value}
            onChange={e => onChange({ value: Math.max(1, Number(e.target.value) || 1) })}
            style={{ width: '100%', fontSize: 12, padding: '5px 8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', background: '#111318', color: '#fff', outline: 'none', boxSizing: 'border-box' }}
          />
        </Field>
      )}

      {shape.type === 'label' && (
        <Field label="Text">
          <input
            type="text" value={shape.text}
            onChange={e => onChange({ text: e.target.value })}
            style={{ width: '100%', fontSize: 12, padding: '5px 8px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.15)', background: '#111318', color: '#fff', outline: 'none', boxSizing: 'border-box' }}
          />
        </Field>
      )}

      <Field label="Color">
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <input
            type="color"
            value={shape.type === 'label' || shape.type === 'number' ? shape.color : (shape as any).color}
            onChange={e => onChange({ color: e.target.value } as Partial<Shape>)}
            style={{ width: 30, height: 26, border: 'none', borderRadius: 4, background: 'none', cursor: 'pointer' }}
          />
          <span style={{ fontSize: 11, opacity: 0.7, fontFamily: 'monospace' }}>
            {((shape.type === 'label' || shape.type === 'number') ? shape.color : (shape as any).color)?.toUpperCase()}
          </span>
        </div>
      </Field>

      {hasStyle && 'opacity' in shape && (
        <Field label={`Opacity — ${Math.round((shape as any).opacity * 100)}%`}>
          <input
            type="range" min={0.1} max={1} step={0.05}
            value={(shape as any).opacity}
            onChange={e => onChange({ opacity: parseFloat(e.target.value) } as Partial<Shape>)}
            style={{ width: '100%' }}
          />
        </Field>
      )}

      {hasStyle && 'dash' in shape && (
        <Field label="Line style">
          <div style={{ display: 'flex', gap: 4 }}>
            {DASH_OPTIONS.map(opt => (
              <button
                key={opt.key}
                onClick={() => onChange({ dash: opt.key } as Partial<Shape>)}
                style={{
                  flex: 1, padding: '5px 0', fontSize: 12, borderRadius: 5,
                  border: `1.5px solid ${(shape as any).dash === opt.key ? '#4ade80' : 'rgba(255,255,255,0.15)'}`,
                  background: (shape as any).dash === opt.key ? 'rgba(74,222,128,0.15)' : 'transparent',
                  color: '#fff', cursor: 'pointer',
                }}
              >{opt.label}</button>
            ))}
          </div>
        </Field>
      )}

      <button
        onClick={onRemove}
        style={{ width: '100%', fontSize: 12, fontWeight: 600, padding: '7px 0', borderRadius: 6, border: 'none', background: '#b91c1c', color: '#fff', cursor: 'pointer', marginTop: 2 }}
      >
        Remove
      </button>
    </div>
  )
}
