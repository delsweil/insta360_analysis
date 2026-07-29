'use client'

// components/DrawTools.tsx
// Tool selector for the tactical annotation canvas (Select / Zone / Arrow / Label),
// styled to match the LabelPicker pill-button pattern already used in game/[id]/page.tsx.

import type { Tool } from './AnnotationCanvas'

interface DrawToolsProps {
  tool: Tool
  onChange: (t: Tool) => void
  onUndo: () => void
  onClear: () => void
  onDone: () => void
  canUndo: boolean
}

const TOOLS: { key: Tool; label: string; icon: string }[] = [
  { key: 'select', label: 'Select', icon: '↖' },
  { key: 'zone',   label: 'Zone',   icon: '▭' },
  { key: 'arrow',  label: 'Arrow',  icon: '↗' },
  { key: 'label',  label: 'Label',  icon: 'T' },
]

export default function DrawTools({ tool, onChange, onUndo, onClear, onDone, canUndo }: DrawToolsProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      flexWrap: 'wrap', padding: '8px 0',
    }}>
      {TOOLS.map(t => {
        const isSelected = tool === t.key
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              fontSize: 11, fontWeight: 600,
              padding: '4px 10px', borderRadius: 99,
              cursor: 'pointer',
              border: `1.5px solid ${isSelected ? '#4ade80' : '#E4E6EE'}`,
              background: isSelected ? '#4ade8018' : '#fff',
              color: isSelected ? '#166534' : '#8A8F9E',
              whiteSpace: 'nowrap', flexShrink: 0,
            }}
          >
            <span style={{ fontSize: 12 }}>{t.icon}</span>
            {t.label}
          </button>
        )
      })}

      <div style={{ flex: 1 }} />

      <button
        onClick={onUndo}
        disabled={!canUndo}
        style={{
          fontSize: 11, fontWeight: 600,
          padding: '4px 10px', borderRadius: 99,
          border: '1.5px solid #E4E6EE',
          background: '#fff',
          color: canUndo ? '#8A8F9E' : '#C0C4CE',
          cursor: canUndo ? 'pointer' : 'default',
        }}
      >
        Undo
      </button>

      <button
        onClick={onClear}
        style={{
          fontSize: 11, fontWeight: 600,
          padding: '4px 10px', borderRadius: 99,
          border: '1.5px solid #fde0e0',
          background: '#fff',
          color: '#b91c1c',
          cursor: 'pointer',
        }}
      >
        Clear
      </button>

      <button
        onClick={onDone}
        style={{
          fontSize: 11, fontWeight: 700,
          padding: '4px 12px', borderRadius: 99,
          border: 'none',
          background: '#0f2972',
          color: '#fff',
          cursor: 'pointer',
        }}
      >
        Done
      </button>
    </div>
  )
}
