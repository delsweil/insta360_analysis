// components/AnnotationShape.tsx
// Renders the correct SVG shape for an annotation label

import { labelColor, labelDef } from '@/lib/supabase'

interface Props {
  labelKey: string
  size?: number        // diameter in px
  style?: React.CSSProperties
}

export default function AnnotationShape({ labelKey, size = 10, style }: Props) {
  const def = labelDef(labelKey)
  const color = labelColor(labelKey)
  const shape = def?.shape ?? 'circle'
  const s = size

  return (
    <svg
      width={s}
      height={s}
      viewBox="0 0 10 10"
      style={{ flexShrink: 0, display: 'inline-block', ...style }}
    >
      {shape === 'circle' && (
        <circle cx="5" cy="5" r="4.5" fill={color} />
      )}
      {shape === 'square' && (
        <rect x="0.5" y="0.5" width="9" height="9" fill={color} rx="1" />
      )}
      {shape === 'triangle' && (
        <polygon points="5,0.5 9.5,9.5 0.5,9.5" fill={color} />
      )}
      {shape === 'cross' && (
        <>
          <line x1="1" y1="1" x2="9" y2="9" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <line x1="9" y1="1" x2="1" y2="9" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
        </>
      )}
    </svg>
  )
}
