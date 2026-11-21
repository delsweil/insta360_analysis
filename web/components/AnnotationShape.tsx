// components/AnnotationShape.tsx
// Renders the correct SVG shape for an annotation label

import { labelColor, labelDef } from '@/lib/supabase'

interface Props {
  labelKey: string
  size?: number        // diameter in px
  style?: React.CSSProperties
  /** overrides the label's own colour, e.g. the team's real shirt colour */
  color?: string
  /** dark outline so a light team colour stays visible on any bar */
  outline?: boolean
}

export default function AnnotationShape({ labelKey, size = 10, style, color: colorOverride, outline = false }: Props) {
  const def = labelDef(labelKey)
  const color = colorOverride ?? labelColor(labelKey)
  const ol = outline ? { stroke: '#0a1020', strokeWidth: 1.3, strokeLinejoin: 'round' as const } : {}
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
        <circle cx="5" cy="5" r="4.2" fill={color} {...ol} />
      )}
      {shape === 'square' && (
        <rect x="0.7" y="0.7" width="8.6" height="8.6" fill={color} rx="1" {...ol} />
      )}
      {shape === 'triangle' && (
        <polygon points="5,0.7 9.3,9.3 0.7,9.3" fill={color} {...ol} />
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
