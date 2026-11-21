// components/MatchStrip.tsx
// A thin momentum silhouette on the timeline's own time axis: the home team's area rises above the centre line
// when it had the play in the other half, the away team's hangs below it. It draws nothing without statistics.
import { visible, type Strip } from '@/lib/matchStrip'

interface Props {
  strip: Strip
  duration: number
  height: number
  /** over the dark video HUD (true) or on the light desktop bar (false) */
  dark?: boolean
  opacity?: number
  style?: React.CSSProperties
}

const W = 1000 // the drawing is stretched to the bar's width: no resize handling needed

export default function MatchStrip({ strip, duration, height, dark = false, opacity = 0.85, style }: Props) {
  if (!strip.runs.length || duration <= 0) return null
  const mid = height / 2
  // a thin strip needs the curve stretched: the strongest spell of the game fills it (never below 35 % of the range)
  const peak = Math.max(0.35, ...strip.runs.flatMap(r => r.map(p => Math.abs(p.v))))
  const amp = Math.max(1, mid - 1) / peak
  const x = (t: number) => ((t / duration) * W).toFixed(1)
  const area = (run: typeof strip.runs[number], sign: 1 | -1) => {
    const y = (v: number) => (mid - sign * Math.max(0, sign * v) * amp).toFixed(1)
    return `M${x(run[0].t)},${mid} ${run.map(p => `L${x(p.t)},${y(p.v)}`).join(' ')} L${x(run[run.length - 1].t)},${mid} Z`
  }
  const home = visible(strip.colours.home, dark)
  const away = visible(strip.colours.away, dark)
  return (
    <svg
      viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" width="100%" height={height}
      aria-hidden="true" focusable="false"
      style={{ display: 'block', pointerEvents: 'none', ...style }}
    >
      {strip.runs.map((run, i) => (
        <g key={i} opacity={opacity}>
          <path d={area(run, 1)} fill={home} />
          <path d={area(run, -1)} fill={away} />
        </g>
      ))}
      <line x1={0} x2={W} y1={mid} y2={mid} stroke={dark ? 'rgba(255,255,255,0.45)' : 'rgba(15,41,114,0.25)'} strokeWidth={1} vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
