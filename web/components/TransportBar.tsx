'use client'

// components/TransportBar.tsx
// Desktop playback controls under the video: play/pause, quick jumps back and forward, single frames, playback speed.
// The same keys work from the keyboard (see the title of the ⌨ button); the page owns the shortcuts.

export const RATES = [0.25, 0.5, 1, 1.5, 2]
export const JUMPS = [-30, -10, -5, 5, 10, 30]

interface Props {
  isPlaying: boolean
  timeLabel: string
  rate: number
  onTogglePlay: () => void
  onSeekBy: (delta: number) => void
  onStep: (dir: 1 | -1) => void
  onRate: (r: number) => void
}

const btn: React.CSSProperties = {
  height: 28, minWidth: 34, padding: '0 7px', borderRadius: 8, border: '1px solid #E4E6EE', background: '#fff',
  color: '#0f2972', fontSize: 11, fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center',
  justifyContent: 'center', fontFamily: 'DM Sans, sans-serif', flexShrink: 0,
}

export default function TransportBar({ isPlaying, timeLabel, rate, onTogglePlay, onSeekBy, onStep, onRate }: Props) {
  const jump = (d: number) => (
    <button key={d} onClick={() => onSeekBy(d)} title={`${d < 0 ? 'Back' : 'Forward'} ${Math.abs(d)} s`} style={btn}>
      {d < 0 ? `−${Math.abs(d)}` : `+${d}`}
    </button>
  )
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
      {JUMPS.filter(d => d < 0).map(jump)}
      <button onClick={() => onStep(-1)} title="One frame back (,)" style={btn}>◀|</button>
      <button
        onClick={onTogglePlay}
        title={isPlaying ? 'Pause (space)' : 'Play (space)'}
        style={{ ...btn, width: 40, background: '#0f2972', borderColor: '#0f2972', color: '#fff' }}
      >
        {isPlaying ? (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><rect x="1" y="0" width="4" height="12" /><rect x="7" y="0" width="4" height="12" /></svg>
        ) : (
          <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor"><polygon points="1,0 11,6 1,12" /></svg>
        )}
      </button>
      <button onClick={() => onStep(1)} title="One frame forward (.)" style={btn}>|▶</button>
      {JUMPS.filter(d => d > 0).map(jump)}

      <span style={{
        fontFamily: 'Bebas Neue, sans-serif', fontSize: 16, color: '#0f2972', letterSpacing: '0.04em', margin: '0 6px 0 8px',
      }}>
        {timeLabel}
      </span>

      <div style={{ display: 'flex', border: '1px solid #E4E6EE', borderRadius: 8, overflow: 'hidden', flexShrink: 0 }}>
        {RATES.map(r => (
          <button
            key={r}
            onClick={() => onRate(r)}
            title={`Speed ${r}× ([ and ])`}
            style={{
              height: 28, padding: '0 8px', border: 'none', fontSize: 11, fontWeight: 700, cursor: 'pointer',
              background: rate === r ? '#E8780A' : '#fff', color: rate === r ? '#fff' : '#8A8F9E',
              fontFamily: 'DM Sans, sans-serif',
            }}
          >
            {r}×
          </button>
        ))}
      </div>

      <span
        title={'Keyboard: space or K = play/pause · ← → = 5 s · Shift+← → = 1 s · J L = 10 s · , . = one frame · [ ] = speed · D = draw · 1–8 = tool · Ctrl/⌘+Z = undo'}
        style={{ fontSize: 14, color: '#8A8F9E', cursor: 'help', padding: '0 4px' }}
      >
        ⌨
      </span>
    </div>
  )
}
