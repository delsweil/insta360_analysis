'use client'

import { useMemo, useState } from 'react'
import type { Annotation } from '@/lib/supabase'
import { createClipLink, drawingsIn } from '@/lib/clipLinks'

function formatTime(sec: number) {
  const s = Math.max(0, sec)
  const m = Math.floor(s / 60)
  const r = s - m * 60
  return `${m}:${r.toFixed(1).padStart(4, '0')}`
}

interface Props {
  gameId: string
  annotations: Annotation[]
  currentTime: number
  duration: number
  parentToken?: string            // set when a viewer cuts a clip from a shared link
  limit?: [number, number] | null // the stretch a viewer may cut from
  onPreview: (start: number, end: number) => void
  onClose: () => void
  hidden?: boolean   // while the clip is previewing, the dialog steps aside so the video is visible
}

const LEAD_IN = 12   // same window the suggestions use: build-up before the moment ...
const LEAD_OUT = 8   // ... and the finish after it

const NUDGES = [-5, -1, -0.2, 0.2, 1, 5]

export default function ClipModal({ gameId, annotations, currentTime, duration, parentToken, limit, onPreview, onClose, hidden }: Props) {
  const lo = limit ? limit[0] : 0
  const hi = limit ? limit[1] : (duration > 0 ? duration : Infinity)
  const clamp = (v: number) => Math.min(Math.max(lo, v), hi)
  const [start, setStart] = useState(() => clamp(currentTime - LEAD_IN))
  const [end, setEnd] = useState(() => clamp(currentTime + LEAD_OUT))
  const [includeDrawings, setIncludeDrawings] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState('')
  const [url, setUrl] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const valid = end - start >= 1

  // Coach drawings (tactical annotations with shapes) that are on screen at some point inside the clip.
  const drawings = useMemo(() => drawingsIn(annotations, start, end), [annotations, start, end])

  const move = (which: 'start' | 'end', delta: number) => {
    setUrl(null)
    if (which === 'start') setStart(s => clamp(Math.min(s + delta, end - 1)))
    else setEnd(e => clamp(Math.max(e + delta, start + 1)))
  }

  const generate = async () => {
    if (!valid) return
    setGenerating(true); setError('')
    const made = await createClipLink({ gameId, parentToken, annotations, start, end, includeDrawings })
    if (made.url) setUrl(made.url)
    else setError(made.error ?? 'Could not create the link.')
    setGenerating(false)
  }

  const copy = async () => {
    if (!url) return
    await navigator.clipboard.writeText(url)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const btn = (primary = false): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '7px 12px', borderRadius: 8, cursor: 'pointer',
    border: primary ? 'none' : '1px solid #E4E6EE',
    background: primary ? '#0f2972' : '#fff', color: primary ? '#fff' : '#0f2972',
    fontFamily: 'DM Sans, sans-serif',
  })

  const row = (label: string, which: 'start' | 'end', value: number) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      <div style={{ width: 44, fontSize: 12, color: '#8A8F9E' }}>{label}</div>
      <div style={{ width: 64, fontFamily: 'Bebas Neue, sans-serif', fontSize: 20, color: '#0f2972' }}>{formatTime(value)}</div>
      {NUDGES.map(d => (
        <button key={d} onClick={() => move(which, d)} style={{ ...btn(), padding: '4px 8px', fontSize: 11 }}>
          {d > 0 ? `+${d}` : d}
        </button>
      ))}
      <button
        onClick={() => {
          setUrl(null)
          if (which === 'start') setStart(clamp(Math.min(currentTime, end - 1)))
          else setEnd(clamp(Math.max(currentTime, start + 1)))
        }}
        style={{ ...btn(), padding: '4px 8px', fontSize: 11 }}
        title="Use the time the video is at now"
      >
        Now
      </button>
    </div>
  )

  return (
    <div
      style={{ display: hidden ? 'none' : 'flex', position: 'fixed', inset: 0, zIndex: 100, background: 'rgba(15,41,114,0.5)', alignItems: 'center', justifyContent: 'center', padding: 20 }}
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
    >
      <div style={{ background: '#fff', borderRadius: 16, width: '100%', maxWidth: 560, maxHeight: '85vh', overflowY: 'auto' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #E4E6EE', display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 20, color: '#0f2972', letterSpacing: '0.04em' }}>Clip link</div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 2 }}>One clip, nothing else: the viewer cannot go past it.{limit ? ' You can cut from the highlights that were shared.' : ''}</div>
          </div>
          <button onClick={onClose} style={{ fontSize: 18, color: '#8A8F9E', background: 'none', border: 'none', cursor: 'pointer' }}>✕</button>
        </div>

        <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button style={btn()} onClick={() => { setUrl(null); setStart(clamp(currentTime - LEAD_IN)); setEnd(clamp(currentTime + LEAD_OUT)) }}>
              Around now (−{LEAD_IN} s / +{LEAD_OUT} s)
            </button>
            <button style={btn()} onClick={() => { setUrl(null); setStart(clamp(currentTime - 5)); setEnd(clamp(currentTime + 5)) }}>±5 s</button>
            <button style={btn()} onClick={() => { setUrl(null); setStart(clamp(currentTime - 20)); setEnd(clamp(currentTime)) }}>Last 20 s</button>
          </div>

          {row('Start', 'start', start)}
          {row('End', 'end', end)}

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <button style={btn()} disabled={!valid} onClick={() => onPreview(start, end)}>▶ Preview clip</button>
            <span style={{ fontSize: 12, color: valid ? '#4A4F5C' : '#ef4444' }}>
              {valid ? `${Math.round(end - start)} s long` : 'The clip must be at least 1 s long'}
            </span>
          </div>

          <label style={{ display: 'flex', gap: 8, alignItems: 'flex-start', fontSize: 13, color: '#111318', cursor: 'pointer' }}>
            <input type="checkbox" checked={includeDrawings} onChange={e => { setIncludeDrawings(e.target.checked); setUrl(null) }} style={{ marginTop: 2 }} />
            <span>
              {parentToken ? 'Include the drawings' : 'Include my drawings'} ({drawings.length} in this clip)
              <span style={{ display: 'block', fontSize: 11, color: '#8A8F9E' }}>
                {parentToken ? 'Drawn on the video by the coach.' : 'Drawn on the video with the pen tools; they appear for the time you set on the timing strip.'}
              </span>
            </span>
          </label>

          {error && <div style={{ fontSize: 12, color: '#ef4444' }}>{error}</div>}

          {url ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ background: '#F8F8F6', borderRadius: 8, padding: '10px 14px', fontSize: 12, color: '#4A4F5C', wordBreak: 'break-all', border: '1px solid #E4E6EE' }}>{url}</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={copy} style={{ ...btn(true), flex: 1, background: copied ? '#22c55e' : '#0f2972' }}>{copied ? '✓ Copied!' : 'Copy link'}</button>
                <button onClick={() => setUrl(null)} style={btn()}>New link</button>
              </div>
            </div>
          ) : (
            <button onClick={generate} disabled={!valid || generating} style={{ ...btn(true), padding: 12, fontSize: 14, opacity: valid ? 1 : 0.5 }}>
              {generating ? 'Generating...' : 'Create clip link'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
