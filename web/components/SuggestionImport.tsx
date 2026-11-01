'use client'

// Loads the shots and goals the analysis app suggests (`<game>_suggestions.json`, made by suggest.py) as ordinary
// annotations: the coach checks them, moves their start / end, changes the label or deletes them like any other.
// Every one carries a note "Suggested (model id, score, moment)" so that later the suggestions can be compared with what the
// coach kept, moved or deleted: that comparison is how the suggestions get better (suggest.py review).
import { useState } from 'react'
import { supabase } from '@/lib/supabase'
import type { Annotation } from '@/lib/supabase'

type Suggestion = {
  id: string
  label: string        // shot_home | shot_away
  t_peak: number       // the moment the ball is furthest forward, board seconds
  t_start: number
  t_end: number
  score: number
}

type Row = Suggestion & { on: boolean; dup: boolean }

const fmt = (sec: number) => {
  const s = Math.max(0, Math.round(sec))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export const suggestedNote = (model: string, s: Suggestion) =>
  `Suggested (${model} ${s.id}, score ${s.score}, moment ${s.t_peak})`

const pill: React.CSSProperties = {
  fontSize: 10, fontWeight: 600, padding: '3px 8px', borderRadius: 99, border: '1.5px solid #E4E6EE',
  background: '#fff', color: '#4A4F5C', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', whiteSpace: 'nowrap',
}

type Props = {
  gameId: string
  userId: string
  shift: number                 // the board's saved sync shift (s): annotation time = suggestion time - shift
  existing: Annotation[]        // so that a suggestion that is already in is not added twice
  homeTeam: string
  awayTeam: string
  onImported: () => void
}

export default function SuggestionImport({ gameId, userId, shift, existing, homeTeam, awayTeam, onImported }: Props) {
  const [open, setOpen] = useState(false)
  const [model, setModel] = useState('')
  const [sh, setSh] = useState(0)                // the shift that applies to this file (0 when its times are video times already)
  const [rows, setRows] = useState<Row[] | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const close = () => { setOpen(false); setRows(null); setError(''); setBusy(false) }

  async function pick(file: File | undefined) {
    if (!file) return
    setError('')
    try {
      const j = JSON.parse(await file.text())
      if (j?.format !== 'football-autopan-suggestions' || !Array.isArray(j.suggestions)) {
        throw new Error('This is not a suggestions file (made by suggest.py).')
      }
      const m = String(j.model ?? 'model')
      const have = new Set(existing.map(a => a.note ?? ''))
      setModel(m)
      setSh(j.time_base === 'video' ? 0 : shift)
      setRows((j.suggestions as Suggestion[]).map(s => {
        const dup = have.has(suggestedNote(m, s))
        return { ...s, on: !dup, dup }
      }))
    } catch (e) {
      setRows(null)
      setError(e instanceof Error ? e.message : 'Could not read the file.')
    }
  }

  async function doImport() {
    if (!rows) return
    const todo = rows.filter(r => r.on && !r.dup)
    if (todo.length === 0) { close(); return }
    setBusy(true)
    const { error: err } = await supabase.from('annotations').insert(todo.map(r => ({
      game_id: gameId,
      user_id: userId,
      timestamp_sec: Math.max(0, r.t_start - sh),
      end_timestamp_sec: Math.max(0, r.t_end - sh),
      label: r.label,
      note: suggestedNote(model, r),
      is_public: false,                 // not shown to others until the coach has looked at them
    })))
    if (err) { setError(err.message); setBusy(false); return }
    onImported()
    close()
  }

  const on = rows?.filter(r => r.on && !r.dup).length ?? 0
  return (
    <>
      <button onClick={() => setOpen(true)} style={pill}
        title="Load the shots and goals the analysis suggests, to check them like any other annotations">+ Suggestions</button>
      {open && (
        <div onClick={e => { if (e.target === e.currentTarget) close() }}
          style={{ position: 'fixed', inset: 0, background: 'rgba(15,41,114,0.45)', display: 'flex', alignItems: 'center',
            justifyContent: 'center', zIndex: 9000, padding: 16 }}>
          <div style={{ background: '#fff', borderRadius: 16, width: '100%', maxWidth: 560, maxHeight: '90vh', overflow: 'hidden',
            display: 'flex', flexDirection: 'column', fontFamily: 'DM Sans, sans-serif' }}>
            <div style={{ padding: '16px 18px 12px', borderBottom: '1px solid #E4E6EE' }}>
              <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 20, color: '#0f2972' }}>Suggested shots and goals</div>
              <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 1 }}>
                Choose the <code>_suggestions.json</code> file. They are added as private annotations (start 12 s before the ball
                arrives): check, move or delete them. Change a shot to a goal where it was one.
              </div>
            </div>
            <div style={{ padding: '14px 18px', overflowY: 'auto', flex: 1 }}>
              <input type="file" accept=".json,application/json" onChange={e => pick(e.target.files?.[0])} />
              {error && <div style={{ color: '#c0392b', fontSize: 12, marginTop: 8 }}>{error}</div>}
              {rows && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: '#4A4F5C', marginBottom: 6 }}>
                    {rows.length} suggestions ({model}){rows.some(r => r.dup) ? `, ${rows.filter(r => r.dup).length} already added` : ''}.
                    Times are in the video as it is now{sh !== 0 ? ` (sync shift ${sh > 0 ? '+' : ''}${sh.toFixed(1)} s applied)` : ''}.
                  </div>
                  {rows.map((r, i) => (
                    <label key={r.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', fontSize: 12,
                      opacity: r.dup ? 0.45 : 1 }}>
                      <input type="checkbox" checked={r.on && !r.dup} disabled={r.dup}
                        onChange={e => setRows(rows.map((x, k) => k === i ? { ...x, on: e.target.checked } : x))} />
                      <span style={{ width: 54, fontVariantNumeric: 'tabular-nums' }}>{fmt(r.t_start - sh)}</span>
                      <span style={{ width: 120 }}>{r.label === 'shot_home' ? homeTeam : awayTeam}</span>
                      <span style={{ color: '#8A8F9E' }}>score {r.score}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
            <div style={{ padding: '12px 18px', borderTop: '1px solid #E4E6EE', display: 'flex', gap: 8 }}>
              <button onClick={close} style={{ ...pill, padding: '8px 14px', fontSize: 12 }}>Cancel</button>
              <button onClick={doImport} disabled={!rows || on === 0 || busy}
                style={{ ...pill, padding: '8px 14px', fontSize: 12, background: '#0f2972', color: '#fff', border: 'none',
                  opacity: !rows || on === 0 || busy ? 0.5 : 1 }}>
                {busy ? 'Adding…' : `Add ${on} as annotations`}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
