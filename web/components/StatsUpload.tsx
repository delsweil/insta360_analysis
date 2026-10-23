'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { type HalfStats, readChartFiles } from '@/lib/stats'
import { readable } from '@/components/StatsPanel'

const NAVY = '#0f2972'
const ORANGE = '#E8780A'

type Entry = { key: string; file: string; half: string; data: HalfStats; fresh: boolean }

export type StatsDraft = {
  /** the halves that should be stored when the game is saved */
  halves: Record<string, HalfStats>
  /** something was added, changed or removed */
  dirty: boolean
  /** a reason why this cannot be saved yet (two files for one half) */
  problem: string | null
}

const halfName = (h: string) => `${h}. Halbzeit`
const minutes = (s: number) => (s > 0 ? `${Math.round(s / 60)} min` : '')

/** Upload of the analysis app's "*_chartdata.json" files (one per half) on the add-game and edit-game screens.
 *  `existing` = what is stored already (edit screen). Calls `onChange` with the halves to store. */
export default function StatsUpload({ existing, onChange }: {
  existing: Record<string, HalfStats> | null
  onChange: (d: StatsDraft) => void
}) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [dirty, setDirty] = useState(false)
  const [notes, setNotes] = useState<string[]>([])
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const seeded = useRef(false)

  // what is stored already shows up as the first rows (once)
  useEffect(() => {
    if (seeded.current || !existing) return
    seeded.current = true
    setEntries(Object.entries(existing).sort(([a], [b]) => a.localeCompare(b))
      .map(([half, data]) => ({ key: `old${half}`, file: 'gespeichert', half, data, fresh: false })))
  }, [existing])

  const counts: Record<string, number> = {}
  for (const e of entries) counts[e.half] = (counts[e.half] ?? 0) + 1
  const clash = Object.entries(counts).find(([, n]) => n > 1)?.[0] ?? null

  useEffect(() => {
    const halves: Record<string, HalfStats> = {}
    for (const e of entries) halves[e.half] = e.data
    onChange({ halves, dirty, problem: clash ? `Zwei Dateien für die ${halfName(clash)}. Bitte eine entfernen oder die Halbzeit ändern.` : null })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries, dirty, clash])

  const add = useCallback(async (list: FileList | File[]) => {
    const files = Array.from(list)
    if (!files.length) return
    const read = await readChartFiles(files)
    const msgs: string[] = []
    setEntries(prev => {
      let next = [...prev]
      for (const r of read) {
        if (r.error || !r.data) { msgs.push(`${r.file}: ${r.error}`); continue }
        // the name says the half; otherwise the first free one. A new file replaces the one stored for that half.
        let half = r.half
        if (!half) { half = ['1', '2', '3', '4'].find(h => !next.some(e => e.half === h)) ?? '1' }
        next = next.filter(e => e.half !== half)
        next.push({ key: `${Date.now()}-${r.file}`, file: r.file, half, data: r.data, fresh: true })
      }
      return next.sort((a, b) => a.half.localeCompare(b.half))
    })
    setNotes(msgs)
    setDirty(true)
  }, [])

  const setHalf = (key: string, half: string) => {
    setEntries(prev => prev.map(e => (e.key === key ? { ...e, half } : e)).sort((a, b) => a.half.localeCompare(b.half)))
    setDirty(true)
  }
  const remove = (key: string) => { setEntries(prev => prev.filter(e => e.key !== key)); setDirty(true) }

  return (
    <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 10, padding: '12px 14px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8, marginBottom: 2 }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: NAVY }}>📊 Spielstatistik</label>
        <span style={{ fontSize: 11, color: '#8A8F9E' }}>optional</span>
      </div>
      <div style={{ fontSize: 11, color: '#8A8F9E', marginBottom: 10, lineHeight: 1.5 }}>
        Die Dateien <b>…_half1_chartdata.json</b> und <b>…_half2_chartdata.json</b> aus dem Ordner der Analyse-App. Die Halbzeit wird am Dateinamen erkannt.
      </div>

      <div
        onClick={() => input.current?.click()}
        onDragOver={e => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={e => { e.preventDefault(); setOver(false); void add(e.dataTransfer.files) }}
        role="button" tabIndex={0}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') input.current?.click() }}
        style={{
          border: `1.5px dashed ${over ? ORANGE : '#c5cbe0'}`, background: over ? '#fff7ee' : '#F8F9FC',
          borderRadius: 10, padding: '14px 12px', textAlign: 'center', cursor: 'pointer',
          fontSize: 12, color: '#4A4F5C', transition: 'all 0.15s',
        }}
      >
        <span style={{ fontWeight: 600, color: NAVY }}>Dateien wählen</span> oder hierher ziehen
        <input
          ref={input} type="file" multiple accept=".json,application/json" style={{ display: 'none' }}
          onChange={e => { if (e.target.files) void add(e.target.files); e.target.value = '' }}
        />
      </div>

      {entries.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
          {entries.map(e => {
            const a = e.data.teams.A
            const b = e.data.teams.B
            return (
              <div key={e.key} style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', borderRadius: 8,
                border: `1px solid ${e.half === clash ? '#fca5a5' : '#E4E6EE'}`, background: '#fff',
              }}>
                <span style={{ fontSize: 15, color: e.fresh ? ORANGE : '#22c55e' }}>{e.fresh ? '●' : '✓'}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: '#111318', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <select
                      value={e.half} onChange={ev => setHalf(e.key, ev.target.value)}
                      style={{ fontSize: 12, fontWeight: 600, border: '1px solid #E4E6EE', borderRadius: 6, padding: '2px 4px', color: NAVY, background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}
                    >
                      {['1', '2', '3', '4'].map(h => <option key={h} value={h}>{halfName(h)}</option>)}
                    </select>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontWeight: 500, color: '#4A4F5C' }}>
                      <i style={{ width: 8, height: 8, borderRadius: '50%', background: readable(a.colour, false), display: 'inline-block' }} />{a.name || 'A'}
                      <span style={{ color: '#aab3c5' }}>·</span>
                      <i style={{ width: 8, height: 8, borderRadius: '50%', background: readable(b.colour, false), display: 'inline-block' }} />{b.name || 'B'}
                    </span>
                  </div>
                  <div style={{ fontSize: 10, color: '#8A8F9E', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {e.fresh ? `${e.file} · wird beim Speichern hochgeladen` : 'gespeichert'}{minutes(e.data.duration_s) && ` · ${minutes(e.data.duration_s)}`}
                  </div>
                </div>
                <button
                  type="button" onClick={() => remove(e.key)} aria-label="Entfernen"
                  style={{ border: 'none', background: 'transparent', color: '#8A8F9E', fontSize: 16, cursor: 'pointer', padding: 4 }}
                >✕</button>
              </div>
            )
          })}
        </div>
      )}

      {clash && (
        <div style={{ fontSize: 11, color: '#b91c1c', background: '#fef2f2', borderRadius: 8, padding: '6px 10px', marginTop: 8 }}>
          Zwei Dateien für die {halfName(clash)}. Bitte eine entfernen oder die Halbzeit ändern.
        </div>
      )}
      {notes.map((n, i) => (
        <div key={i} style={{ fontSize: 11, color: '#b91c1c', background: '#fef2f2', borderRadius: 8, padding: '6px 10px', marginTop: 8 }}>{n}</div>
      ))}
    </div>
  )
}

export const EMPTY_DRAFT: StatsDraft = { halves: {}, dirty: false, problem: null }
