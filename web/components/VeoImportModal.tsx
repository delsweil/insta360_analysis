'use client'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface VeoEvent {
  start: number
  type: string          // 'goal' | 'shot_on_goal' | …
  thumbnail?: string
  tags?: { name: string; slug: string }[]
}

interface ImportRow {
  start: number
  type: string
  thumbnail?: string
  // which team the coach assigns — null = unset (rendered as pending)
  team: 'home' | 'away'
}

interface Props {
  gameId: string
  homeTeam: string
  awayTeam: string
  userId: string
  onClose: () => void
  onImported: (count: number) => void
}

export default function VeoImportModal({ gameId, homeTeam, awayTeam, userId, onClose, onImported }: Props) {
  const [raw, setRaw] = useState('')
  const [rows, setRows] = useState<ImportRow[] | null>(null)
  const [parseError, setParseError] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')

  // Map Veo event type → base label (without _home / _away suffix)
  function baseLabel(type: string): 'goal' | 'shot' {
    if (type === 'goal') return 'goal'
    return 'shot'
  }

  function parseJson() {
    setParseError('')
    try {
      const parsed = JSON.parse(raw)
      // Accept either { highlights: [...] } or a plain array
      const events: VeoEvent[] = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed?.highlights)
        ? parsed.highlights
        : []

      if (events.length === 0) {
        setParseError('No events found. Expected an array or { highlights: [...] }.')
        return
      }

      const supported = events.filter(e => e.type === 'goal' || e.type === 'shot_on_goal')

      if (supported.length === 0) {
        setParseError(`Found ${events.length} events but none were goals or shots on goal.`)
        return
      }

      setRows(
        supported.map(e => ({
          start: e.start,
          type: e.type,
          thumbnail: e.thumbnail,
          team: 'home', // default — coach will correct
        }))
      )
    } catch {
      setParseError('Invalid JSON. Paste the full JSON from Veo.')
    }
  }

  function toggleTeam(idx: number) {
    setRows(prev =>
      prev!.map((r, i) => i === idx ? { ...r, team: r.team === 'home' ? 'away' : 'home' } : r)
    )
  }

  async function handleImport() {
    if (!rows) return
    setImporting(true)
    setImportError('')

    const records = rows.map(r => ({
      game_id: gameId,
      user_id: userId,
      timestamp_sec: r.start,
      label: `${baseLabel(r.type)}_${r.team}`,
      note: `Veo import · ${r.type === 'goal' ? 'Goal' : 'Shot on goal'}`,
      is_public: true,
    }))

    const { error } = await supabase.from('annotations').insert(records)
    if (error) {
      setImportError(error.message)
      setImporting(false)
      return
    }

    onImported(records.length)
    onClose()
  }

  const overlay: React.CSSProperties = {
    position: 'fixed', inset: 0,
    background: 'rgba(15,41,114,0.45)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    zIndex: 9000, padding: 16,
  }

  const modal: React.CSSProperties = {
    background: '#fff', borderRadius: 16,
    width: '100%', maxWidth: 520,
    maxHeight: '90vh', overflow: 'hidden',
    display: 'flex', flexDirection: 'column',
    fontFamily: 'DM Sans, sans-serif',
    boxShadow: '0 20px 60px rgba(15,41,114,0.18)',
  }

  const header: React.CSSProperties = {
    padding: '16px 18px 12px',
    borderBottom: '1px solid #E4E6EE',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  }

  const body: React.CSSProperties = {
    flex: 1, overflowY: 'auto', padding: '14px 18px',
  }

  const footer: React.CSSProperties = {
    padding: '12px 18px',
    borderTop: '1px solid #E4E6EE',
    display: 'flex', gap: 8,
  }

  const labelStyle: React.CSSProperties = {
    fontSize: 11, fontWeight: 600, color: '#4A4F5C',
    display: 'block', marginBottom: 4,
  }

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '8px 10px',
    border: '1px solid #E4E6EE', borderRadius: 8,
    fontSize: 12, fontFamily: 'DM Sans, sans-serif',
    color: '#111318', outline: 'none', background: '#F8F8F6',
    resize: 'vertical' as const, boxSizing: 'border-box' as const,
  }

  return (
    <div style={overlay} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={modal}>
        <div style={header}>
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 20, color: '#0f2972', letterSpacing: '0.02em',
            }}>
              Import from Veo
            </div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 1 }}>
              Paste the Veo highlights JSON — goals &amp; shots will be imported as annotations.
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              fontSize: 18, color: '#8A8F9E', background: 'none',
              border: 'none', cursor: 'pointer', padding: '4px 8px', lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        <div style={body}>
          {!rows ? (
            <>
              <label style={labelStyle}>Veo JSON</label>
              <textarea
                rows={8}
                placeholder={'Paste Veo highlights JSON here…\n\nTip: In Veo, open the game → Highlights → Export / copy JSON.'}
                value={raw}
                onChange={e => { setRaw(e.target.value); setParseError('') }}
                style={inputStyle}
              />
              {parseError && (
                <div style={{
                  fontSize: 11, color: '#b91c1c',
                  background: '#fef2f2', border: '1px solid #fecaca',
                  borderRadius: 6, padding: '7px 10px', marginTop: 8,
                }}>
                  {parseError}
                </div>
              )}
              <div style={{ marginTop: 10, fontSize: 11, color: '#8A8F9E', lineHeight: 1.5 }}>
                Veo does not identify which team scored — after parsing you&apos;ll assign each event to {homeTeam} or {awayTeam} before importing.
              </div>
            </>
          ) : (
            <>
              <div style={{
                fontSize: 11, color: '#4A4F5C', marginBottom: 10,
                background: '#FEF0E0', border: '1px solid #fcd38a',
                borderRadius: 6, padding: '7px 10px',
              }}>
                ⚠️ Veo doesn&apos;t say which team scored. Toggle each event below before importing.
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {rows.map((row, idx) => {
                  const isHome = row.team === 'home'
                  const typeLabel = row.type === 'goal' ? 'Goal' : 'Shot on goal'
                  const teamLabel = isHome ? homeTeam : awayTeam
                  const teamColor = isHome ? '#0f2972' : '#E8780A'

                  return (
                    <div
                      key={idx}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        padding: '7px 10px',
                        border: '1px solid #E4E6EE', borderRadius: 8,
                        background: '#fff',
                      }}
                    >
                      {row.thumbnail && (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={row.thumbnail}
                          alt=""
                          style={{ width: 48, height: 27, objectFit: 'cover', borderRadius: 4, flexShrink: 0 }}
                        />
                      )}
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 12, fontWeight: 600, color: '#111318' }}>
                          {typeLabel}
                        </div>
                        <div style={{
                          fontFamily: 'Bebas Neue, sans-serif',
                          fontSize: 13, color: '#0f2972', letterSpacing: '0.04em',
                        }}>
                          {formatTime(row.start)}
                        </div>
                      </div>

                      {/* Team toggle */}
                      <button
                        onClick={() => toggleTeam(idx)}
                        style={{
                          fontSize: 11, fontWeight: 700,
                          padding: '5px 12px', borderRadius: 99,
                          border: `2px solid ${teamColor}`,
                          background: `${teamColor}18`,
                          color: teamColor, cursor: 'pointer',
                          fontFamily: 'DM Sans, sans-serif',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {teamLabel} ⇄
                      </button>
                    </div>
                  )
                })}
              </div>

              <div style={{ marginTop: 10, fontSize: 11, color: '#8A8F9E' }}>
                {rows.length} event{rows.length !== 1 ? 's' : ''} ready to import.
                Click a team button to toggle between {homeTeam} and {awayTeam}.
              </div>

              {importError && (
                <div style={{
                  fontSize: 11, color: '#b91c1c',
                  background: '#fef2f2', border: '1px solid #fecaca',
                  borderRadius: 6, padding: '7px 10px', marginTop: 8,
                }}>
                  {importError}
                </div>
              )}
            </>
          )}
        </div>

        <div style={footer}>
          <button
            onClick={rows ? () => setRows(null) : onClose}
            style={{
              flex: 1, padding: '10px',
              background: '#fff', color: '#0f2972',
              border: '2px solid #0f2972', borderRadius: 8,
              fontSize: 13, fontWeight: 600, cursor: 'pointer',
              fontFamily: 'DM Sans, sans-serif',
            }}
          >
            {rows ? '← Back' : 'Cancel'}
          </button>

          {!rows ? (
            <button
              onClick={parseJson}
              disabled={!raw.trim()}
              style={{
                flex: 2, padding: '10px',
                background: raw.trim() ? '#0f2972' : '#E4E6EE',
                color: raw.trim() ? '#fff' : '#8A8F9E',
                border: 'none', borderRadius: 8,
                fontSize: 13, fontWeight: 600,
                cursor: raw.trim() ? 'pointer' : 'default',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              Parse events →
            </button>
          ) : (
            <button
              onClick={handleImport}
              disabled={importing}
              style={{
                flex: 2, padding: '10px',
                background: importing ? '#E4E6EE' : '#0f2972',
                color: importing ? '#8A8F9E' : '#fff',
                border: 'none', borderRadius: 8,
                fontSize: 13, fontWeight: 600,
                cursor: importing ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              {importing ? 'Importing…' : `Import ${rows.length} annotation${rows.length !== 1 ? 's' : ''}`}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
