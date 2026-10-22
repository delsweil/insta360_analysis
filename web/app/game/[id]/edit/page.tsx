'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin } from '@/lib/supabase'
import Topbar from '@/components/Topbar'

const ASN = 'ASN Pfeil Phönix'
const STATUSES = ['raw', 'calibrated', 'autopanned', 'uploaded'] as const

// Accepts watch, short, embed, shorts and live links, or a bare 11-character video id.
function extractYouTubeId(input: string): string | null {
  const s = input.trim()
  if (/^[a-zA-Z0-9_-]{11}$/.test(s)) return s
  const m = s.match(/(?:youtube\.com\/(?:watch\?(?:.*&)?v=|embed\/|shorts\/|live\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})/)
  return m ? m[1] : null
}

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '10px 14px', border: '1px solid #E4E6EE', borderRadius: 8,
  fontSize: 14, fontFamily: 'DM Sans, sans-serif', color: '#111318', outline: 'none',
  background: '#fff', boxSizing: 'border-box',
}
const labelStyle: React.CSSProperties = { fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4 }

interface Venue { id: string; name: string }

export default function EditGamePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = React.use(params)
  const router = useRouter()
  const [state, setState] = useState<'loading' | 'denied' | 'missing' | 'ready'>('loading')
  const [role, setRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [venues, setVenues] = useState<Venue[]>([])
  const [venueId, setVenueId] = useState('')
  const [homeAway, setHomeAway] = useState<'home' | 'away'>('home')
  const [opponent, setOpponent] = useState('')
  const [title, setTitle] = useState('')
  const [date, setDate] = useState('')
  const [status, setStatus] = useState<string>('raw')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [oldUrl, setOldUrl] = useState('')
  const [durationMin, setDurationMin] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const derive = (side: 'home' | 'away', opp: string) =>
    opp ? `${side === 'home' ? ASN : opp} vs. ${side === 'home' ? opp : ASN}` : ''

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      const r = await getCurrentUserRole()
      setRole(r)
      if (!isCoachOrAdmin(r)) { setState('denied'); return }
      const [{ data: g }, { data: v }] = await Promise.all([
        supabase.from('games').select('*').eq('id', id).single(),
        supabase.from('venues').select('id, name').order('name'),
      ])
      if (!g) { setState('missing'); return }
      setVenues(v ?? [])
      // older games may lack home_away / opponent: work them out from the team names
      const side: 'home' | 'away' = g.home_away === 'away' || (!g.home_away && g.away_team === ASN) ? 'away' : 'home'
      setHomeAway(side)
      setOpponent(g.opponent ?? (side === 'home' ? g.away_team : g.home_team) ?? '')
      setTitle(g.title ?? '')
      setDate(g.date ? String(g.date).slice(0, 10) : '')
      setVenueId(g.venue_id ?? '')
      setStatus(g.status ?? 'raw')
      setYoutubeUrl(g.video_url ?? '')
      setOldUrl(g.video_url ?? '')
      setDurationMin(g.duration_sec ? String(Math.round(g.duration_sec / 60)) : '')
      setState('ready')
    }
    load()
  }, [id, router])

  const videoId = extractYouTubeId(youtubeUrl)
  const newUrl = videoId ? `https://www.youtube.com/embed/${videoId}` : ''
  const urlChanged = newUrl !== oldUrl
  const homeTeam = homeAway === 'home' ? ASN : opponent
  const awayTeam = homeAway === 'home' ? opponent : ASN

  // the title follows side and opponent as long as it still is the automatic one
  const setSide = (side: 'home' | 'away') => {
    if (title === derive(homeAway, opponent)) setTitle(derive(side, opponent))
    setHomeAway(side)
  }
  const setOpp = (opp: string) => {
    if (title === derive(homeAway, opponent)) setTitle(derive(homeAway, opp))
    setOpponent(opp)
  }

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    if (!opponent.trim()) { setError('Bitte Gegner eingeben'); return }
    if (!title.trim()) { setError('Bitte einen Titel eingeben'); return }
    if (!date) { setError('Bitte Datum auswählen'); return }
    if (youtubeUrl.trim() && !videoId) { setError('Keine gültige YouTube URL'); return }
    setSaving(true)
    const patch: Record<string, unknown> = {
      title: title.trim(),
      opponent: opponent.trim(),
      home_team: homeTeam.trim(),
      away_team: awayTeam.trim(),
      home_away: homeAway,
      date,
      venue_id: venueId || null,
      status,
      video_url: videoId ? newUrl : null,
      duration_sec: durationMin ? Math.round(parseFloat(durationMin) * 60) : null,
    }
    const { data, error: err } = await supabase.from('games').update(patch).eq('id', id).select('id')
    setSaving(false)
    if (err) { setError(err.message); return }
    if (!data || data.length === 0) {
      setError('Nichts gespeichert: Die Datenbank erlaubt das Ändern nicht. Führe web/supabase/games_update.sql im Supabase SQL-Editor aus.')
      return
    }
    router.push(`/game/${id}`)
  }

  const shell = (children: React.ReactNode) => (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={role} backHref={`/game/${id}`} />
      <div style={{ padding: '24px 20px', maxWidth: 560, margin: '0 auto' }}>{children}</div>
    </div>
  )

  if (state === 'loading') return shell(<div style={{ color: '#4A4F5C' }}>Loading...</div>)
  if (state === 'denied') return shell(<div style={{ color: '#991b1b' }}>Nur Trainer und Admins können Spiele bearbeiten.</div>)
  if (state === 'missing') return shell(<div style={{ color: '#991b1b' }}>Spiel nicht gefunden.</div>)

  return shell(
    <>
      <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 28, color: '#0f2972', letterSpacing: '0.02em', marginBottom: 4 }}>
        Spiel bearbeiten
      </div>
      <div style={{ fontSize: 12, color: '#8A8F9E', marginBottom: 24 }}>
        Änderungen gelten sofort für alle, auch für bestehende Freigabe-Links.
      </div>

      <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

        {/* Home / Away toggle */}
        <div>
          <label style={labelStyle}>ASN spielt</label>
          <div style={{ display: 'flex', gap: 0, borderRadius: 8, overflow: 'hidden', border: '1px solid #E4E6EE' }}>
            {(['home', 'away'] as const).map(side => (
              <button
                key={side} type="button" onClick={() => setSide(side)}
                style={{
                  flex: 1, padding: '10px 0', fontSize: 13, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', border: 'none',
                  background: homeAway === side ? '#0f2972' : '#fff',
                  color: homeAway === side ? '#fff' : '#8A8F9E',
                }}
              >
                {side === 'home' ? '🏠 Heimspiel' : '✈️ Auswärtsspiel'}
              </button>
            ))}
          </div>
        </div>

        {/* Opponent */}
        <div>
          <label style={labelStyle}>Gegner</label>
          <input type="text" placeholder="FC Gegner" value={opponent} onChange={e => setOpp(e.target.value)} required style={inputStyle} />
        </div>

        {/* Title */}
        <div>
          <label style={labelStyle}>Titel <span style={{ fontWeight: 400, color: '#8A8F9E' }}>(folgt Gegner und Heim/Auswärts, solange du ihn nicht selbst änderst)</span></label>
          <input type="text" value={title} onChange={e => setTitle(e.target.value)} required style={inputStyle} />
        </div>

        {/* Date */}
        <div>
          <label style={labelStyle}>Datum</label>
          <input type="date" value={date} onChange={e => setDate(e.target.value)} required style={inputStyle} />
        </div>

        {/* Venue */}
        <div>
          <label style={labelStyle}>Spielstätte <span style={{ fontWeight: 400, color: '#8A8F9E' }}>(optional)</span></label>
          <select value={venueId} onChange={e => setVenueId(e.target.value)} style={{ ...inputStyle, background: '#F8F8F6' }}>
            <option value="">— Spielstätte wählen —</option>
            {venues.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
          </select>
        </div>

        {/* Status */}
        <div>
          <label style={labelStyle}>Status</label>
          <select value={status} onChange={e => setStatus(e.target.value)} style={{ ...inputStyle, background: '#F8F8F6' }}>
            {STATUSES.map(s => <option key={s} value={s}>{s}</option>)}
            {!STATUSES.includes(status as typeof STATUSES[number]) && <option value={status}>{status}</option>}
          </select>
        </div>

        {/* YouTube URL */}
        <div style={{ background: '#fff', border: '2px solid #0f2972', borderRadius: 10, padding: '12px 14px' }}>
          <label style={{ ...labelStyle, color: '#0f2972', fontSize: 13 }}>YouTube URL (Video des Spiels)</label>
          <input
            type="text" placeholder="https://www.youtube.com/watch?v=..." value={youtubeUrl}
            onChange={e => setYoutubeUrl(e.target.value)}
            style={{ ...inputStyle, border: `1px solid ${videoId ? '#22c55e' : '#E4E6EE'}` }}
          />
          {youtubeUrl.trim() && !videoId && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>Keine gültige YouTube URL erkannt</div>}
          {videoId && <div style={{ fontSize: 11, color: '#22c55e', marginTop: 4 }}>✓ Video ID: {videoId}{!urlChanged && ' (aktuelles Video)'}</div>}
          {!youtubeUrl.trim() && oldUrl && <div style={{ fontSize: 11, color: '#b45309', marginTop: 4 }}>Das Video wird entfernt.</div>}
          {urlChanged && oldUrl && newUrl && (
            <div style={{ fontSize: 11, color: '#92400e', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, padding: '8px 10px', marginTop: 8 }}>
              Neues Video: Annotationen behalten ihre Zeitpunkte (Sekunden im Video). Hat das neue Video eine andere Länge oder einen anderen Anfang,
              passen sie nicht mehr. Die Taktiktafel (Board) braucht den Zeitverlauf des zusammengefügten Gesamtvideos.
            </div>
          )}
          {videoId && (
            <div style={{ borderRadius: 10, overflow: 'hidden', aspectRatio: '16/9', background: '#000', marginTop: 10 }}>
              <iframe src={`https://www.youtube.com/embed/${videoId}`} style={{ width: '100%', height: '100%', border: 'none' }} allowFullScreen />
            </div>
          )}
        </div>

        {/* Duration */}
        <div>
          <label style={labelStyle}>Spieldauer (Minuten, optional)</label>
          <input type="number" min="0" step="any" placeholder="90" value={durationMin} onChange={e => setDurationMin(e.target.value)} style={inputStyle} />
        </div>

        {error && (
          <div style={{ fontSize: 12, color: '#ef4444', background: '#fef2f2', padding: '8px 12px', borderRadius: 8 }}>{error}</div>
        )}

        <div style={{ display: 'flex', gap: 8 }}>
          <button type="button" onClick={() => router.push(`/game/${id}`)}
            style={{ flex: 1, padding: 11, background: '#fff', color: '#0f2972', border: '2px solid #0f2972', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>
            Abbrechen
          </button>
          <button type="submit" disabled={saving}
            style={{ flex: 1, padding: 11, background: saving ? '#E4E6EE' : '#0f2972', color: saving ? '#8A8F9E' : '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: saving ? 'default' : 'pointer', fontFamily: 'DM Sans, sans-serif' }}>
            {saving ? 'Speichern...' : 'Änderungen speichern'}
          </button>
        </div>
      </form>
    </>
  )
}
