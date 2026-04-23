'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'
import Topbar from '@/components/Topbar'

const ASN = 'ASN Pfeil Phönix'

function extractYouTubeId(url: string): string | null {
  const patterns = [
    /youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/,
    /youtu\.be\/([a-zA-Z0-9_-]{11})/,
    /youtube\.com\/embed\/([a-zA-Z0-9_-]{11})/,
  ]
  for (const pattern of patterns) {
    const match = url.match(pattern)
    if (match) return match[1]
  }
  return null
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '10px 14px',
  border: '1px solid #E4E6EE',
  borderRadius: 8,
  fontSize: 14,
  fontFamily: 'DM Sans, sans-serif',
  color: '#111318',
  outline: 'none',
  background: '#fff',
  boxSizing: 'border-box',
}

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  fontWeight: 600,
  color: '#4A4F5C',
  display: 'block',
  marginBottom: 4,
}

interface Venue {
  id: string
  name: string
}

export default function AddGamePage() {
  const router = useRouter()
  const [venues, setVenues]       = useState<Venue[]>([])
  const [venueId, setVenueId]     = useState('')
  const [homeAway, setHomeAway]   = useState<'home' | 'away'>('home')
  const [opponent, setOpponent]   = useState('')
  const [date, setDate]           = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [durationMin, setDurationMin] = useState('')
  const [error, setError]         = useState('')
  const [saving, setSaving]       = useState(false)

  const videoId = extractYouTubeId(youtubeUrl)

  const homeTeam = homeAway === 'home' ? ASN : opponent
  const awayTeam = homeAway === 'home' ? opponent : ASN
  const title    = opponent ? `${homeTeam} vs. ${awayTeam}` : ''

  useEffect(() => {
    supabase.from('venues').select('id, name').order('name')
      .then(({ data }) => setVenues(data ?? []))
  }, [])

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!opponent.trim()) { setError('Bitte Gegner eingeben'); return }
    if (!date)            { setError('Bitte Datum auswählen'); return }
    if (youtubeUrl && !videoId) { setError('Keine gültige YouTube URL'); return }

    setSaving(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { router.push('/login'); return }

    const { data, error: err } = await supabase
      .from('games')
      .insert({
        title,
        opponent:     opponent.trim(),
        home_team:    homeTeam,
        away_team:    awayTeam,
        home_away:    homeAway,
        date,
        venue_id:     venueId || null,
        video_url:    videoId ? `https://www.youtube.com/embed/${videoId}` : null,
        duration_sec: durationMin ? parseInt(durationMin) * 60 : null,
        status:       'raw',
        uploaded_by:  user.id,
      })
      .select()
      .single()

    if (err) { setError(err.message); setSaving(false); return }
    router.push(`/game/${data.id}`)
  }

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar backHref="/" />

      <div style={{ padding: '24px 20px', maxWidth: 560, margin: '0 auto' }}>
        <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 28, color: '#0f2972', letterSpacing: '0.02em', marginBottom: 4 }}>
          Spiel hinzufügen
        </div>
        <div style={{ fontSize: 12, color: '#8A8F9E', marginBottom: 24 }}>
          Neues Spiel anlegen. YouTube-Link kann später ergänzt werden.
        </div>

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Home / Away toggle */}
          <div>
            <label style={labelStyle}>ASN spielt</label>
            <div style={{ display: 'flex', gap: 0, borderRadius: 8, overflow: 'hidden', border: '1px solid #E4E6EE' }}>
              {(['home', 'away'] as const).map(side => (
                <button
                  key={side}
                  type="button"
                  onClick={() => setHomeAway(side)}
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
            <input
              type="text"
              placeholder="FC Gegner"
              value={opponent}
              onChange={e => setOpponent(e.target.value)}
              required
              style={inputStyle}
            />
          </div>

          {/* Title preview */}
          {title && (
            <div style={{ fontSize: 13, color: '#0f2972', fontWeight: 600, padding: '8px 12px', background: '#e8edf8', borderRadius: 8 }}>
              {title}
            </div>
          )}

          {/* Date */}
          <div>
            <label style={labelStyle}>Datum</label>
            <input
              type="date"
              value={date}
              onChange={e => setDate(e.target.value)}
              required
              style={inputStyle}
            />
          </div>

          {/* Venue */}
          <div>
            <label style={labelStyle}>Spielstätte <span style={{ fontWeight: 400, color: '#8A8F9E' }}>(optional)</span></label>
            <select
              value={venueId}
              onChange={e => setVenueId(e.target.value)}
              style={{ ...inputStyle, background: '#F8F8F6' }}
            >
              <option value="">— Spielstätte wählen —</option>
              {venues.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
            </select>
          </div>

          {/* Status indicator — read only, always 'raw' on creation */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', background: '#F8F8F6', borderRadius: 8, border: '1px solid #E4E6EE' }}>
            <span style={{ fontSize: 12, color: '#8A8F9E' }}>Status</span>
            <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, background: '#f3f4f6', color: '#8A8F9E' }}>
              raw
            </span>
            <span style={{ fontSize: 11, color: '#8A8F9E', marginLeft: 4 }}>→ kalibriert → autopanned → hochgeladen</span>
          </div>

          {/* YouTube URL — optional */}
          <div>
            <label style={labelStyle}>YouTube URL <span style={{ fontWeight: 400, color: '#8A8F9E' }}>(optional)</span></label>
            <input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={e => setYoutubeUrl(e.target.value)}
              style={{
                ...inputStyle,
                border: `1px solid ${videoId ? '#22c55e' : '#E4E6EE'}`,
              }}
            />
            {youtubeUrl && !videoId && (
              <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>Keine gültige YouTube URL erkannt</div>
            )}
            {videoId && (
              <div style={{ fontSize: 11, color: '#22c55e', marginTop: 4 }}>✓ Video ID: {videoId}</div>
            )}
          </div>

          {/* Duration */}
          <div>
            <label style={labelStyle}>Spieldauer (Minuten, optional)</label>
            <input
              type="number"
              placeholder="90"
              value={durationMin}
              onChange={e => setDurationMin(e.target.value)}
              style={inputStyle}
            />
          </div>

          {/* YouTube preview */}
          {videoId && (
            <div style={{ borderRadius: 10, overflow: 'hidden', aspectRatio: '16/9', background: '#000' }}>
              <iframe
                src={`https://www.youtube.com/embed/${videoId}`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allowFullScreen
              />
            </div>
          )}

          {error && (
            <div style={{ fontSize: 12, color: '#ef4444', background: '#fef2f2', padding: '8px 12px', borderRadius: 8 }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              onClick={() => router.push('/')}
              style={{ flex: 1, padding: 11, background: '#fff', color: '#0f2972', border: '2px solid #0f2972', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={saving}
              style={{ flex: 1, padding: 11, background: saving ? '#E4E6EE' : '#0f2972', color: saving ? '#8A8F9E' : '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: saving ? 'default' : 'pointer', fontFamily: 'DM Sans, sans-serif' }}
            >
              {saving ? 'Speichern...' : 'Spiel speichern'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
