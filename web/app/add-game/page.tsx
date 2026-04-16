'use client'

export const dynamic = 'force-dynamic'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'
import Topbar from '@/components/Topbar'

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

const inputStyle = {
  width: '100%',
  padding: '10px 14px',
  border: '1px solid #E4E6EE',
  borderRadius: 8,
  fontSize: 14,
  fontFamily: 'DM Sans, sans-serif',
  color: '#111318',
  outline: 'none',
  background: '#fff',
  boxSizing: 'border-box' as const,
}

const labelStyle = {
  fontSize: 12,
  fontWeight: 600 as const,
  color: '#4A4F5C',
  display: 'block' as const,
  marginBottom: 4,
}

export default function AddGamePage() {
  const router = useRouter()
  const [homeTeam, setHomeTeam] = useState('ASN Pfeil Phönix')
  const [awayTeam, setAwayTeam] = useState('')
  const [date, setDate] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [durationMin, setDurationMin] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const videoId = extractYouTubeId(youtubeUrl)
  const title = homeTeam && awayTeam ? `${homeTeam} vs. ${awayTeam}` : ''

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!videoId) {
      setError('Please enter a valid YouTube URL')
      return
    }
    if (!homeTeam.trim() || !awayTeam.trim()) {
      setError('Please enter both team names')
      return
    }

    setSaving(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { router.push('/login'); return }

    const embedUrl = `https://www.youtube.com/embed/${videoId}`
    const { data, error: err } = await supabase
      .from('games')
      .insert({
        title,
        home_team: homeTeam.trim(),
        away_team: awayTeam.trim(),
        date,
        video_url: embedUrl,
        duration_sec: durationMin ? parseInt(durationMin) * 60 : null,
        uploaded_by: user.id,
      })
      .select()
      .single()

    if (err) {
      setError(err.message)
      setSaving(false)
      return
    }

    router.push(`/game/${data.id}`)
  }

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar backHref="/" />

      <div style={{ padding: '24px 20px', maxWidth: 560, margin: '0 auto' }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 28, color: '#0f2972',
          letterSpacing: '0.02em', marginBottom: 4,
        }}>
          Spiel hinzufügen
        </div>
        <div style={{ fontSize: 12, color: '#8A8F9E', marginBottom: 24 }}>
          YouTube-Video als Unlisted hochladen, dann URL hier einfügen.
        </div>

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

          {/* Teams */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div>
              <label style={labelStyle}>Heimteam</label>
              <input
                type="text"
                placeholder="ASN Pfeil Phönix"
                value={homeTeam}
                onChange={e => setHomeTeam(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
            <div>
              <label style={labelStyle}>Auswärtsteam</label>
              <input
                type="text"
                placeholder="FC Gegner"
                value={awayTeam}
                onChange={e => setAwayTeam(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
          </div>

          {/* Auto-generated title preview */}
          {title && (
            <div style={{
              fontSize: 13, color: '#0f2972', fontWeight: 600,
              padding: '8px 12px', background: '#e8edf8',
              borderRadius: 8,
            }}>
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

          {/* YouTube URL */}
          <div>
            <label style={labelStyle}>YouTube URL</label>
            <input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={e => setYoutubeUrl(e.target.value)}
              required
              style={{
                ...inputStyle,
                border: `1px solid ${videoId ? '#22c55e' : '#E4E6EE'}`,
              }}
            />
            {youtubeUrl && !videoId && (
              <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>
                Keine gültige YouTube URL erkannt
              </div>
            )}
            {videoId && (
              <div style={{ fontSize: 11, color: '#22c55e', marginTop: 4 }}>
                ✓ Video ID: {videoId}
              </div>
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
            <div style={{
              fontSize: 12, color: '#ef4444',
              background: '#fef2f2', padding: '8px 12px', borderRadius: 8,
            }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              onClick={() => router.push('/')}
              style={{
                flex: 1, padding: '11px',
                background: '#fff', color: '#0f2972',
                border: '2px solid #0f2972', borderRadius: 8,
                fontSize: 14, fontWeight: 600,
                cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
              }}
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={saving || !videoId}
              style={{
                flex: 1, padding: '11px',
                background: saving || !videoId ? '#E4E6EE' : '#0f2972',
                color: saving || !videoId ? '#8A8F9E' : '#fff',
                border: 'none', borderRadius: 8,
                fontSize: 14, fontWeight: 600,
                cursor: saving || !videoId ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              {saving ? 'Speichern...' : 'Spiel speichern'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
