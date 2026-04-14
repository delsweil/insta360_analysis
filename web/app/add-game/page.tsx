'use client'

export const dynamic = 'force-dynamic'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'
import { useRouter } from 'next/navigation'

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

export default function AddGamePage() {
  const router = useRouter()
  const [title, setTitle] = useState('')
  const [date, setDate] = useState('')
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [durationMin, setDurationMin] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const videoId = extractYouTubeId(youtubeUrl)

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    if (!videoId) {
      setError('Please enter a valid YouTube URL')
      return
    }

    setSaving(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      router.push('/login')
      return
    }

    const embedUrl = `https://www.youtube.com/embed/${videoId}`
    const { data, error: err } = await supabase
      .from('games')
      .insert({
        title,
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
      {/* Topbar */}
      <div style={{
        background: '#0f2972',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 20,
          color: '#fff',
          letterSpacing: '0.05em',
          cursor: 'pointer',
        }} onClick={() => router.push('/')}>
          Pfeil Phönix · Spielanalyse
        </div>
      </div>

      <div style={{ padding: '24px 20px', maxWidth: 560, margin: '0 auto' }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 28,
          color: '#0f2972',
          letterSpacing: '0.02em',
          marginBottom: 4,
        }}>
          Spiel hinzufügen
        </div>
        <div style={{ fontSize: 12, color: '#8A8F9E', marginBottom: 24 }}>
          YouTube-Video als Unlisted hochladen, dann URL hier einfügen.
        </div>

        <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Title */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4 }}>
              Spieltitel
            </label>
            <input
              type="text"
              placeholder="z.B. ASN Pfeil Phönix vs. FC Grün-Weiß"
              value={title}
              onChange={e => setTitle(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '10px 14px',
                border: '1px solid #E4E6EE',
                borderRadius: 8,
                fontSize: 14,
                fontFamily: 'DM Sans, sans-serif',
                color: '#111318',
                outline: 'none',
                background: '#fff',
              }}
            />
          </div>

          {/* Date */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4 }}>
              Datum
            </label>
            <input
              type="date"
              value={date}
              onChange={e => setDate(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '10px 14px',
                border: '1px solid #E4E6EE',
                borderRadius: 8,
                fontSize: 14,
                fontFamily: 'DM Sans, sans-serif',
                color: '#111318',
                outline: 'none',
                background: '#fff',
              }}
            />
          </div>

          {/* YouTube URL */}
          <div>
            <label style={{ fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4 }}>
              YouTube URL
            </label>
            <input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={e => setYoutubeUrl(e.target.value)}
              required
              style={{
                width: '100%',
                padding: '10px 14px',
                border: `1px solid ${videoId ? '#22c55e' : '#E4E6EE'}`,
                borderRadius: 8,
                fontSize: 14,
                fontFamily: 'DM Sans, sans-serif',
                color: '#111318',
                outline: 'none',
                background: '#fff',
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
            <label style={{ fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4 }}>
              Spieldauer (Minuten, optional)
            </label>
            <input
              type="number"
              placeholder="90"
              value={durationMin}
              onChange={e => setDurationMin(e.target.value)}
              style={{
                width: '100%',
                padding: '10px 14px',
                border: '1px solid #E4E6EE',
                borderRadius: 8,
                fontSize: 14,
                fontFamily: 'DM Sans, sans-serif',
                color: '#111318',
                outline: 'none',
                background: '#fff',
              }}
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
              style={{
                flex: 1,
                padding: '11px',
                background: '#fff',
                color: '#0f2972',
                border: '2px solid #0f2972',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 600,
                cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              Abbrechen
            </button>
            <button
              type="submit"
              disabled={saving || !videoId}
              style={{
                flex: 1,
                padding: '11px',
                background: saving || !videoId ? '#E4E6EE' : '#0f2972',
                color: saving || !videoId ? '#8A8F9E' : '#fff',
                border: 'none',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 600,
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
