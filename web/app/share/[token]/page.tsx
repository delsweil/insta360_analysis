'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { supabase } from '@/lib/supabase'
import type { Annotation } from '@/lib/supabase'

const LABELS = [
  { key: 'goal',     label: 'Goal',     color: '#ef4444' },
  { key: 'shot',     label: 'Shot',     color: '#E8780A' },
  { key: 'chance',   label: 'Chance',   color: '#22c55e' },
  { key: 'tactical', label: 'Tactical', color: '#0f2972' },
  { key: 'other',    label: 'Other',    color: '#8A8F9E' },
]

function labelColor(label: string) {
  return LABELS.find(l => l.key === label)?.color ?? '#8A8F9E'
}

const PRE_ROLL_SEC = 5  // seconds before annotation to start playback

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface Props {
  params: Promise<{ token: string }>
}

export default function SharePage({ params }: Props) {
  const { token } = React.use(params)
  const playerRef = useRef<HTMLIFrameElement>(null)

  const [title, setTitle] = useState('')
  const [videoUrl, setVideoUrl] = useState('')
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [currentIdx, setCurrentIdx] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const autoPlayRef = useRef(autoPlay)
  autoPlayRef.current = autoPlay

  useEffect(() => {
    async function load() {
      // Load share link
      const { data: link } = await supabase
        .from('share_links')
        .select('*')
        .eq('token', token)
        .single()

      if (!link) {
        setError('This share link does not exist or has expired.')
        setLoading(false)
        return
      }

      // Check expiry
      if (link.expires_at && new Date(link.expires_at) < new Date()) {
        setError('This share link has expired.')
        setLoading(false)
        return
      }

      // Load game
      const { data: game } = await supabase
        .from('games')
        .select('title, video_url, duration_sec')
        .eq('id', link.game_id)
        .single()

      if (!game) {
        setError('Game not found.')
        setLoading(false)
        return
      }

      setTitle(game.title)
      setVideoUrl(game.video_url)
      if (game.duration_sec) setDuration(game.duration_sec)

      // Load selected annotations
      let query = supabase
        .from('annotations')
        .select('*')
        .eq('game_id', link.game_id)
        .order('timestamp_sec')

      if (link.annotation_ids && link.annotation_ids.length > 0) {
        query = query.in('id', link.annotation_ids)
      }

      const { data: anns } = await query
      if (anns) setAnnotations(anns)
      setLoading(false)
    }
    load()
  }, [token])

  // YouTube postMessage tracking
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      try {
        const data = typeof e.data === 'string' ? JSON.parse(e.data) : e.data
        if (data?.event === 'infoDelivery' && data?.info) {
          if (data.info.currentTime !== undefined) setCurrentTime(data.info.currentTime)
          if (data.info.duration !== undefined && data.info.duration > 0) setDuration(data.info.duration)
        }
      } catch {}
    }
    window.addEventListener('message', handler)
    return () => window.removeEventListener('message', handler)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => {
      playerRef.current?.contentWindow?.postMessage(
        JSON.stringify({ event: 'listening' }), '*'
      )
    }, 500)
    return () => clearInterval(interval)
  }, [])

  // Auto-advance: when current annotation's end time is reached, jump to next
  useEffect(() => {
    if (!autoPlay || annotations.length === 0) return
    const ann = annotations[currentIdx]
    if (!ann) return

    const endTime = ann.end_timestamp_sec ?? (ann.timestamp_sec + 10)
    if (currentTime >= endTime) {
      const nextIdx = currentIdx + 1
      if (nextIdx < annotations.length) {
        setCurrentIdx(nextIdx)
        seekTo(annotations[nextIdx].timestamp_sec)
      } else {
        // End of highlights
        setAutoPlay(false)
      }
    }
  }, [currentTime, autoPlay, currentIdx, annotations])

  const seekTo = useCallback((sec: number) => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [sec, true] }), '*'
    )
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*'
    )
  }, [])

  const handleSelectAnnotation = (idx: number) => {
    setCurrentIdx(idx)
    seekTo(Math.max(0, annotations[idx].timestamp_sec - PRE_ROLL_SEC))
    if (!autoPlay) setAutoPlay(false)
  }

  const handlePlayAll = () => {
    setCurrentIdx(0)
    seekTo(Math.max(0, annotations[0].timestamp_sec - PRE_ROLL_SEC))
    setAutoPlay(true)
  }

  if (loading) return (
    <div style={{
      minHeight: '100vh', background: '#0f2972',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif', color: 'rgba(255,255,255,0.6)',
    }}>
      Loading highlights...
    </div>
  )

  if (error) return (
    <div style={{
      minHeight: '100vh', background: '#0f2972',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      <div style={{
        background: '#fff', borderRadius: 16,
        padding: '32px 40px', textAlign: 'center',
        maxWidth: 400,
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 24, color: '#0f2972', marginBottom: 12,
        }}>
          ASN Pfeil Phönix
        </div>
        <div style={{ fontSize: 14, color: '#4A4F5C' }}>{error}</div>
      </div>
    </div>
  )

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      fontFamily: 'DM Sans, sans-serif', color: '#111318',
    }}>
      {/* Topbar */}
      <div style={{
        background: '#0f2972',
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/asn-logo.svg"
            alt="ASN Pfeil Phönix"
            width={32} height={32}
            style={{ filter: 'brightness(0) invert(1)' }}
          />
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 18, color: '#fff', letterSpacing: '0.05em', lineHeight: 1,
            }}>
              ASN Pfeil Phönix
            </div>
            <div style={{
              fontSize: 9, color: 'rgba(255,255,255,0.4)',
              letterSpacing: '0.12em', textTransform: 'uppercase',
            }}>
              Highlights · Nürnberg
            </div>
          </div>
        </div>
        <div style={{
          background: '#E8780A', color: '#fff',
          fontSize: 10, fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          padding: '3px 10px', borderRadius: 99,
        }}>
          Highlights
        </div>
      </div>

      <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Game title */}
        <div style={{
          background: '#fff', border: '1px solid #E4E6EE',
          borderRadius: 12, padding: '10px 14px',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 20, letterSpacing: '0.02em',
            }}>
              {title}
            </div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 2 }}>
              <span style={{ color: '#E8780A', fontWeight: 600 }}>
                {annotations.length} highlights
              </span>
            </div>
          </div>
          {annotations.length > 0 && (
            <button
              onClick={handlePlayAll}
              style={{
                fontSize: 12, fontWeight: 700,
                padding: '8px 18px', borderRadius: 99,
                border: 'none',
                background: autoPlay ? '#22c55e' : '#0f2972',
                color: '#fff', cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif',
                transition: 'background 0.2s',
              }}
            >
              {autoPlay ? '▶ Playing all...' : '▶ Play all'}
            </button>
          )}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 10 }}>
          {/* Left: video + timeline */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{
              background: '#091d52', borderRadius: 12,
              overflow: 'hidden', aspectRatio: '16/9',
            }}>
              {videoUrl && (
                <iframe
                  ref={playerRef}
                  src={`${videoUrl}?enablejsapi=1&origin=${typeof window !== 'undefined' ? window.location.origin : ''}`}
                  style={{ width: '100%', height: '100%', border: 'none' }}
                  allowFullScreen
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                />
              )}
            </div>

            {/* Timeline */}
            <div style={{
              background: '#fff', border: '1px solid #E4E6EE',
              borderRadius: 12, padding: '12px 14px',
            }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: 8,
              }}>
                <span style={{
                  fontFamily: 'Bebas Neue, sans-serif',
                  fontSize: 16, color: '#0f2972', letterSpacing: '0.04em',
                }}>
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                {autoPlay && (
                  <button
                    onClick={() => setAutoPlay(false)}
                    style={{
                      fontSize: 11, color: '#8A8F9E',
                      background: 'none', border: '1px solid #E4E6EE',
                      borderRadius: 99, padding: '3px 10px',
                      cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                    }}
                  >
                    Stop auto-play
                  </button>
                )}
              </div>

              <div
                style={{
                  position: 'relative', height: 24,
                  background: '#F8F8F6', borderRadius: 4,
                  border: '1px solid #E4E6EE', cursor: 'pointer', marginBottom: 5,
                }}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect()
                  seekTo(((e.clientX - rect.left) / rect.width) * duration)
                }}
              >
                <div style={{
                  position: 'absolute', left: 0, top: 0,
                  height: '100%', width: `${progressPct}%`,
                  background: '#e8edf8', borderRadius: '3px 0 0 3px',
                }} />
                <div style={{
                  position: 'absolute', top: -4, bottom: -4,
                  width: 2.5, background: '#E8780A',
                  left: `${progressPct}%`, borderRadius: 2,
                }} />
                {annotations.map((ann, idx) => (
                  <div
                    key={ann.id}
                    onClick={(e) => { e.stopPropagation(); handleSelectAnnotation(idx) }}
                    style={{
                      position: 'absolute', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: idx === currentIdx ? 14 : 10,
                      height: idx === currentIdx ? 14 : 10,
                      borderRadius: '50%',
                      background: labelColor(ann.label),
                      border: `2px solid ${idx === currentIdx ? '#fff' : '#fff'}`,
                      boxShadow: idx === currentIdx ? `0 0 0 2px ${labelColor(ann.label)}` : 'none',
                      left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                      cursor: 'pointer', zIndex: 2,
                      transition: 'all 0.15s',
                    }}
                  />
                ))}
              </div>

              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: 10, color: '#8A8F9E',
              }}>
                <span>0:00</span>
                <span>{formatTime(duration * 0.25)}</span>
                <span>{formatTime(duration * 0.5)}</span>
                <span>{formatTime(duration * 0.75)}</span>
                <span>{formatTime(duration)}</span>
              </div>
            </div>
          </div>

          {/* Right: highlights list */}
          <div style={{
            background: '#fff', border: '1px solid #E4E6EE',
            borderRadius: 12, padding: '12px 14px',
            overflowY: 'auto', maxHeight: 600,
          }}>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 16, color: '#0f2972', letterSpacing: '0.04em',
              marginBottom: 10,
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              Highlights
              <span style={{
                fontFamily: 'DM Sans, sans-serif',
                fontSize: 10, fontWeight: 700,
                padding: '2px 8px', borderRadius: 99,
                background: '#e8edf8', color: '#0f2972',
              }}>
                {annotations.length}
              </span>
            </div>

            {annotations.length === 0 && (
              <div style={{
                fontSize: 12, color: '#8A8F9E',
                textAlign: 'center', padding: '20px 0',
              }}>
                No highlights in this share link.
              </div>
            )}

            {annotations.map((ann, idx) => (
              <div
                key={ann.id}
                onClick={() => handleSelectAnnotation(idx)}
                style={{
                  padding: '8px 10px',
                  borderRadius: 8,
                  marginBottom: 4,
                  cursor: 'pointer',
                  background: idx === currentIdx ? '#e8edf8' : 'transparent',
                  border: `1px solid ${idx === currentIdx ? '#0f2972' : 'transparent'}`,
                  transition: 'all 0.15s',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%',
                    background: labelColor(ann.label), flexShrink: 0,
                  }} />
                  <span style={{
                    fontFamily: 'Bebas Neue, sans-serif',
                    fontSize: 14, color: '#0f2972',
                    minWidth: 36, letterSpacing: '0.04em',
                  }}>
                    {formatTime(ann.timestamp_sec)}
                  </span>
                  <span style={{ fontSize: 12, color: '#111318', flex: 1 }}>
                    {LABELS.find(l => l.key === ann.label)?.label ?? ann.label}
                  </span>
                  {ann.end_timestamp_sec && (
                    <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                      {formatTime(ann.end_timestamp_sec - ann.timestamp_sec)}s
                    </span>
                  )}
                </div>
                {ann.note && (
                  <div style={{
                    fontSize: 10, color: '#4A4F5C',
                    paddingLeft: 16, marginTop: 2, fontStyle: 'italic',
                  }}>
                    {ann.note}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
