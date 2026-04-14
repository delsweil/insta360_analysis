'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { supabase, type Game, type Annotation } from '@/lib/supabase'

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

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface Props {
  params: Promise<{ id: string }>
}

export default function GamePage({ params }: Props) {
  const { id } = React.use(params)
  const streamRef = useRef<any>(null)
  const playerRef = useRef<HTMLDivElement>(null)

  const [game, setGame] = useState<Game | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [markIn, setMarkIn] = useState<number | null>(null)
  const [selectedLabel, setSelectedLabel] = useState('goal')
  const [note, setNote] = useState('')
  const [isCoach, setIsCoach] = useState(true) // TODO: from auth
  const [loading, setLoading] = useState(true)

  // Load game + annotations
  useEffect(() => {
    async function load() {
      const { data: gameData } = await supabase
        .from('games')
        .select('*')
        .eq('id', id)
        .single()

      if (gameData) {
        setGame(gameData)
        setDuration(gameData.duration_sec ?? 0)
      }

      const { data: annData } = await supabase
        .from('annotations')
        .select('*')
        .eq('game_id', id)
        .order('timestamp_sec')

      if (annData) setAnnotations(annData)
      setLoading(false)
    }
    load()
  }, [id])

  // Load Cloudflare Stream player SDK
  useEffect(() => {
    if (!game?.video_url) return
    const script = document.createElement('script')
    script.src = 'https://embed.cloudflarestream.com/embed/sdk.latest.js'
    script.onload = () => {
      if (playerRef.current && (window as any).Stream) {
        streamRef.current = (window as any).Stream(
          playerRef.current.querySelector('iframe')
        )
        streamRef.current.addEventListener('timeupdate', () => {
          setCurrentTime(streamRef.current.currentTime)
        })
        streamRef.current.addEventListener('durationchange', () => {
          setDuration(streamRef.current.duration)
        })
      }
    }
    document.head.appendChild(script)
    return () => { document.head.removeChild(script) }
  }, [game?.video_url])

  // Seek video when clicking timeline
  const seekTo = useCallback((sec: number) => {
    if (streamRef.current) {
      streamRef.current.currentTime = sec
      streamRef.current.play()
    }
  }, [])

  // Seek when clicking annotation
  const seekToAnnotation = useCallback((ann: Annotation) => {
    seekTo(ann.timestamp_sec)
  }, [seekTo])

  // Mark in
  const handleMarkIn = useCallback(() => {
    setMarkIn(currentTime)
  }, [currentTime])

  // Save annotation
  const handleSave = useCallback(async () => {
    if (markIn === null) return
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) return

    const ann: Partial<Annotation> = {
      game_id: id,
      user_id: user.id,
      timestamp_sec: markIn,
      end_timestamp_sec: currentTime > markIn + 1 ? currentTime : undefined,
      label: selectedLabel,
      note: note.trim() || undefined,
      is_public: true,
    }

    const { data } = await supabase
      .from('annotations')
      .insert(ann)
      .select()
      .single()

    if (data) {
      setAnnotations(prev =>
        [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
      )
    }
    setMarkIn(null)
    setNote('')
  }, [markIn, currentTime, selectedLabel, note, id])

  // Delete annotation
  const handleDelete = useCallback(async (annId: string) => {
    await supabase.from('annotations').delete().eq('id', annId)
    setAnnotations(prev => prev.filter(a => a.id !== annId))
  }, [])

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        background: '#F8F8F6',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'DM Sans, sans-serif',
        color: '#4A4F5C',
      }}>
        Loading...
      </div>
    )
  }

  if (!game) {
    return (
      <div style={{
        minHeight: '100vh',
        background: '#F8F8F6',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'DM Sans, sans-serif',
      }}>
        Game not found.
      </div>
    )
  }

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div style={{
      minHeight: '100vh',
      background: '#F8F8F6',
      fontFamily: 'DM Sans, sans-serif',
      color: '#111318',
    }}>
      {/* Top bar */}
      <div style={{
        background: '#0f2972',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif',
            fontSize: 20,
            color: '#fff',
            letterSpacing: '0.05em',
          }}>
            Pfeil Phönix
          </div>
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11 }}>
            Analysis
          </div>
        </div>
        {isCoach && (
          <div style={{
            background: '#E8780A',
            color: '#fff',
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
            padding: '3px 10px',
            borderRadius: 99,
          }}>
            Coach
          </div>
        )}
      </div>

      <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Game title */}
        <div style={{
          background: '#fff',
          border: '1px solid #E4E6EE',
          borderRadius: 12,
          padding: '10px 14px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div>
            <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 20, letterSpacing: '0.02em' }}>
              {game.title}
            </div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 2 }}>
              {new Date(game.date).toLocaleDateString('de-DE', { day: '2-digit', month: 'long', year: 'numeric' })}
              &nbsp;·&nbsp;
              <span style={{ color: '#E8780A', fontWeight: 600 }}>
                {annotations.length} annotations
              </span>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 10 }}>
          {/* Left: video + timeline */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Video player */}
            <div ref={playerRef} style={{
              background: '#091d52',
              borderRadius: 12,
              overflow: 'hidden',
              aspectRatio: '16/9',
              position: 'relative',
            }}>
              <iframe
                src={`${game.video_url}?preload=auto&loop=false&controls=true`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allow="accelerometer; gyroscope; autoplay; encrypted-media; picture-in-picture;"
                allowFullScreen
              />
            </div>

            {/* Timeline card */}
            <div style={{
              background: '#fff',
              border: '1px solid #E4E6EE',
              borderRadius: 12,
              padding: '12px 14px',
            }}>
              {/* Time display */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 8,
              }}>
                <span style={{
                  fontFamily: 'Bebas Neue, sans-serif',
                  fontSize: 16,
                  color: '#0f2972',
                  letterSpacing: '0.04em',
                }}>
                  {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                  {markIn !== null
                    ? `Mark in: ${formatTime(markIn)}`
                    : 'Click timeline or use mark in/out'}
                </span>
              </div>

              {/* Timeline track */}
              <div
                style={{
                  position: 'relative',
                  height: 24,
                  background: '#F8F8F6',
                  borderRadius: 4,
                  border: '1px solid #E4E6EE',
                  cursor: 'pointer',
                  marginBottom: 5,
                }}
                onClick={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect()
                  const pct = (e.clientX - rect.left) / rect.width
                  seekTo(pct * duration)
                }}
              >
                {/* Progress */}
                <div style={{
                  position: 'absolute',
                  left: 0, top: 0,
                  height: '100%',
                  width: `${progressPct}%`,
                  background: '#e8edf8',
                  borderRadius: '3px 0 0 3px',
                }} />
                {/* Playhead */}
                <div style={{
                  position: 'absolute',
                  top: -4, bottom: -4,
                  width: 2.5,
                  background: '#E8780A',
                  left: `${progressPct}%`,
                  borderRadius: 2,
                }} />
                {/* Mark in indicator */}
                {markIn !== null && duration > 0 && (
                  <div style={{
                    position: 'absolute',
                    top: -4, bottom: -4,
                    width: 2,
                    background: '#0f2972',
                    left: `${(markIn / duration) * 100}%`,
                    borderRadius: 2,
                  }} />
                )}
                {/* Annotation dots */}
                {annotations.map(ann => (
                  <div
                    key={ann.id}
                    onClick={(e) => { e.stopPropagation(); seekToAnnotation(ann) }}
                    title={`${formatTime(ann.timestamp_sec)} — ${ann.label}`}
                    style={{
                      position: 'absolute',
                      top: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: labelColor(ann.label),
                      border: '2px solid #fff',
                      left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                      cursor: 'pointer',
                      zIndex: 2,
                    }}
                  />
                ))}
              </div>

              {/* Time labels */}
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 10,
                color: '#8A8F9E',
                marginBottom: 10,
              }}>
                <span>0:00</span>
                <span>{formatTime(duration * 0.25)}</span>
                <span>{formatTime(duration * 0.5)}</span>
                <span>{formatTime(duration * 0.75)}</span>
                <span>{formatTime(duration)}</span>
              </div>

              {/* Label pills */}
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 8 }}>
                {LABELS.map(l => (
                  <button
                    key={l.key}
                    onClick={() => setSelectedLabel(l.key)}
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      padding: '3px 10px',
                      borderRadius: 99,
                      cursor: 'pointer',
                      border: `1.5px solid ${selectedLabel === l.key ? l.color : '#E4E6EE'}`,
                      background: selectedLabel === l.key
                        ? l.key === 'goal' ? '#fef2f2'
                          : l.key === 'shot' ? '#FEF0E0'
                          : l.key === 'chance' ? '#f0fdf4'
                          : l.key === 'tactical' ? '#e8edf8'
                          : '#F8F8F6'
                        : '#fff',
                      color: selectedLabel === l.key ? l.color : '#8A8F9E',
                    }}
                  >
                    {l.label}
                  </button>
                ))}
              </div>

              {/* Mark in/out + note */}
              <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                <button
                  onClick={handleMarkIn}
                  style={{
                    flex: 1,
                    fontSize: 11,
                    fontWeight: 600,
                    padding: '6px 8px',
                    borderRadius: 6,
                    border: `2px solid ${markIn !== null ? '#E8780A' : '#E4E6EE'}`,
                    background: markIn !== null ? '#FEF0E0' : '#F8F8F6',
                    color: markIn !== null ? '#E8780A' : '#8A8F9E',
                    cursor: 'pointer',
                  }}
                >
                  {markIn !== null ? `Mark in: ${formatTime(markIn)}` : 'Mark in'}
                </button>
                <button
                  onClick={handleSave}
                  disabled={markIn === null}
                  style={{
                    flex: 1,
                    fontSize: 11,
                    fontWeight: 600,
                    padding: '6px 8px',
                    borderRadius: 6,
                    border: 'none',
                    background: markIn !== null ? '#0f2972' : '#E4E6EE',
                    color: markIn !== null ? '#fff' : '#8A8F9E',
                    cursor: markIn !== null ? 'pointer' : 'default',
                  }}
                >
                  Save annotation
                </button>
              </div>

              {isCoach && (
                <input
                  type="text"
                  placeholder="Add a note (coaches only)..."
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  style={{
                    width: '100%',
                    fontSize: 12,
                    padding: '6px 10px',
                    border: '1px solid #E4E6EE',
                    borderRadius: 6,
                    outline: 'none',
                    fontFamily: 'DM Sans, sans-serif',
                    color: '#111318',
                    background: '#fff',
                  }}
                />
              )}

              {/* Legend */}
              <div style={{ display: 'flex', gap: 10, marginTop: 10, flexWrap: 'wrap' }}>
                {LABELS.map(l => (
                  <div key={l.key} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#8A8F9E' }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: l.color }} />
                    {l.label}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: annotations list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{
              background: '#fff',
              border: '1px solid #E4E6EE',
              borderRadius: 12,
              padding: '12px 14px',
            }}>
              <div style={{
                fontFamily: 'Bebas Neue, sans-serif',
                fontSize: 16,
                color: '#0f2972',
                letterSpacing: '0.04em',
                marginBottom: 10,
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}>
                Annotations
                <span style={{
                  fontFamily: 'DM Sans, sans-serif',
                  fontSize: 10,
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: 99,
                  background: '#e8edf8',
                  color: '#0f2972',
                }}>
                  {annotations.length}
                </span>
              </div>

              {annotations.length === 0 && (
                <div style={{ fontSize: 12, color: '#8A8F9E', textAlign: 'center', padding: '20px 0' }}>
                  No annotations yet. Mark moments while watching.
                </div>
              )}

              {annotations.map(ann => (
                <div
                  key={ann.id}
                  style={{
                    padding: '7px 0',
                    borderBottom: '1px solid #F8F8F6',
                    cursor: 'pointer',
                  }}
                  onClick={() => seekToAnnotation(ann)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                    <div style={{
                      width: 8, height: 8,
                      borderRadius: '50%',
                      background: labelColor(ann.label),
                      flexShrink: 0,
                    }} />
                    <span style={{
                      fontFamily: 'Bebas Neue, sans-serif',
                      fontSize: 14,
                      color: '#0f2972',
                      minWidth: 36,
                      letterSpacing: '0.04em',
                    }}>
                      {formatTime(ann.timestamp_sec)}
                    </span>
                    <span style={{ fontSize: 12, color: '#111318', flex: 1 }}>
                      {LABELS.find(l => l.key === ann.label)?.label ?? ann.label}
                    </span>
                    {isCoach && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(ann.id) }}
                        style={{
                          fontSize: 10,
                          color: '#8A8F9E',
                          background: 'none',
                          border: 'none',
                          cursor: 'pointer',
                          padding: '2px 4px',
                        }}
                      >
                        ✕
                      </button>
                    )}
                  </div>
                  {ann.end_timestamp_sec && (
                    <div style={{ fontSize: 10, color: '#8A8F9E', paddingLeft: 15, marginTop: 2 }}>
                      {formatTime(ann.timestamp_sec)} → {formatTime(ann.end_timestamp_sec)}
                    </div>
                  )}
                  {ann.note && (
                    <div style={{ fontSize: 10, color: '#4A4F5C', paddingLeft: 15, marginTop: 2, fontStyle: 'italic' }}>
                      {ann.note}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Season stats */}
            <div style={{
              background: '#0f2972',
              borderRadius: 12,
              padding: '12px 14px',
            }}>
              <div style={{
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                color: 'rgba(255,255,255,0.35)',
                marginBottom: 10,
                display: 'flex',
                alignItems: 'center',
                gap: 6,
              }}>
                <div style={{ width: 16, height: 2, background: '#E8780A', borderRadius: 2 }} />
                Season 2025/26
              </div>
              <div style={{ display: 'flex' }}>
                {[
                  { n: '—', l: 'Games' },
                  { n: annotations.length.toString(), l: 'This game', orange: true },
                  { n: '—', l: 'Total' },
                ].map((s, i) => (
                  <div key={i} style={{
                    flex: 1,
                    textAlign: 'center',
                    borderRight: i < 2 ? '1px solid rgba(255,255,255,0.1)' : 'none',
                    padding: '0 4px',
                  }}>
                    <div style={{
                      fontFamily: 'Bebas Neue, sans-serif',
                      fontSize: 22,
                      color: s.orange ? '#E8780A' : '#fff',
                      lineHeight: 1,
                    }}>
                      {s.n}
                    </div>
                    <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.35)', marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                      {s.l}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
