'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { supabase, type Game, type Annotation } from '@/lib/supabase'
import ShareModal from '@/components/ShareModal'
import Topbar from '@/components/Topbar'

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
  const playerRef = useRef<HTMLIFrameElement>(null)

  const [game, setGame] = useState<Game | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [selectedLabel, setSelectedLabel] = useState('goal')
  const [note, setNote] = useState('')
  const [isCoach, setIsCoach] = useState(false)
  const [userRole, setUserRole] = useState<'admin' | 'coach' | 'player'>( 'player')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [showAnnotations, setShowAnnotations] = useState(false)
  const [showShare, setShowShare] = useState(false)

  // Detect mobile
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768)
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  // Load game + user + annotations
  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()

      // Check if coach (you can extend this with a roles table later)
    const { data: roleData } = await supabase
      .from('user_roles')
      .select('role')
      .eq('user_id', user?.id ?? '')
      .single()

    const role = roleData?.role ?? 'player'
    setIsCoach(role === 'coach' || role === 'admin')
    setUserRole(role as 'admin' | 'coach' | 'player')

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

    // Realtime subscription — live annotations from other users
    const channel = supabase
      .channel(`game-${id}`)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'annotations',
          filter: `game_id=eq.${id}`,
        },
        (payload) => {
          const newAnn = payload.new as Annotation
          setAnnotations(prev => {
            // Avoid duplicates (our own saves come through here too)
            if (prev.find(a => a.id === newAnn.id)) return prev
            return [...prev, newAnn].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
          })
        }
      )
      .on(
        'postgres_changes',
        {
          event: 'DELETE',
          schema: 'public',
          table: 'annotations',
          filter: `game_id=eq.${id}`,
        },
        (payload) => {
          setAnnotations(prev => prev.filter(a => a.id !== payload.old.id))
        }
      )
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [id])

  // YouTube postMessage time tracking
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

  const seekTo = useCallback((sec: number) => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [sec, true] }), '*'
    )
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*'
    )
  }, [])

  // Single-tap save (mobile) or mark+save (desktop coach)
  const handleQuickSave = useCallback(async () => {
    if (saving) return
    setSaving(true)

    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { setSaving(false); return }

    const { data } = await supabase
      .from('annotations')
      .insert({
        game_id: id,
        user_id: user.id,
        timestamp_sec: currentTime,
        label: selectedLabel,
        note: note.trim() || null,
        is_public: true,
      })
      .select()
      .single()

    if (data) {
      setAnnotations(prev =>
        [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
      )
      setSaved(true)
      setTimeout(() => setSaved(false), 1500)
    }
    setNote('')
    setSaving(false)
  }, [saving, currentTime, selectedLabel, note, id])

  // Range save (desktop coach only)
  const [markIn, setMarkIn] = useState<number | null>(null)

  const handleRangeSave = useCallback(async () => {
    if (saving || markIn === null) return
    setSaving(true)

    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { setSaving(false); return }

    const { data } = await supabase
      .from('annotations')
      .insert({
        game_id: id,
        user_id: user.id,
        timestamp_sec: markIn,
        end_timestamp_sec: currentTime > markIn + 1 ? currentTime : null,
        label: selectedLabel,
        note: note.trim() || null,
        is_public: true,
      })
      .select()
      .single()

    if (data) {
      setAnnotations(prev =>
        [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
      )
    }
    setMarkIn(null)
    setNote('')
    setSaving(false)
  }, [saving, markIn, currentTime, selectedLabel, note, id])

  const handleDelete = useCallback(async (annId: string) => {
    await supabase.from('annotations').delete().eq('id', annId)
    setAnnotations(prev => prev.filter(a => a.id !== annId))
  }, [])

  if (loading) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C',
    }}>
      Loading...
    </div>
  )

  if (!game) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif',
    }}>
      Game not found.
    </div>
  )

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0

  // ── MOBILE LAYOUT ──────────────────────────────────────────────
  if (isMobile) {
    return (
      <div style={{
        minHeight: '100vh', background: '#F8F8F6',
        fontFamily: 'DM Sans, sans-serif', color: '#111318',
        display: 'flex', flexDirection: 'column',
      }}>
        <Topbar role={userRole} />

        {/* Game title */}
        <div style={{ padding: '10px 14px 0' }}>
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif',
            fontSize: 17, color: '#0f2972', letterSpacing: '0.02em',
            lineHeight: 1.1,
          }}>
            {game.title}
          </div>
          <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 1 }}>
            {new Date(game.date).toLocaleDateString('de-DE', {
              day: '2-digit', month: 'long', year: 'numeric'
            })}
          </div>
        </div>

        {showAnnotations ? (
          // ── Annotations view ──
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 14px' }}>
            {annotations.length === 0 ? (
              <div style={{
                textAlign: 'center', color: '#8A8F9E',
                fontSize: 14, padding: '40px 0',
              }}>
                No annotations yet.
              </div>
            ) : annotations.map(ann => (
              <div
                key={ann.id}
                onClick={() => {
                  seekTo(ann.timestamp_sec)
                  setShowAnnotations(false)
                }}
                style={{
                  background: '#fff',
                  border: '1px solid #E4E6EE',
                  borderRadius: 10,
                  padding: '12px 14px',
                  marginBottom: 8,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                }}
              >
                <div style={{
                  width: 10, height: 10, borderRadius: '50%',
                  background: labelColor(ann.label), flexShrink: 0,
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{
                      fontFamily: 'Bebas Neue, sans-serif',
                      fontSize: 15, color: '#0f2972', letterSpacing: '0.04em',
                    }}>
                      {formatTime(ann.timestamp_sec)}
                    </span>
                    <span style={{ fontSize: 13, color: '#111318' }}>
                      {LABELS.find(l => l.key === ann.label)?.label ?? ann.label}
                    </span>
                  </div>
                  {ann.note && (
                    <div style={{ fontSize: 11, color: '#4A4F5C', marginTop: 2, fontStyle: 'italic' }}>
                      {ann.note}
                    </div>
                  )}
                </div>
                {isCoach && (
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDelete(ann.id) }}
                    style={{
                      fontSize: 14, color: '#8A8F9E',
                      background: 'none', border: 'none',
                      cursor: 'pointer', padding: '4px 6px',
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>
        ) : (
          // ── Video + annotation controls ──
          <>
            {/* Video */}
            <div style={{
              background: '#091d52',
              aspectRatio: '16/9',
              flexShrink: 0,
            }}>
              <iframe
                ref={playerRef}
                src={`${game.video_url}?enablejsapi=1`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
            </div>

            {/* Timeline */}
            <div style={{ padding: '10px 14px 6px', flexShrink: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{
                  fontFamily: 'Bebas Neue, sans-serif',
                  fontSize: 14, color: '#0f2972', letterSpacing: '0.04em',
                }}>
                  {formatTime(currentTime)}
                </span>
                <span style={{ fontSize: 11, color: '#8A8F9E' }}>
                  {formatTime(duration)}
                </span>
              </div>
              <div
                style={{
                  position: 'relative', height: 20,
                  background: '#F8F8F6', borderRadius: 4,
                  border: '1px solid #E4E6EE', cursor: 'pointer',
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
                  position: 'absolute', top: -3, bottom: -3,
                  width: 3, background: '#E8780A',
                  left: `${progressPct}%`, borderRadius: 2,
                }} />
                {annotations.map(ann => (
                  <div
                    key={ann.id}
                    style={{
                      position: 'absolute', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: 12, height: 12, borderRadius: '50%',
                      background: labelColor(ann.label),
                      border: '2px solid #fff',
                      left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                      zIndex: 2,
                    }}
                  />
                ))}
              </div>
            </div>

            {/* Label selector */}
            <div style={{
              padding: '0 14px 8px',
              display: 'flex', gap: 6, overflowX: 'auto',
              flexShrink: 0,
            }}>
              {LABELS.map(l => (
                <button
                  key={l.key}
                  onClick={() => setSelectedLabel(l.key)}
                  style={{
                    fontSize: 12, fontWeight: 600,
                    padding: '6px 14px', borderRadius: 99,
                    border: `2px solid ${selectedLabel === l.key ? l.color : '#E4E6EE'}`,
                    background: selectedLabel === l.key
                      ? l.key === 'goal' ? '#fef2f2'
                        : l.key === 'shot' ? '#FEF0E0'
                        : l.key === 'chance' ? '#f0fdf4'
                        : l.key === 'tactical' ? '#e8edf8'
                        : '#F8F8F6'
                      : '#fff',
                    color: selectedLabel === l.key ? l.color : '#8A8F9E',
                    cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: 0,
                  }}
                >
                  {l.label}
                </button>
              ))}
            </div>

            {/* Coach note (coaches only) */}
            {isCoach && (
              <div style={{ padding: '0 14px 8px', flexShrink: 0 }}>
                <input
                  type="text"
                  placeholder="Add a tactical note..."
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  style={{
                    width: '100%', fontSize: 13,
                    padding: '8px 12px',
                    border: '1px solid #E4E6EE', borderRadius: 8,
                    outline: 'none', fontFamily: 'DM Sans, sans-serif',
                    color: '#111318', background: '#fff',
                    boxSizing: 'border-box',
                  }}
                />
              </div>
            )}

            {/* Mark in / Mark out / Save */}
            <div style={{ padding: '0 14px 20px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {markIn === null ? (
                // Step 1: Mark in
                <button
                  onClick={() => setMarkIn(currentTime)}
                  style={{
                    width: '100%', padding: '16px',
                    fontSize: 16, fontWeight: 700,
                    fontFamily: 'DM Sans, sans-serif',
                    borderRadius: 12, border: 'none',
                    background: '#0f2972', color: '#fff',
                    cursor: 'pointer', letterSpacing: '0.02em',
                  }}
                >
                  {`Mark in — ${formatTime(currentTime)}`}
                </button>
              ) : (
                // Step 2: Mark out or save now
                <>
                  <div style={{
                    display: 'flex', alignItems: 'center',
                    justifyContent: 'space-between',
                    background: '#FEF0E0', borderRadius: 10,
                    padding: '10px 14px',
                  }}>
                    <span style={{ fontSize: 13, color: '#E8780A', fontWeight: 600 }}>
                      In: {formatTime(markIn)}
                    </span>
                    <button
                      onClick={() => setMarkIn(null)}
                      style={{
                        fontSize: 11, color: '#E8780A',
                        background: 'none', border: 'none',
                        cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                      }}
                    >
                      Reset
                    </button>
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={handleRangeSave}
                      disabled={saving}
                      style={{
                        flex: 1, padding: '14px',
                        fontSize: 14, fontWeight: 700,
                        fontFamily: 'DM Sans, sans-serif',
                        borderRadius: 12, border: 'none',
                        background: saved ? '#22c55e' : saving ? '#E4E6EE' : '#E8780A',
                        color: saving ? '#8A8F9E' : '#fff',
                        cursor: saving ? 'default' : 'pointer',
                      }}
                    >
                      {saved ? '✓ Saved' : saving ? 'Saving...' : `Save now`}
                    </button>
                    <button
                      onClick={() => {
                        // Mark out = save with current time as end
                        handleRangeSave()
                      }}
                      disabled={saving || currentTime <= markIn}
                      style={{
                        flex: 1, padding: '14px',
                        fontSize: 14, fontWeight: 700,
                        fontFamily: 'DM Sans, sans-serif',
                        borderRadius: 12, border: 'none',
                        background: currentTime > markIn ? '#0f2972' : '#E4E6EE',
                        color: currentTime > markIn ? '#fff' : '#8A8F9E',
                        cursor: currentTime > markIn ? 'pointer' : 'default',
                      }}
                    >
                      {`Mark out — ${formatTime(currentTime)}`}
                    </button>
                  </div>
                </>
              )}
            </div>
          </>
        )}
      </div>
    )
  }

  // ── DESKTOP LAYOUT ─────────────────────────────────────────────
  return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      fontFamily: 'DM Sans, sans-serif', color: '#111318',
    }}>
      {/* Top bar */}
      <div style={{
        background: '#0f2972',
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
        padding: '8px 20px',
      }}>
        <div style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: 20, color: '#fff', letterSpacing: '0.05em',
          cursor: 'pointer',
        }} onClick={() => window.location.href = '/'}>
          ASN Pfeil Phönix · Spielanalyse
        </div>
        {isCoach && (
          <div style={{
            background: '#E8780A', color: '#fff',
            fontSize: 10, fontWeight: 700,
            letterSpacing: '0.1em', textTransform: 'uppercase',
            padding: '3px 10px', borderRadius: 99,
          }}>
            Coach
          </div>
        )}
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
              {game.title}
            </div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 2 }}>
              {new Date(game.date).toLocaleDateString('de-DE', {
                day: '2-digit', month: 'long', year: 'numeric'
              })}
              &nbsp;·&nbsp;
              <span style={{ color: '#E8780A', fontWeight: 600 }}>
                {annotations.length} annotations
              </span>
              &nbsp;·&nbsp;
              <span style={{ color: '#22c55e', fontWeight: 600 }}>
                ● live
              </span>
            </div>
          </div>
          {isCoach && (
            <button
              onClick={() => setShowShare(true)}
              style={{
                fontSize: 12, fontWeight: 600,
                padding: '7px 16px', borderRadius: 8,
                border: 'none', background: '#E8780A',
                color: '#fff', cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif',
                flexShrink: 0,
              }}
            >
              Share highlights
            </button>
          )}
        </div>

        {showShare && (
          <ShareModal
            gameId={id}
            annotations={annotations}
            onClose={() => setShowShare(false)}
          />
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: 10 }}>
          {/* Left */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Video */}
            <div style={{
              background: '#091d52', borderRadius: 12,
              overflow: 'hidden', aspectRatio: '16/9',
            }}>
              <iframe
                ref={playerRef}
                src={`${game.video_url}?enablejsapi=1&origin=${typeof window !== 'undefined' ? window.location.origin : ''}`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
            </div>

            {/* Timeline card */}
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
                <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                  {isCoach
                    ? markIn !== null ? `Mark in: ${formatTime(markIn)}` : 'Mark in/out for range'
                    : 'Tap a label then mark'}
                </span>
              </div>

              {/* Track */}
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
                {markIn !== null && duration > 0 && (
                  <div style={{
                    position: 'absolute', top: -4, bottom: -4,
                    width: 2, background: '#0f2972',
                    left: `${(markIn / duration) * 100}%`, borderRadius: 2,
                  }} />
                )}
                {annotations.map(ann => (
                  <div
                    key={ann.id}
                    onClick={(e) => { e.stopPropagation(); seekTo(ann.timestamp_sec) }}
                    title={`${formatTime(ann.timestamp_sec)} — ${ann.label}`}
                    style={{
                      position: 'absolute', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: 10, height: 10, borderRadius: '50%',
                      background: labelColor(ann.label),
                      border: '2px solid #fff',
                      left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                      cursor: 'pointer', zIndex: 2,
                    }}
                  />
                ))}
              </div>

              <div style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: 10, color: '#8A8F9E', marginBottom: 10,
              }}>
                <span>0:00</span>
                <span>{formatTime(duration * 0.25)}</span>
                <span>{formatTime(duration * 0.5)}</span>
                <span>{formatTime(duration * 0.75)}</span>
                <span>{formatTime(duration)}</span>
              </div>

              {/* Labels */}
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 8 }}>
                {LABELS.map(l => (
                  <button
                    key={l.key}
                    onClick={() => setSelectedLabel(l.key)}
                    style={{
                      fontSize: 11, fontWeight: 600,
                      padding: '3px 10px', borderRadius: 99, cursor: 'pointer',
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

              {/* Controls */}
              {isCoach ? (
                <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                  <button
                    onClick={() => setMarkIn(currentTime)}
                    style={{
                      flex: 1, fontSize: 11, fontWeight: 600,
                      padding: '6px 8px', borderRadius: 6,
                      border: `2px solid ${markIn !== null ? '#E8780A' : '#E4E6EE'}`,
                      background: markIn !== null ? '#FEF0E0' : '#F8F8F6',
                      color: markIn !== null ? '#E8780A' : '#8A8F9E',
                      cursor: 'pointer',
                    }}
                  >
                    {markIn !== null ? `Mark in: ${formatTime(markIn)}` : 'Mark in'}
                  </button>
                  <button
                    onClick={handleRangeSave}
                    disabled={markIn === null || saving}
                    style={{
                      flex: 1, fontSize: 11, fontWeight: 600,
                      padding: '6px 8px', borderRadius: 6, border: 'none',
                      background: markIn !== null ? '#0f2972' : '#E4E6EE',
                      color: markIn !== null ? '#fff' : '#8A8F9E',
                      cursor: markIn !== null ? 'pointer' : 'default',
                    }}
                  >
                    Save annotation
                  </button>
                </div>
              ) : (
                <button
                  onClick={handleQuickSave}
                  disabled={saving}
                  style={{
                    width: '100%', fontSize: 12, fontWeight: 600,
                    padding: '8px', borderRadius: 6, border: 'none',
                    background: saved ? '#22c55e' : saving ? '#E4E6EE' : '#0f2972',
                    color: saving ? '#8A8F9E' : '#fff',
                    cursor: saving ? 'default' : 'pointer',
                    marginBottom: 8, transition: 'background 0.2s',
                  }}
                >
                  {saved
                    ? `✓ Saved at ${formatTime(currentTime)}`
                    : `Mark ${LABELS.find(l => l.key === selectedLabel)?.label} at ${formatTime(currentTime)}`
                  }
                </button>
              )}

              {isCoach && (
                <input
                  type="text"
                  placeholder="Add a tactical note..."
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  style={{
                    width: '100%', fontSize: 12, padding: '6px 10px',
                    border: '1px solid #E4E6EE', borderRadius: 6,
                    outline: 'none', fontFamily: 'DM Sans, sans-serif',
                    color: '#111318', background: '#fff',
                  }}
                />
              )}

              {/* Legend */}
              <div style={{ display: 'flex', gap: 10, marginTop: 10, flexWrap: 'wrap' }}>
                {LABELS.map(l => (
                  <div key={l.key} style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    fontSize: 10, color: '#8A8F9E',
                  }}>
                    <div style={{ width: 8, height: 8, borderRadius: '50%', background: l.color }} />
                    {l.label}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right: annotations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{
              background: '#fff', border: '1px solid #E4E6EE',
              borderRadius: 12, padding: '12px 14px',
            }}>
              <div style={{
                fontFamily: 'Bebas Neue, sans-serif',
                fontSize: 16, color: '#0f2972', letterSpacing: '0.04em',
                marginBottom: 10,
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              }}>
                Annotations
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
                  No annotations yet.
                </div>
              )}

              {annotations.map(ann => (
                <div
                  key={ann.id}
                  style={{
                    padding: '7px 0', borderBottom: '1px solid #F8F8F6',
                    cursor: 'pointer',
                  }}
                  onClick={() => seekTo(ann.timestamp_sec)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
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
                    {isCoach && (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(ann.id) }}
                        style={{
                          fontSize: 10, color: '#8A8F9E',
                          background: 'none', border: 'none',
                          cursor: 'pointer', padding: '2px 4px',
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
            <div style={{ background: '#0f2972', borderRadius: 12, padding: '12px 14px' }}>
              <div style={{
                fontSize: 9, fontWeight: 700, letterSpacing: '0.14em',
                textTransform: 'uppercase', color: 'rgba(255,255,255,0.35)',
                marginBottom: 10,
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <div style={{ width: 16, height: 2, background: '#E8780A', borderRadius: 2 }} />
                Season 2025/26
              </div>
              <div style={{ display: 'flex' }}>
                {[
                  { n: '—', l: 'Games' },
                  { n: String(annotations.length), l: 'This game', orange: true },
                  { n: '—', l: 'Total' },
                ].map((s, i) => (
                  <div key={i} style={{
                    flex: 1, textAlign: 'center',
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
                    <div style={{
                      fontSize: 9, color: 'rgba(255,255,255,0.35)',
                      marginTop: 3, textTransform: 'uppercase', letterSpacing: '0.08em',
                    }}>
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
