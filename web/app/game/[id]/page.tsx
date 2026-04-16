'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { supabase, type Game, type Annotation, LABEL_DEFS, labelColor, labelDisplay } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import AnnotationShape from '@/components/AnnotationShape'
import ShareModal from '@/components/ShareModal'
import AnnotationFilter, { type FilterState, ALL_FILTERS, passesFilter } from '@/components/AnnotationFilter'
import ClipRecorder from '@/components/ClipRecorder'

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
  const [selectedLabel, setSelectedLabel] = useState('goal_home')
  const [note, setNote] = useState('')
  const [isCoach, setIsCoach] = useState(false)
  const [userRole, setUserRole] = useState<'admin' | 'coach' | 'player'>('player')
  const [userId, setUserId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [showAnnotations, setShowAnnotations] = useState(false)
  const [showShare, setShowShare] = useState(false)
  const [markIn, setMarkIn] = useState<number | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [filter, setFilter] = useState<FilterState>(ALL_FILTERS)

  // Filtered annotations
  const filteredAnnotations = annotations.filter(a => passesFilter(a.label, filter))

  // Count per type for filter bar
  const filterCounts = {
    goal: annotations.filter(a => a.label.startsWith('goal')).length,
    shot: annotations.filter(a => a.label.startsWith('shot')).length,
    setpiece: annotations.filter(a => a.label.startsWith('setpiece')).length,
    tactical: annotations.filter(a => a.label === 'tactical').length,
  }

  // Detect mobile once on mount — don't switch on rotation
  // to prevent YouTube iframe from reloading
  useEffect(() => {
    // Use screen width (physical device width) not window width
    // This doesn't change when rotating
    const w = screen.width < screen.height ? screen.width : screen.height
    setIsMobile(w < 768)
  }, [])

  // Load game + user + annotations
  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()

      if (user) {
        setUserId(user.id)
        const { data: roleData } = await supabase
          .from('user_roles')
          .select('role')
          .eq('user_id', user?.id ?? '')
          .single()
        const role = roleData?.role ?? 'player'
        setUserRole(role as 'admin' | 'coach' | 'player')
        setIsCoach(role === 'coach' || role === 'admin')
      }

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
        .select('*, profiles(display_name)')
        .eq('game_id', id)
        .order('timestamp_sec')

      if (annData) setAnnotations(annData)
      setLoading(false)
    }
    load()

    // Realtime subscription
    const channel = supabase
      .channel(`game-${id}`)
      .on('postgres_changes', {
        event: 'INSERT', schema: 'public',
        table: 'annotations', filter: `game_id=eq.${id}`,
      }, (payload) => {
        const newAnn = payload.new as Annotation
        setAnnotations(prev => {
          if (prev.find(a => a.id === newAnn.id)) return prev
          return [...prev, newAnn].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
        })
      })
      .on('postgres_changes', {
        event: 'DELETE', schema: 'public',
        table: 'annotations', filter: `game_id=eq.${id}`,
      }, (payload) => {
        setAnnotations(prev => prev.filter(a => a.id !== payload.old.id))
      })
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [id])

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

  const seekTo = useCallback((sec: number) => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [sec, true] }), '*'
    )
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*'
    )
  }, [])

  const handleQuickSave = useCallback(async () => {
    if (saving) return
    setSaving(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { setSaving(false); return }

    const { data } = await supabase
      .from('annotations')
      .insert({
        game_id: id, user_id: user.id,
        timestamp_sec: currentTime,
        label: selectedLabel,
        note: note.trim() || null,
        is_public: true,
      })
      .select('*, profiles(display_name)')
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

  const handleRangeSave = useCallback(async () => {
    if (saving || markIn === null) return
    setSaving(true)
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) { setSaving(false); return }

    const { data } = await supabase
      .from('annotations')
      .insert({
        game_id: id, user_id: user.id,
        timestamp_sec: markIn,
        end_timestamp_sec: currentTime > markIn + 1 ? currentTime : null,
        label: selectedLabel,
        note: note.trim() || null,
        is_public: true,
      })
      .select('*, profiles(display_name)')
      .single()

    if (data) {
      setAnnotations(prev =>
        [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
      )
      setSaved(true)
      setTimeout(() => setSaved(false), 1500)
    }
    setMarkIn(null)
    setNote('')
    setSaving(false)
  }, [saving, markIn, currentTime, selectedLabel, note, id])

  const handleDelete = useCallback(async (annId: string) => {
    await supabase.from('annotations').delete().eq('id', annId)
    setAnnotations(prev => prev.filter(a => a.id !== annId))
  }, [])

  // Label picker — available labels based on role
  const availableLabels = LABEL_DEFS.filter(l => isCoach || !l.coachOnly)

  const LabelPicker = ({ size = 'normal' }: { size?: 'normal' | 'large' }) => (
    <div style={{
      display: 'flex', gap: 6, flexWrap: 'wrap',
      overflowX: size === 'large' ? 'auto' : undefined,
    }}>
      {LABEL_DEFS.map(l => {
        const isSelected = selectedLabel === l.key
        const isDisabled = l.coachOnly && !isCoach
        const color = labelColor(l.key)
        return (
          <button
            key={l.key}
            onClick={() => !isDisabled && setSelectedLabel(l.key)}
            disabled={isDisabled}
            title={isDisabled ? 'Coaches only' : undefined}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              fontSize: size === 'large' ? 13 : 11,
              fontWeight: 600,
              padding: size === 'large' ? '7px 14px' : '4px 10px',
              borderRadius: 99,
              cursor: isDisabled ? 'default' : 'pointer',
              border: `1.5px solid ${isSelected ? color : '#E4E6EE'}`,
              background: isSelected ? `${color}18` : isDisabled ? '#F8F8F6' : '#fff',
              color: isDisabled ? '#C0C4CE' : isSelected ? color : '#8A8F9E',
              whiteSpace: 'nowrap', flexShrink: 0,
              opacity: isDisabled ? 0.5 : 1,
            }}
          >
            <AnnotationShape labelKey={l.key} size={size === 'large' ? 12 : 10} />
            {l.display}
            {l.team !== 'none' && (
              <span style={{
                fontSize: size === 'large' ? 10 : 9,
                opacity: 0.7,
                fontWeight: 400,
              }}>
                {l.team === 'home' ? game?.home_team ?? 'Home' : game?.away_team ?? 'Away'}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )

  // Timeline with shapes
  const Timeline = ({ height = 24, mobileSize = false }: { height?: number, mobileSize?: boolean }) => (
    <div
      style={{
        position: 'relative', height,
        background: '#F8F8F6', borderRadius: 4,
        border: '1px solid #E4E6EE', cursor: 'pointer',
      }}
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        seekTo(((e.clientX - rect.left) / rect.width) * duration)
      }}
    >
      {/* Progress fill */}
      <div style={{
        position: 'absolute', left: 0, top: 0,
        height: '100%',
        width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
        background: '#e8edf8', borderRadius: '3px 0 0 3px',
      }} />
      {/* Playhead */}
      <div style={{
        position: 'absolute', top: -4, bottom: -4, width: 2.5,
        background: '#E8780A',
        left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
        borderRadius: 2,
      }} />
      {/* Mark in indicator */}
      {markIn !== null && duration > 0 && (
        <div style={{
          position: 'absolute', top: -4, bottom: -4, width: 2,
          background: '#0f2972',
          left: `${(markIn / duration) * 100}%`, borderRadius: 2,
        }} />
      )}
      {/* Annotation shapes */}
      {annotations.map(ann => {
        const visible = passesFilter(ann.label, filter)
        return (
          <div
            key={ann.id}
            onClick={(e) => { e.stopPropagation(); seekTo(ann.timestamp_sec) }}
            style={{
              position: 'absolute', top: '50%',
              transform: 'translate(-50%, -50%)',
              left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
              cursor: 'pointer', zIndex: 2,
              opacity: visible ? 1 : 0.2,
              transition: 'opacity 0.2s',
            }}
          >
            <AnnotationShape
              labelKey={ann.label}
              size={mobileSize ? 14 : 11}
            />
          </div>
        )
      })}
    </div>
  )

  // Annotation row
  const AnnRow = ({ ann, compact = false }: { ann: Annotation, compact?: boolean }) => (
    <div
      style={{
        padding: compact ? '8px 10px' : '7px 0',
        borderBottom: compact ? 'none' : '1px solid #F8F8F6',
        borderRadius: compact ? 8 : 0,
        marginBottom: compact ? 8 : 0,
        background: compact ? '#fff' : 'transparent',
        border: compact ? '1px solid #E4E6EE' : undefined,
        cursor: 'pointer',
      }}
      onClick={() => {
        seekTo(ann.timestamp_sec)
        if (isMobile) setShowAnnotations(false)
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        <AnnotationShape labelKey={ann.label} size={compact ? 12 : 10} />
        <span style={{
          fontFamily: 'Bebas Neue, sans-serif',
          fontSize: compact ? 15 : 14,
          color: '#0f2972', minWidth: 36, letterSpacing: '0.04em',
        }}>
          {formatTime(ann.timestamp_sec)}
        </span>
        <span style={{ fontSize: compact ? 13 : 12, color: '#111318', flex: 1 }}>
          {labelDisplay(ann.label, game?.home_team, game?.away_team)}
        </span>
        {ann.profiles?.display_name && (
          <span style={{ fontSize: 10, color: '#8A8F9E', flexShrink: 0 }}>
            {ann.profiles.display_name}
          </span>
        )}
        {(isCoach || ann.user_id === userId) && (
          <button
            onClick={(e) => { e.stopPropagation(); handleDelete(ann.id) }}
            style={{
              fontSize: 12, color: '#8A8F9E',
              background: 'none', border: 'none',
              cursor: 'pointer', padding: '2px 4px', flexShrink: 0,
            }}
          >
            ✕
          </button>
        )}
      </div>
      {ann.end_timestamp_sec && (
        <div style={{ fontSize: 10, color: '#8A8F9E', paddingLeft: 17, marginTop: 2 }}>
          {formatTime(ann.timestamp_sec)} → {formatTime(ann.end_timestamp_sec)}
        </div>
      )}
      {ann.note && (
        <div style={{ fontSize: 10, color: '#4A4F5C', paddingLeft: 17, marginTop: 2, fontStyle: 'italic' }}>
          {ann.note}
        </div>
      )}
    </div>
  )

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
            fontSize: 17, color: '#0f2972', letterSpacing: '0.02em', lineHeight: 1.1,
          }}>
            {game.title}
          </div>
          <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 1 }}>
            {new Date(game.date).toLocaleDateString('de-DE', {
              day: '2-digit', month: 'long', year: 'numeric'
            })}
            &nbsp;·&nbsp;
            <span style={{ color: '#E8780A', fontWeight: 600 }}>{annotations.length}</span>
            &nbsp;annotations&nbsp;·&nbsp;
            <span style={{ color: '#22c55e', fontWeight: 600 }}>● live</span>
          </div>
        </div>

        {showAnnotations ? (
          // Annotations list view
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 14px' }}>
            <button
              onClick={() => setShowAnnotations(false)}
              style={{
                marginBottom: 10, fontSize: 12, fontWeight: 600,
                color: '#0f2972', background: 'none', border: 'none',
                cursor: 'pointer', padding: 0,
              }}
            >
              ← Back to video
            </button>
            {annotations.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#8A8F9E', fontSize: 14, padding: '40px 0' }}>
                No annotations yet.
              </div>
            ) : filteredAnnotations.map(ann => (
              <AnnRow key={ann.id} ann={ann} compact />
            ))}
          </div>
        ) : (
          <>
            {/* Video */}
            <div style={{ background: '#091d52', aspectRatio: '16/9', flexShrink: 0 }}>
              <iframe
                ref={playerRef}
                src={`${game.video_url}?enablejsapi=1`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
            </div>

            {/* Filter bar - mobile */}
            <div style={{ padding: '6px 14px 0', flexShrink: 0, overflowX: 'auto' }}>
              <AnnotationFilter
                filter={filter}
                onChange={setFilter}
                counts={filterCounts}
              />
            </div>

            {/* Timeline */}
            <div style={{ padding: '6px 14px 4px', flexShrink: 0 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{
                  fontFamily: 'Bebas Neue, sans-serif',
                  fontSize: 14, color: '#0f2972', letterSpacing: '0.04em',
                }}>
                  {formatTime(currentTime)}
                </span>
                <button
                  onClick={() => setShowAnnotations(true)}
                  style={{
                    fontSize: 11, fontWeight: 600, color: '#0f2972',
                    background: '#e8edf8', border: 'none', borderRadius: 99,
                    padding: '3px 10px', cursor: 'pointer',
                  }}
                >
                  Annotations ({annotations.length})
                </button>
              </div>
              <Timeline height={20} mobileSize />
            </div>

            {/* Label picker */}
            <div style={{ padding: '6px 14px', flexShrink: 0, overflowX: 'auto' }}>
              <LabelPicker size="large" />
            </div>

            {/* Coach note */}
            {isCoach && (
              <div style={{ padding: '4px 14px', flexShrink: 0 }}>
                <input
                  type="text"
                  placeholder="Tactical note..."
                  value={note}
                  onChange={e => setNote(e.target.value)}
                  style={{
                    width: '100%', fontSize: 13, padding: '8px 12px',
                    border: '1px solid #E4E6EE', borderRadius: 8,
                    outline: 'none', fontFamily: 'DM Sans, sans-serif',
                    color: '#111318', background: '#fff', boxSizing: 'border-box',
                  }}
                />
              </div>
            )}

            {/* Clip recorder */}
            {!isRecording ? (
              <div style={{ padding: '8px 14px 20px', flexShrink: 0 }}>
                <button
                  onClick={() => {
                    setMarkIn(currentTime)
                    setIsRecording(true)
                    setSaved(false)
                  }}
                  style={{
                    width: '100%', padding: '16px',
                    fontSize: 15, fontWeight: 700,
                    fontFamily: 'DM Sans, sans-serif',
                    borderRadius: 12, border: 'none',
                    background: '#0f2972', color: '#fff', cursor: 'pointer',
                  }}
                >
                  Start clip — {formatTime(currentTime)}
                </button>
              </div>
            ) : (
              <ClipRecorder
                currentTime={currentTime}
                saving={saving}
                saved={saved}
                onSave={async (startTime, endTime) => {
                  setSaving(true)
                  const { data: { user } } = await supabase.auth.getUser()
                  if (!user) { setSaving(false); return }
                  const { data } = await supabase
                    .from('annotations')
                    .insert({
                      game_id: id, user_id: user.id,
                      timestamp_sec: startTime,
                      end_timestamp_sec: endTime,
                      label: selectedLabel,
                      note: note.trim() || null,
                      is_public: true,
                    })
                    .select('*, profiles(display_name)')
                    .single()
                  if (data) {
                    setAnnotations(prev =>
                      [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
                    )
                    setSaved(true)
                    setTimeout(() => {
                      setSaved(false)
                      setIsRecording(false)
                      setMarkIn(null)
                    }, 1500)
                  }
                  setSaving(false)
                }}
                onCancel={() => {
                  setIsRecording(false)
                  setMarkIn(null)
                  setSaved(false)
                }}
              />
            )}
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
      <Topbar role={userRole} backHref="/" />

      <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Game title bar */}
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
              <span style={{ color: '#22c55e', fontWeight: 600 }}>● live</span>
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
                fontFamily: 'DM Sans, sans-serif', flexShrink: 0,
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
            initialFilter={filter}
            onClose={() => setShowShare(false)}
          />
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 10 }}>
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
              {/* Time + hint */}
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
                    : 'Pick a label then mark'}
                </span>
              </div>

              {/* Timeline */}
              <div style={{ marginBottom: 5 }}>
                <Timeline height={26} />
              </div>

              {/* Time labels */}
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

              {/* Filter bar */}
              <div style={{ marginBottom: 8 }}>
                <AnnotationFilter
                  filter={filter}
                  onChange={setFilter}
                  counts={filterCounts}
                />
              </div>

              {/* Label picker */}
              <div style={{ marginBottom: 10 }}>
                <LabelPicker />
              </div>

              {/* Mark in/out (coach) or quick save (player) */}
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
                    : `Mark ${labelDisplay(selectedLabel, game.home_team, game.away_team)} at ${formatTime(currentTime)}`
                  }
                </button>
              )}

              {/* Coach note */}
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
              <div style={{
                display: 'flex', gap: 8, marginTop: 10,
                flexWrap: 'wrap',
              }}>
                <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8,
                    background: '#0f2972', borderRadius: 1, marginRight: 3,
                  }} />
                  {game.home_team}
                </span>
                <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                  <span style={{
                    display: 'inline-block', width: 8, height: 8,
                    background: '#E8780A', borderRadius: 1, marginRight: 3,
                  }} />
                  {game.away_team}
                </span>
              </div>
            </div>
          </div>

          {/* Right: annotations */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
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

              {filteredAnnotations.map(ann => (
                <AnnRow key={ann.id} ann={ann} />
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
                      fontFamily: 'Bebas Neue, sans-serif', fontSize: 22,
                      color: s.orange ? '#E8780A' : '#fff', lineHeight: 1,
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
