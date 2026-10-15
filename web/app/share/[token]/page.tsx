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

function labelDisplay(label: string) {
  return LABELS.find(l => l.key === label)?.label ?? label
}

const PRE_ROLL_SEC = 5  // seconds before annotation to start playback

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

const hudTransportBtn: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
  width: 44, height: 44,
  background: 'rgba(255,255,255,0.12)', border: '1px solid rgba(255,255,255,0.2)',
  borderRadius: 10, color: '#fff', cursor: 'pointer',
  justifyContent: 'center',
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
  const [isMobile, setIsMobile] = useState(false)
  const [hudVisible, setHudVisible] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [isPlaying, setIsPlaying] = useState(true)
  const [mobilePlaybackRate, setMobilePlaybackRate] = useState(1)
  const [focusMode, setFocusMode] = useState(false)
  const autoPlayRef = useRef(autoPlay)
  autoPlayRef.current = autoPlay

  useEffect(() => {
    const check = () => setIsMobile(Math.min(window.screen.width, window.screen.height) < 768)
    check()
  }, [])

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
          if (data.info.playerState !== undefined) {
            // 1 = playing, 2 = paused
            setIsPlaying(data.info.playerState === 1)
          }
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

  const scrubTo = useCallback((sec: number) => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [sec, true] }), '*'
    )
  }, [])

  const pauseVideo = useCallback(() => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'pauseVideo', args: [] }), '*'
    )
  }, [])

  const resumeVideo = useCallback(() => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*'
    )
  }, [])

  const setVideoRate = useCallback((rate: number) => {
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'setPlaybackRate', args: [rate] }), '*'
    )
  }, [])

  const handleSelectAnnotation = (idx: number) => {
    setCurrentIdx(idx)
    seekTo(Math.max(0, annotations[idx].timestamp_sec - PRE_ROLL_SEC))
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

  // ── MOBILE FULLSCREEN HUD (read-only) ──────────────────────────────────
  if (isMobile) {
    const SPEED_STEPS = [0.5, 1, 1.25, 1.5, 2]

    const cycleSpeed = () => {
      const idx = SPEED_STEPS.indexOf(mobilePlaybackRate)
      const next = SPEED_STEPS[(idx + 1) % SPEED_STEPS.length]
      setMobilePlaybackRate(next)
      setVideoRate(next)
    }

    return (
      <div style={{
        position: 'fixed', inset: 0,
        background: '#000',
        fontFamily: 'DM Sans, sans-serif',
        overflow: 'hidden',
      }}>
        {/* ── VIDEO fills entire screen ── */}
        {videoUrl && (
          <iframe
            ref={playerRef}
            src={`${videoUrl}?enablejsapi=1&controls=0&rel=0&modestbranding=1&autoplay=1&mute=1&playsinline=1`}
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', border: 'none' }}
            allowFullScreen
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          />
        )}

        {/* Invisible tap-catcher — sits above the iframe (zIndex 7).
            Receives taps on the video area and toggles HUD visibility. */}
        <div
          onClick={() => setHudVisible(v => !v)}
          style={{ position: 'absolute', inset: 0, zIndex: 7, background: 'transparent' }}
        />

        {/* ── HUD wrapper — shown/hidden by tap ── */}
        <div
          onClick={() => setHudVisible(false)}
          style={{
            position: 'absolute', inset: 0,
            opacity: hudVisible ? 1 : 0,
            transition: 'opacity 0.2s ease',
            pointerEvents: hudVisible ? 'auto' : 'none',
            zIndex: 8,
          }}
        >
          {/* Top gradient */}
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, height: '35%',
            background: 'linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, transparent 100%)',
            pointerEvents: 'none', zIndex: 6,
          }} />
          {/* Bottom gradient */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: '50%',
            background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%)',
            pointerEvents: 'none', zIndex: 6,
          }} />

          {/* Top bar */}
          <div onClick={e => e.stopPropagation()} style={{
            position: 'absolute', top: 0, left: 0, right: 0,
            zIndex: 10, display: 'flex', alignItems: 'center',
            justifyContent: 'space-between',
            padding: '10px 14px 0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {/* Unmute button — prominent until user unmutes */}
              {isMuted && (
                <button
                  onClick={() => {
                    playerRef.current?.contentWindow?.postMessage(
                      JSON.stringify({ event: 'command', func: 'unMute', args: [] }), '*'
                    )
                    setIsMuted(false)
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    background: '#E8780A', border: 'none',
                    borderRadius: 8, color: '#fff',
                    fontSize: 12, fontWeight: 700, padding: '5px 10px', cursor: 'pointer',
                  }}
                >
                  🔇 Ton an
                </button>
              )}
            </div>

            {/* Title */}
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif', fontSize: 14,
              color: 'rgba(255,255,255,0.9)', letterSpacing: '0.08em',
              textShadow: '0 1px 4px rgba(0,0,0,0.5)',
              textAlign: 'center', flex: 1, padding: '0 8px',
            }}>
              {title}
            </div>

            {/* Highlights badge */}
            <div style={{
              background: 'rgba(232,120,10,0.85)',
              color: '#fff', fontSize: 10, fontWeight: 700,
              letterSpacing: '0.08em', textTransform: 'uppercase',
              padding: '3px 10px', borderRadius: 99,
              flexShrink: 0,
            }}>
              {annotations.length} Clips
            </div>
          </div>

          {/* Transport controls — centred vertically */}
          <div onClick={e => e.stopPropagation()} style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%, -50%)',
            zIndex: 10, display: 'flex', alignItems: 'center', gap: 14,
          }}>
            {([-30, -10] as const).map(delta => (
              <button
                key={delta}
                onClick={() => scrubTo(Math.max(0, currentTime + delta))}
                style={hudTransportBtn}
              >
                {delta === -30
                  ? <svg width="18" height="18" viewBox="0 0 18 18" fill="white"><polygon points="15,3 6,9 15,15"/><rect x="2" y="3" width="2.5" height="12" rx="1" fill="white"/></svg>
                  : <svg width="18" height="18" viewBox="0 0 18 18" fill="white"><polygon points="14,3 5,9 14,15"/></svg>
                }
                <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.04em', color: 'rgba(255,255,255,0.6)' }}>
                  {delta}s
                </span>
              </button>
            ))}

            {/* Play/Pause */}
            <button
              onClick={() => isPlaying ? pauseVideo() : resumeVideo()}
              style={{
                ...hudTransportBtn,
                width: 56, height: 56,
                background: 'rgba(255,255,255,0.2)',
                borderColor: 'rgba(255,255,255,0.4)',
              }}
            >
              {isPlaying
                ? <svg width="22" height="22" viewBox="0 0 22 22" fill="white"><rect x="4" y="3" width="4.5" height="16" rx="1.5"/><rect x="13.5" y="3" width="4.5" height="16" rx="1.5"/></svg>
                : <svg width="22" height="22" viewBox="0 0 22 22" fill="white"><polygon points="5,3 19,11 5,19"/></svg>
              }
            </button>

            {([10, 30] as const).map(delta => (
              <button
                key={delta}
                onClick={() => scrubTo(Math.min(duration || 9999, currentTime + delta))}
                style={hudTransportBtn}
              >
                {delta === 10
                  ? <svg width="18" height="18" viewBox="0 0 18 18" fill="white"><polygon points="4,3 13,9 4,15"/></svg>
                  : <svg width="18" height="18" viewBox="0 0 18 18" fill="white"><polygon points="3,3 12,9 3,15"/><rect x="13.5" y="3" width="2.5" height="12" rx="1" fill="white"/></svg>
                }
                <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '0.04em', color: 'rgba(255,255,255,0.6)' }}>
                  +{delta}s
                </span>
              </button>
            ))}
          </div>

          {/* ── Bottom HUD ── */}
          <div onClick={e => e.stopPropagation()} style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            zIndex: 10, padding: '0 12px 10px',
          }}>
            {/* Timeline row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{
                fontFamily: 'Bebas Neue, sans-serif', fontSize: 13,
                color: 'rgba(255,255,255,0.85)', letterSpacing: '0.06em',
                flexShrink: 0, minWidth: 38,
              }}>
                {formatTime(currentTime)}
              </span>

              {/* Scrub bar */}
              <div
                style={{
                  flex: 1, height: 32, position: 'relative',
                  cursor: 'pointer', touchAction: 'none',
                }}
                onPointerDown={e => {
                  e.currentTarget.setPointerCapture(e.pointerId)
                  const r = e.currentTarget.getBoundingClientRect()
                  scrubTo(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (duration || 1))
                }}
                onPointerMove={e => {
                  if (e.buttons !== 1) return
                  const r = e.currentTarget.getBoundingClientRect()
                  scrubTo(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (duration || 1))
                }}
              >
                {/* Track */}
                <div style={{
                  position: 'absolute', left: 0, right: 0,
                  top: '50%', height: 5, transform: 'translateY(-50%)',
                  background: 'rgba(255,255,255,0.2)', borderRadius: 3,
                }}>
                  {/* Progress fill */}
                  <div style={{
                    position: 'absolute', left: 0, top: 0, bottom: 0,
                    width: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                    background: '#E8780A', borderRadius: 3,
                  }} />
                </div>
                {/* Playhead */}
                <div style={{
                  position: 'absolute', top: '50%',
                  transform: 'translate(-50%, -50%)',
                  left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                  width: 14, height: 14, borderRadius: '50%',
                  background: '#fff',
                  boxShadow: '0 0 0 2px #E8780A',
                }} />
                {/* Annotation markers (coloured dots) */}
                {annotations.map((ann, idx) => (
                  <div
                    key={ann.id}
                    onClick={e => { e.stopPropagation(); handleSelectAnnotation(idx) }}
                    style={{
                      position: 'absolute', top: '50%',
                      transform: 'translate(-50%, -50%)',
                      left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                      width: idx === currentIdx ? 12 : 9,
                      height: idx === currentIdx ? 12 : 9,
                      borderRadius: '50%',
                      background: labelColor(ann.label),
                      border: '2px solid rgba(0,0,0,0.4)',
                      boxShadow: idx === currentIdx ? `0 0 0 2px ${labelColor(ann.label)}` : 'none',
                      cursor: 'pointer', zIndex: 2,
                    }}
                  />
                ))}
              </div>

              <span style={{
                fontFamily: 'Bebas Neue, sans-serif', fontSize: 13,
                color: 'rgba(255,255,255,0.5)', letterSpacing: '0.06em',
                flexShrink: 0,
              }}>
                {formatTime(duration)}
              </span>
            </div>

            {/* Action row — read-only: no mark/save, just clips + play-all + speed */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              {/* Play all */}
              <button
                onClick={() => {
                  if (autoPlay) {
                    setAutoPlay(false)
                  } else if (annotations.length > 0) {
                    handlePlayAll()
                  }
                }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '0 14px', height: 38,
                  background: autoPlay ? '#22c55e' : '#0f2972',
                  border: 'none', borderRadius: 9,
                  color: '#fff', fontSize: 13, fontWeight: 700,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer',
                  flexShrink: 0,
                }}
              >
                {autoPlay ? '⏹ Stop' : '▶ Play all'}
              </button>

              {/* Clip list toggle */}
              <button
                onClick={() => setFocusMode(prev => !prev)}
                style={{
                  marginLeft: 'auto',
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '0 11px', height: 38,
                  background: focusMode ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: 9, color: 'rgba(255,255,255,0.75)',
                  fontSize: 12, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                {focusMode ? '✕ Clips' : `≡ ${annotations.length} Clips`}
              </button>

              {/* Playback speed */}
              <button
                onClick={cycleSpeed}
                style={{
                  padding: '0 11px', height: 38,
                  background: mobilePlaybackRate !== 1 ? 'rgba(232,120,10,0.15)' : 'rgba(255,255,255,0.08)',
                  border: `1px solid ${mobilePlaybackRate !== 1 ? 'rgba(232,120,10,0.4)' : 'rgba(255,255,255,0.15)'}`,
                  borderRadius: 9,
                  color: mobilePlaybackRate !== 1 ? '#E8780A' : 'rgba(255,255,255,0.75)',
                  fontSize: 12, fontWeight: 700,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                {mobilePlaybackRate === 1 ? '1×' : `${mobilePlaybackRate}×`}
              </button>
            </div>
          </div>

          {/* ── Clip list panel (slides in from right) ── */}
          {focusMode && (
            <div style={{
              position: 'absolute', top: 0, right: 0, bottom: 0,
              width: '40%', maxWidth: 220,
              background: 'rgba(6,10,22,0.88)', backdropFilter: 'blur(16px)',
              borderLeft: '1px solid rgba(255,255,255,0.08)',
              zIndex: 15, display: 'flex', flexDirection: 'column',
            }}>
              <div style={{
                padding: '10px 12px 8px',
                borderBottom: '1px solid rgba(255,255,255,0.07)',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              }}>
                <span style={{
                  fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
                  letterSpacing: '0.1em', color: 'rgba(255,255,255,0.35)',
                }}>{annotations.length} Clips</span>
                <button
                  onClick={() => setFocusMode(false)}
                  style={{
                    background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)',
                    fontSize: 16, cursor: 'pointer', padding: '0 2px', lineHeight: 1,
                  }}
                >✕</button>
              </div>
              <div style={{ overflowY: 'auto', flex: 1 }}>
                {annotations.length === 0 ? (
                  <div style={{ padding: '20px 12px', fontSize: 12, color: 'rgba(255,255,255,0.3)', textAlign: 'center' }}>
                    No highlights
                  </div>
                ) : annotations.map((ann, idx) => {
                  const color = labelColor(ann.label)
                  const isActive = idx === currentIdx
                  return (
                    <div
                      key={ann.id}
                      onClick={() => { handleSelectAnnotation(idx); setFocusMode(false) }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 8,
                        padding: '8px 12px',
                        borderBottom: '1px solid rgba(255,255,255,0.04)',
                        cursor: 'pointer',
                        background: isActive ? 'rgba(37,99,235,0.15)' : 'transparent',
                      }}
                    >
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                      <div style={{
                        fontFamily: 'Bebas Neue, sans-serif', fontSize: 14,
                        color: isActive ? '#93bbff' : 'rgba(255,255,255,0.75)',
                        letterSpacing: '0.06em', flexShrink: 0, minWidth: 34,
                      }}>
                        {formatTime(ann.timestamp_sec)}
                      </div>
                      <div style={{
                        fontSize: 10, fontWeight: 600,
                        color: isActive ? '#93bbff' : 'rgba(255,255,255,0.5)',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {labelDisplay(ann.label)}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

        </div>{/* end HUD wrapper */}
      </div>
    )
  }

  // ── DESKTOP LAYOUT ───────────────────────────────────────────────
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
                    {labelDisplay(ann.label)}
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
