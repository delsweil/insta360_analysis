'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { supabase, type Game, type Annotation, LABEL_DEFS, labelColor, labelDisplay } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import AnnotationShape from '@/components/AnnotationShape'
import ShareModal from '@/components/ShareModal'
import AnnotationFilter, { type FilterState, ALL_FILTERS, passesFilter } from '@/components/AnnotationFilter'
import ClipRecorder from '@/components/ClipRecorder'
import AnnotationCanvas, { type Shape, type Tool, type AnnotationCanvasHandle } from '@/components/AnnotationCanvas'
import RecordExportButton from '@/components/RecordExportButton'
import TimingStrip from '@/components/TimingStrip'
import TransportBar, { RATES } from '@/components/TransportBar'
import FloatingDrawTools, { DRAW_TOOLS } from '@/components/FloatingDrawTools'
import VeoImportModal from '@/components/VeoImportModal'
import BoardPanel from '@/components/BoardPanel'
import { useBoard } from '@/lib/board'
import { exportGame } from '@/lib/exportEvents'
import StatsPanel, { StatsCard } from '@/components/StatsPanel'
import { useStats } from '@/lib/stats'

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface Props {
  gameId?: string       // the normal view of a game (signed in)
  shareToken?: string   // a highlights link: the same view, read-only, limited to the shared annotations
}

export default function GameView({ gameId, shareToken }: Props) {
  const readOnly = !!shareToken
  const [id, setId] = useState(gameId ?? '')
  const [shareError, setShareError] = useState('')
  const playerRef = useRef<HTMLIFrameElement>(null)
  const videoWrapperRef = useRef<HTMLDivElement>(null)
  const mobileCanvasRef = useRef<AnnotationCanvasHandle>(null)
  const desktopCanvasRef = useRef<AnnotationCanvasHandle>(null)
  const firedAnnotationIds = useRef<Set<string>>(new Set())
  const prevTimeRef = useRef(0)

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
  const [saveError, setSaveError] = useState<string | null>(null)
  const [isMobile, setIsMobile] = useState(false)
  const [forceHud, setForceHud] = useState(() => { // fullscreen view chosen by hand (tablets), remembered on this device
    if (typeof window === 'undefined') return false
    try { return localStorage.getItem('hud-mode') === '1' } catch { return false }
  })
  const [showAnnotations, setShowAnnotations] = useState(false)
  const [showBoard, setShowBoard] = useState(false) // mobile HUD: board panel over the video
  const [boardOff, setBoardOff] = useState(() => { // where the viewer dragged the mobile board to (remembered)
    if (typeof window === 'undefined') return { x: 0, y: 0 }
    try {
      const v = JSON.parse(localStorage.getItem('board-offset') || 'null')
      if (v && typeof v.x === 'number' && typeof v.y === 'number') return { x: v.x as number, y: v.y as number }
    } catch {}
    return { x: 0, y: 0 }
  })
  const boardBox = useRef<HTMLDivElement>(null)
  const [boardScale, setBoardScale] = useState(() => { // size of the mobile board (1 = normal), remembered
    if (typeof window === 'undefined') return 1
    try {
      const v = Number(localStorage.getItem('board-scale'))
      if (v >= 0.6 && v <= 2.4) return v
    } catch {}
    return 1
  })
  const sizeBoard = (delta: number) => {
    const v = Math.round(Math.min(2.4, Math.max(0.6, boardScale + delta)) * 10) / 10
    setBoardScale(v)
    try { localStorage.setItem('board-scale', String(v)) } catch {}
  }
  const boardDrag = useRef<{ sx: number; sy: number; ox: number; oy: number } | null>(null)
  const { state: boardState, reload: reloadBoard } = useBoard(id)
  const { state: statsState } = useStats(id)
  const [showStats, setShowStats] = useState(false) // phone HUD: statistics sheet
  // where a half starts in the video, known when the board data has it
  const halfStart = useCallback((half: string) => {
    if (boardState.status !== 'ready') return null
    const p = boardState.board.parts.find(x => String(x.half) === half)
    return p ? p.video_start : null
  }, [boardState])
  const [showShare, setShowShare] = useState(false)
  const [markIn, setMarkIn] = useState<number | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [filter, setFilter] = useState<FilterState>(ALL_FILTERS)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawTool, setDrawTool] = useState<Tool>('select')
  const [draftShapes, setDraftShapes] = useState<Shape[]>([])
  const [isPlaying, setIsPlaying] = useState(true)
  const [hasEnded, setHasEnded] = useState(false)
  const [focusMode, setFocusMode] = useState(false)
  const [recordMode, setRecordMode] = useState<'current' | 'reel'>('current')
  const [reelIdx, setReelIdx] = useState(0)
  const [reelAutoPlay, setReelAutoPlay] = useState(false)
  const [reelFinished, setReelFinished] = useState(false)
  const reelStartedRef = useRef(false)
  const REEL_PRE_ROLL_SEC = 2
  const [loopReel, setLoopReel] = useState(false)
  const [loopDelaySec, setLoopDelaySec] = useState(5)
  const loopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [autoResumeSec, setAutoResumeSec] = useState(0) // seconds to stay paused before auto-resuming on its own (0 = wait for manual play)
  const [contextStartSec, setContextStartSec] = useState<number | null>(null) // clip starts — absolute video time playback begins from when jumped to
  const [annotationEndSec, setAnnotationEndSec] = useState<number | null>(null) // annotation removed — absolute video time the shapes stop showing
  const [clipEndSec, setClipEndSec] = useState<number | null>(null) // clip ends — absolute video time this annotation's range finishes
  const [editingAnnotationId, setEditingAnnotationId] = useState<string | null>(null)
  const autoResumeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [showVeoImport, setShowVeoImport] = useState(false)
  const [editingLabelId, setEditingLabelId] = useState<string | null>(null)
  const [hudVisible, setHudVisible] = useState(false)
  const [mobilePlaybackRate, setMobilePlaybackRate] = useState(1)
  const [isMuted, setIsMuted] = useState(true) // starts muted for autoplay
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showFsHint, setShowFsHint] = useState(false)

  // Shapes belonging to whichever tactical annotation's [shown, removed]
  // window currently contains playback — a pure time-range check now, not
  // tied to pause state. "Shown" = timestamp_sec, "removed" = the explicit
  // annotation_end_sec a coach sets on the timing strip (falls back to a
  // brief 1.5s window if never set, e.g. on older annotations).
  const activeTacticalShapes: Shape[] = isDrawing
    ? draftShapes
    : annotations.find(a =>
        a.label === 'tactical' && a.shapes?.length &&
        currentTime >= a.timestamp_sec &&
        currentTime <= (a.annotation_end_sec ?? a.timestamp_sec + 1.5)
      )?.shapes ?? []

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

  // Load game + user + annotations (a highlights link: the shared annotations only, nobody signed in is needed)
  useEffect(() => {
    let gid = gameId ?? ''
    async function load() {
      let only: string[] | null = null
      if (shareToken) {
        const { data: link } = await supabase.from('share_links').select('*').eq('token', shareToken).single()
        if (!link) { setShareError('This share link does not exist or has expired.'); setLoading(false); return }
        if (link.expires_at && new Date(link.expires_at) < new Date()) {
          setShareError('This share link has expired.'); setLoading(false); return
        }
        gid = link.game_id
        setId(gid)
        only = link.annotation_ids && link.annotation_ids.length > 0 ? link.annotation_ids : null
      } else {
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
      }

      const { data: gameData } = await supabase
        .from('games')
        .select('*')
        .eq('id', gid)
        .single()

      if (gameData) {
        setGame(gameData)
        setDuration(gameData.duration_sec ?? 0)
      }

      let q = supabase
        .from('annotations')
        .select(shareToken ? '*' : '*, profiles(display_name)')
        .eq('game_id', gid)
        .order('timestamp_sec')
      if (only) q = q.in('id', only)
      const { data: annData } = await q

      if (annData) setAnnotations(annData as unknown as Annotation[])
      setLoading(false)
    }
    load()
    if (shareToken) return

    // Realtime subscription
    const channel = supabase
      .channel(`game-${gid}`)
      .on('postgres_changes', {
        event: 'INSERT', schema: 'public',
        table: 'annotations', filter: `game_id=eq.${gid}`,
      }, (payload) => {
        const newAnn = payload.new as Annotation
        setAnnotations(prev => {
          if (prev.find(a => a.id === newAnn.id)) return prev
          return [...prev, newAnn].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
        })
      })
      .on('postgres_changes', {
        event: 'DELETE', schema: 'public',
        table: 'annotations', filter: `game_id=eq.${gid}`,
      }, (payload) => {
        setAnnotations(prev => prev.filter(a => a.id !== payload.old.id))
      })
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [gameId, shareToken])

  // YouTube postMessage tracking
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      try {
        const data = typeof e.data === 'string' ? JSON.parse(e.data) : e.data
        if (data?.event === 'infoDelivery' && data?.info) {
          if (data.info.currentTime !== undefined) setCurrentTime(data.info.currentTime)
          if (data.info.duration !== undefined && data.info.duration > 0) setDuration(data.info.duration)
          if (data.info.playerState !== undefined) {
            setIsPlaying(data.info.playerState === 1)
            if (data.info.playerState === 0) setHasEnded(true)
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

  // Track fullscreen state (handles Escape key / browser exit)
  useEffect(() => {
    const handler = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener('fullscreenchange', handler)
    return () => document.removeEventListener('fullscreenchange', handler)
  }, [])

  const seekTo = useCallback((sec: number) => {
    firedAnnotationIds.current.clear() // allow re-triggering annotations when a viewer jumps around
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'seekTo', args: [sec, true] }), '*'
    )
    playerRef.current?.contentWindow?.postMessage(
      JSON.stringify({ event: 'command', func: 'playVideo', args: [] }), '*'
    )
  }, [])

  // Moves the frame without forcing playback — used for live-scrubbing the
  // preview while dragging the trim handle, since the video should stay
  // paused (we're mid-edit) rather than start playing on every drag tick.
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

  // Desktop transport: jumps keep the play state (unlike seekTo, which always plays), single frames pause first
  const timeRef = useRef(0)
  useEffect(() => { timeRef.current = currentTime }, [currentTime])
  const seekBy = useCallback((delta: number) => {
    const t = Math.min(Math.max(0, timeRef.current + delta), duration > 0 ? duration : 99999)
    timeRef.current = t
    setCurrentTime(t)
    firedAnnotationIds.current.clear()
    scrubTo(t)
  }, [duration, scrubTo])
  const stepFrame = useCallback((dir: 1 | -1) => { pauseVideo(); seekBy(dir / 30) }, [pauseVideo, seekBy])
  const changeRate = useCallback((r: number) => { setMobilePlaybackRate(r); setVideoRate(r) }, [setVideoRate])

  const jumpSecFor = (ann: Annotation) =>
    ann.label === 'tactical' && ann.context_start_sec != null ? ann.context_start_sec : ann.timestamp_sec

  const startReelForRecording = useCallback(() => {
    if (filteredAnnotations.length === 0) return
    reelStartedRef.current = true
    setReelFinished(false)
    setReelIdx(0)
    seekTo(Math.max(0, jumpSecFor(filteredAnnotations[0]) - REEL_PRE_ROLL_SEC))
    setReelAutoPlay(true)
  }, [filteredAnnotations, seekTo])

  // Auto-advance through filteredAnnotations when the highlights-reel record
  // mode is running — same pattern as the share page's "Play all".
  useEffect(() => {
    if (!reelAutoPlay || filteredAnnotations.length === 0) return
    const ann = filteredAnnotations[reelIdx]
    if (!ann) return

    const endTime = ann.end_timestamp_sec ?? (ann.timestamp_sec + 10)
    if (currentTime >= endTime) {
      const nextIdx = reelIdx + 1
      if (nextIdx < filteredAnnotations.length) {
        setReelIdx(nextIdx)
        seekTo(Math.max(0, jumpSecFor(filteredAnnotations[nextIdx]) - REEL_PRE_ROLL_SEC))
      } else {
        setReelAutoPlay(false)
        // Signals "this pass is done" — RecordExportButton watches this and
        // stops/saves immediately if a recording is in progress. The optional
        // loop restart below is scheduled separately and happens afterward,
        // so it never extends an in-progress recording — it just resumes
        // normal (unrecorded) viewing once the delay elapses.
        if (reelStartedRef.current) setReelFinished(true)

        if (loopReel) {
          if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current)
          loopTimeoutRef.current = setTimeout(() => {
            startReelForRecording()
          }, loopDelaySec * 1000)
        }
      }
    }
  }, [currentTime, reelAutoPlay, reelIdx, filteredAnnotations, seekTo, loopReel, loopDelaySec, startReelForRecording])

  // Close the label dropdown when the user clicks anywhere outside it
  useEffect(() => {
    if (!editingLabelId) return
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest('[data-label-dropdown]')) {
        setEditingLabelId(null)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [editingLabelId])

  // Cancel any pending loop restart if the coach turns Loop off mid-wait, or leaves the page.
  useEffect(() => {
    if (!loopReel && loopTimeoutRef.current) {
      clearTimeout(loopTimeoutRef.current)
      loopTimeoutRef.current = null
    }
    return () => {
      if (loopTimeoutRef.current) clearTimeout(loopTimeoutRef.current)
    }
  }, [loopReel])

  // Auto-pause the moment playback naturally crosses into a tactical
  // annotation that has shapes, so viewers get a beat to look at it.
  // Guards against: re-triggering on scrub/seek jumps (only fires on
  // small forward steps, i.e. normal playback), and firing more than
  // once per annotation until the viewer seeks elsewhere.
  useEffect(() => {
    const prev = prevTimeRef.current
    const delta = currentTime - prev
    if (!isDrawing && delta > 0 && delta < 2) {
      const hit = annotations.find(a =>
        a.label === 'tactical' && a.shapes?.length &&
        !firedAnnotationIds.current.has(a.id) &&
        a.timestamp_sec > prev && a.timestamp_sec <= currentTime
      )
      if (hit) {
        firedAnnotationIds.current.add(hit.id)
        pauseVideo()
      }
    }
    prevTimeRef.current = currentTime
  }, [currentTime, annotations, isDrawing, pauseVideo])

  // Auto-resume playback on its own once auto_resume_sec elapses for the
  // paused annotation — for viewers watching hands-free only. Never for
  // coaches: a coach pausing near an existing annotation is very often
  // actually pausing to start a *new* one, and having playback yank itself
  // forward mid-thought was the actual bug behind "it restarts before I can
  // annotate" — this feature and coaching are fundamentally different intents.
  useEffect(() => {
    if (autoResumeTimeoutRef.current) {
      clearTimeout(autoResumeTimeoutRef.current)
      autoResumeTimeoutRef.current = null
    }
    if (isCoach || isDrawing || isPlaying) return

    const ann = annotations.find(a =>
      a.label === 'tactical' && a.shapes?.length &&
      currentTime >= a.timestamp_sec &&
      currentTime <= (a.annotation_end_sec ?? a.timestamp_sec + 1.5)
    )
    const delay = ann?.auto_resume_sec ?? 0
    if (ann && delay > 0) {
      autoResumeTimeoutRef.current = setTimeout(() => {
        resumeVideo()
      }, delay * 1000)
    }

    return () => {
      if (autoResumeTimeoutRef.current) clearTimeout(autoResumeTimeoutRef.current)
    }
  }, [currentTime, isPlaying, isDrawing, isCoach, annotations, resumeVideo])

  const startDrawing = useCallback(() => {
    pauseVideo()
    setDrawTool('select')
    setIsDrawing(true)
  }, [pauseVideo])

  const finishDrawing = useCallback(() => {
    mobileCanvasRef.current?.finalizePending()
    desktopCanvasRef.current?.finalizePending()
    setIsDrawing(false); setDrawTool('select'); setEditingAnnotationId(null); setMarkIn(null)
    setAnnotationEndSec(null); setClipEndSec(null)
  }, [])

  // Keyboard on the desktop layout: space/K play, arrows and J/L jump, comma/period frames, brackets speed, D draw, 1-8 tools
  useEffect(() => {
    if (isMobile || forceHud) return
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return
      if (e.metaKey || e.ctrlKey) {
        if (isDrawing && !e.shiftKey && (e.key === 'z' || e.key === 'Z')) { e.preventDefault(); setDraftShapes(prev => prev.slice(0, -1)) }
        return
      }
      if (e.altKey) return
      const k = e.key
      const idx = RATES.indexOf(mobilePlaybackRate)
      if (k === ' ' || k === 'k' || k === 'K') { e.preventDefault(); if (isPlaying) pauseVideo(); else resumeVideo() }
      else if (k === 'ArrowLeft') { e.preventDefault(); seekBy(e.shiftKey ? -1 : -5) }
      else if (k === 'ArrowRight') { e.preventDefault(); seekBy(e.shiftKey ? 1 : 5) }
      else if (k === 'j' || k === 'J') seekBy(-10)
      else if (k === 'l' || k === 'L') seekBy(10)
      else if (k === ',') stepFrame(-1)
      else if (k === '.') stepFrame(1)
      else if (k === '[') changeRate(RATES[Math.max(0, (idx < 0 ? 2 : idx) - 1)])
      else if (k === ']') changeRate(RATES[Math.min(RATES.length - 1, (idx < 0 ? 2 : idx) + 1)])
      else if ((k === 'd' || k === 'D') && isCoach && !isDrawing) {
        if (selectedLabel !== 'tactical') { setSelectedLabel('tactical'); setDraftShapes([]); setEditingAnnotationId(null) }
        startDrawing()
      }
      else if (isDrawing && k >= '1' && k <= String(DRAW_TOOLS.length)) setDrawTool(DRAW_TOOLS[Number(k) - 1].key)
      else if (isDrawing && k === 'Escape') setDrawTool('select')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [isMobile, forceHud, isDrawing, isPlaying, isCoach, selectedLabel, mobilePlaybackRate, pauseVideo, resumeVideo, seekBy, stepFrame, changeRate, startDrawing])

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

    let data: Annotation | null = null
    setSaveError(null)
    if (editingAnnotationId) {
      const patch = {
        timestamp_sec: markIn ?? undefined,
        shapes: draftShapes.length ? draftShapes : null,
        annotation_end_sec: draftShapes.length ? annotationEndSec : null,
        end_timestamp_sec: draftShapes.length ? clipEndSec : null,
        auto_resume_sec: draftShapes.length ? autoResumeSec : null,
        context_start_sec: draftShapes.length ? contextStartSec : null,
        note: note.trim() || null,
      }
      const { error } = await supabase
        .from('annotations')
        .update(patch)
        .eq('id', editingAnnotationId)
      if (error) {
        console.error('[handleRangeSave] update failed:', error)
        setSaveError(error.message)
      } else {
        // Don't rely on reading the row back — RLS may (correctly) block
        // reading annotations another coach created, even though the write
        // itself succeeded. Merge the known patch into local state instead.
        data = annotations.find(a => a.id === editingAnnotationId)
          ? { ...annotations.find(a => a.id === editingAnnotationId)!, ...patch, note: patch.note ?? undefined, end_timestamp_sec: patch.end_timestamp_sec ?? undefined, timestamp_sec: patch.timestamp_sec ?? annotations.find(a => a.id === editingAnnotationId)!.timestamp_sec }
          : null
        if (data) {
          const updated = data
          setAnnotations(prev => prev.map(a => a.id === editingAnnotationId ? updated : a))
        }
      }
    } else {
      const res = await supabase
        .from('annotations')
        .insert({
          game_id: id, user_id: user.id,
          timestamp_sec: markIn,
          end_timestamp_sec: draftShapes.length ? clipEndSec : (currentTime > markIn + 1 ? currentTime : null),
          label: selectedLabel,
          note: note.trim() || null,
          shapes: draftShapes.length ? draftShapes : null,
          annotation_end_sec: draftShapes.length ? annotationEndSec : null,
          auto_resume_sec: draftShapes.length ? autoResumeSec : null,
          context_start_sec: draftShapes.length ? contextStartSec : null,
          is_public: true,
        })
        .select('*, profiles(display_name)')
        .single()
      data = res.data
      if (res.error) {
        console.error('[handleRangeSave] insert failed:', res.error)
        setSaveError(res.error.message)
      }
      if (data) {
        const inserted = data
        setAnnotations(prev => [...prev, inserted].sort((a, b) => a.timestamp_sec - b.timestamp_sec))
      }
    }

    if (data) {
      setSaved(true)
      setTimeout(() => setSaved(false), 1500)
    }
    setMarkIn(null)
    setDraftShapes([])
    setIsDrawing(false)
    setAutoResumeSec(0)
    setContextStartSec(null)
    setAnnotationEndSec(null)
    setClipEndSec(null)
    setEditingAnnotationId(null)
    setNote('')
    setSaving(false)
  }, [saving, markIn, currentTime, selectedLabel, note, draftShapes, annotationEndSec, clipEndSec, autoResumeSec, contextStartSec, editingAnnotationId, id])

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
            onClick={() => {
              if (isDisabled) return
              const switchingToTactical = l.key === 'tactical' && selectedLabel !== 'tactical'
              setSelectedLabel(l.key)
              if (l.key === 'tactical' && isCoach) {
                if (switchingToTactical) { setDraftShapes([]); setEditingAnnotationId(null) }
                startDrawing()
              }
            }}
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
  const Timeline = ({ height = 24, mobileSize = false }: { height?: number, mobileSize?: boolean }) => {
    const scrub = (e: React.PointerEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
      scrubTo(ratio * duration) // moves the frame without forcing playback — lets you land somewhere and stay paused
    }

    return (
    <div
      style={{
        position: 'relative', height,
        background: '#F8F8F6', borderRadius: 4,
        border: '1px solid #E4E6EE', cursor: 'pointer', touchAction: 'none',
      }}
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); scrub(e) }}
      onPointerMove={(e) => { if (e.buttons === 1) scrub(e) }}
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
            onClick={(e) => {
              e.stopPropagation()
              const jumpSec = ann.label === 'tactical' && ann.context_start_sec != null ? ann.context_start_sec : ann.timestamp_sec
              seekTo(jumpSec)
            }}
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
              size={mobileSize ? 10 : 7}
            />
          </div>
        )
      })}
    </div>
    )
  }

  // Patch a single annotation's label in DB + local state
  const handleLabelChange = useCallback(async (annId: string, newLabel: string) => {
    setEditingLabelId(null)
    const { error } = await supabase
      .from('annotations')
      .update({ label: newLabel })
      .eq('id', annId)
    if (!error) {
      setAnnotations(prev => prev.map(a => a.id === annId ? { ...a, label: newLabel } : a))
    }
  }, [])

  // Annotation row
  const AnnRow = ({ ann, compact = false }: { ann: Annotation, compact?: boolean }) => {
    const isLabelOpen = editingLabelId === ann.id
    const color = labelColor(ann.label)

    return (
    <div
      style={{
        padding: compact ? '8px 10px' : '7px 0',
        borderBottom: compact ? 'none' : '1px solid #F8F8F6',
        borderRadius: compact ? 8 : 0,
        marginBottom: compact ? 8 : 0,
        background: compact ? '#fff' : 'transparent',
        border: compact ? '1px solid #E4E6EE' : undefined,
        cursor: 'pointer',
        position: 'relative',
      }}
      onClick={() => {
        if (isLabelOpen) return
        const jumpSec = ann.label === 'tactical' && ann.context_start_sec != null ? ann.context_start_sec : ann.timestamp_sec
        seekTo(jumpSec)
        if (isMobile || forceHud) setShowAnnotations(false)
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

        {/* Label — clickable badge for coaches */}
        {isCoach && ann.label !== 'tactical' ? (
          <div style={{ position: 'relative', flex: 1 }}>
            <button
              data-label-dropdown
              onClick={(e) => {
                e.stopPropagation()
                setEditingLabelId(isLabelOpen ? null : ann.id)
              }}
              title="Click to change label"
              style={{
                fontSize: compact ? 12 : 11, fontWeight: 600,
                padding: '2px 8px', borderRadius: 99,
                border: `1.5px solid ${color}`,
                background: `${color}18`,
                color, cursor: 'pointer',
                fontFamily: 'DM Sans, sans-serif',
                display: 'inline-flex', alignItems: 'center', gap: 4,
              }}
            >
              {labelDisplay(ann.label, game?.home_team, game?.away_team)}
              <span style={{ fontSize: 9, opacity: 0.6 }}>▾</span>
            </button>

            {/* Dropdown */}
            {isLabelOpen && (
              <div
                data-label-dropdown
                onClick={e => e.stopPropagation()}
                style={{
                  position: 'absolute', top: '100%', left: 0, zIndex: 200,
                  marginTop: 4, background: '#fff',
                  border: '1px solid #E4E6EE', borderRadius: 10,
                  boxShadow: '0 8px 24px rgba(15,41,114,0.14)',
                  minWidth: 160, overflow: 'hidden',
                  fontFamily: 'DM Sans, sans-serif',
                }}
              >
                {LABEL_DEFS.filter(l => l.key !== 'tactical').map(l => {
                  const lc = labelColor(l.key)
                  const isActive = l.key === ann.label
                  return (
                    <button
                      key={l.key}
                      onClick={() => handleLabelChange(ann.id, l.key)}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 7,
                        width: '100%', padding: '7px 12px',
                        border: 'none', background: isActive ? `${lc}12` : 'transparent',
                        cursor: 'pointer', textAlign: 'left',
                        fontSize: 12, fontWeight: isActive ? 700 : 400,
                        color: isActive ? lc : '#111318',
                        borderBottom: '1px solid #F8F8F6',
                      }}
                    >
                      <AnnotationShape labelKey={l.key} size={10} />
                      {l.display}
                      {l.team !== 'none' && (
                        <span style={{ fontSize: 10, opacity: 0.55, marginLeft: 'auto' }}>
                          {l.team === 'home' ? game?.home_team ?? 'Home' : game?.away_team ?? 'Away'}
                        </span>
                      )}
                      {isActive && <span style={{ marginLeft: 'auto', fontSize: 10, color: lc }}>✓</span>}
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        ) : (
          <span style={{ fontSize: compact ? 13 : 12, color: '#111318', flex: 1 }}>
            {labelDisplay(ann.label, game?.home_team, game?.away_team)}
          </span>
        )}

        {ann.profiles?.display_name && (
          <span style={{ fontSize: 10, color: '#8A8F9E', flexShrink: 0 }}>
            {ann.profiles.display_name}
          </span>
        )}
        {ann.label === 'tactical' && ann.shapes?.length && isCoach && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              seekTo(ann.timestamp_sec)
              pauseVideo()
              setSelectedLabel('tactical')
              setMarkIn(ann.timestamp_sec)
              setDraftShapes(ann.shapes ?? [])
              setAnnotationEndSec(ann.annotation_end_sec ?? null)
              setClipEndSec(ann.end_timestamp_sec ?? null)
              setAutoResumeSec(ann.auto_resume_sec ?? 0)
              setContextStartSec(ann.context_start_sec ?? null)
              setEditingAnnotationId(ann.id)
              setDrawTool('select')
              setIsDrawing(true)
              if (isMobile || forceHud) setShowAnnotations(false)
            }}
            style={{
              fontSize: 10, color: '#0f2972', fontWeight: 600,
              background: 'none', border: 'none',
              cursor: 'pointer', padding: '2px 4px', flexShrink: 0,
            }}
          >
            Edit
          </button>
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
  }

  if (loading) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C',
    }}>
      Loading...
    </div>
  )

  if (shareError) return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'DM Sans, sans-serif', padding: 24, textAlign: 'center',
    }}>
      {shareError}
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

  // Shared style for HUD transport buttons (used inside the isMobile block)
  const hudTransportBtn: React.CSSProperties = {
    background: 'rgba(0,0,0,0.45)',
    backdropFilter: 'blur(8px)',
    border: '1px solid rgba(255,255,255,0.15)',
    borderRadius: '50%',
    color: '#fff',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
    width: 44,
    height: 44,
  }

  const chooseHud = (on: boolean) => {
    setForceHud(on)
    try { localStorage.setItem('hud-mode', on ? '1' : '0') } catch {}
  }

  // The draggable see-through board over the video: the HUD and the desktop view both show it,
  // and it stays when the controls are hidden
  const boardOverlay = (
    <>
        {/* ── Tactical board: small see-through pitch you can drag (stays on screen, also when the HUD hides) ── */}
        {showBoard && boardState.status === 'ready' && (
          <div
            ref={boardBox}
            style={{
              position: 'absolute', zIndex: 18, pointerEvents: 'none',
              right: 'max(8px, env(safe-area-inset-right))',
              top: 'max(52px, calc(env(safe-area-inset-top) + 42px))',
              width: `calc(min(22vw, 190px) * ${boardScale})`,
              transform: `translate(${boardOff.x}px, ${boardOff.y}px)`,
            }}
          >
            <div
              onPointerDown={e => {
                e.currentTarget.setPointerCapture(e.pointerId)
                boardDrag.current = { sx: e.clientX, sy: e.clientY, ox: boardOff.x, oy: boardOff.y }
              }}
              onPointerMove={e => {
                const d = boardDrag.current
                const el = boardBox.current
                const parent = el?.offsetParent as HTMLElement | null
                if (!d || !el || !parent) return
                const r = el.getBoundingClientRect()
                const pr = parent.getBoundingClientRect()
                let nx = d.ox + (e.clientX - d.sx)
                let ny = d.oy + (e.clientY - d.sy)
                nx = Math.max(boardOff.x + (pr.left - r.left), Math.min(boardOff.x + (pr.right - r.right), nx))
                ny = Math.max(boardOff.y + (pr.top - r.top), Math.min(boardOff.y + (pr.bottom - r.bottom), ny))
                setBoardOff({ x: nx, y: ny })
              }}
              onPointerUp={() => {
                boardDrag.current = null
                try { localStorage.setItem('board-offset', JSON.stringify(boardOff)) } catch {}
              }}
              onPointerCancel={() => { boardDrag.current = null }}
              style={{ pointerEvents: 'auto', touchAction: 'none', cursor: 'grab' }}
            >
              <BoardPanel gameId={id} gameTitle={game.title} state={boardState} reload={reloadBoard} currentTime={currentTime} isCoach={false} compact />
            </div>
            <button
              onClick={() => setShowBoard(false)}
              aria-label="Close board"
              style={{
                position: 'absolute', top: -4, left: -30, width: 26, height: 26, borderRadius: '50%', pointerEvents: 'auto',
                border: '1px solid rgba(255,255,255,0.3)', background: 'rgba(0,0,0,0.5)',
                color: '#fff', fontSize: 13, cursor: 'pointer',
              }}
            >
              ✕
            </button>
            {([['+', 0.2, 'Bigger board', 26], ['−', -0.2, 'Smaller board', 56]] as const).map(([ch, d, label, top]) => (
              <button
                key={ch}
                onClick={() => sizeBoard(d)}
                aria-label={label}
                style={{
                  position: 'absolute', top, left: -30, width: 26, height: 26, borderRadius: '50%', pointerEvents: 'auto',
                  border: '1px solid rgba(255,255,255,0.3)', background: 'rgba(0,0,0,0.5)',
                  color: '#fff', fontSize: 15, lineHeight: 1, cursor: 'pointer',
                }}
              >
                {ch}
              </button>
            ))}
          </div>
        )}
    </>
  )

  // ── MOBILE LAYOUT — landscape fullscreen HUD (phones, or any device where the viewer chose it) ──
  if (isMobile || forceHud) {
    const activeLabelDef = LABEL_DEFS.find(l => l.key === selectedLabel)
    const activeLabelColor = labelColor(selectedLabel)
    const activeLabelDisplay = labelDisplay(selectedLabel, game.home_team, game.away_team)

    // HUD auto-hides after 3 s of no interaction; tap video to show/hide
    // We track this with a simple toggle + timeout ref.

    const mobileMarkClip = () => {
      if (markIn === null) {
        // First tap: mark in
        setMarkIn(currentTime)
        setIsRecording(true)
        setSaved(false)
      } else {
        // Second tap: save the range
        const startTime = markIn
        const endTime = currentTime > markIn + 0.5 ? currentTime : markIn + 1
        ;(async () => {
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
            setAnnotations(prev => [...prev, data].sort((a, b) => a.timestamp_sec - b.timestamp_sec))
            setSaved(true)
            setTimeout(() => setSaved(false), 1500)
          }
          setMarkIn(null)
          setIsRecording(false)
          setSaving(false)
        })()
      }
    }

    const SPEED_STEPS = [0.5, 1, 1.25, 1.5, 2]

    const cycleSpeed = () => {
      const idx = SPEED_STEPS.indexOf(mobilePlaybackRate)
      const next = SPEED_STEPS[(idx + 1) % SPEED_STEPS.length]
      setMobilePlaybackRate(next)
      setVideoRate(next)
    }

    return (
      <div
        style={{
          position: 'fixed', inset: 0,
          background: '#000',
          fontFamily: 'DM Sans, sans-serif',
          overflow: 'hidden',
        }}
      >
        {/* ── VIDEO fills the entire screen ── */}
        <iframe
          ref={playerRef}
          src={`${game.video_url}?enablejsapi=1&controls=0&rel=0&modestbranding=1&autoplay=1&mute=1&playsinline=1`}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', border: 'none' }}
          allowFullScreen
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        />

        {/* Invisible tap-catcher — sits above the iframe (zIndex 7).
            Receives taps on the video area and toggles HUD visibility.
            Disabled while drawing so pointer events reach AnnotationCanvas. */}
        <div
          onClick={() => { if (!isDrawing) setHudVisible(v => !v) }}
          style={{
            position: 'absolute', inset: 0, zIndex: 7, background: 'transparent',
            pointerEvents: isDrawing ? 'none' : 'auto',
          }}
        />

        {/* ── HUD wrapper — shown/hidden by tap ── */}
        {/* Tapping the wrapper background (not a button) hides the HUD.
            While drawing, always visible so the draw toolbar stays accessible. */}
        <div
          onClick={() => { if (!isDrawing) setHudVisible(false) }}
          style={{
            position: 'absolute', inset: 0,
            opacity: (hudVisible || isDrawing) ? 1 : 0,
            transition: 'opacity 0.2s ease',
            pointerEvents: (hudVisible || isDrawing) ? 'auto' : 'none',
            zIndex: 8,
          }}
        >

        {/* Tactical shapes overlay */}
        <AnnotationCanvas
          ref={mobileCanvasRef}
          shapes={activeTacticalShapes}
          editable={isDrawing}
          tool={drawTool}
          onAddShape={s => setDraftShapes(prev => [...prev, s])}
          onUpdateShape={(shapeId, patch) =>
            setDraftShapes(prev => prev.map(s => s.id === shapeId ? { ...s, ...patch } as Shape : s))
          }
          onRemoveShape={shapeId => setDraftShapes(prev =>
            prev.filter(s => s.id !== shapeId)
              .filter(s => !(s.type === 'connector' && (s.fromId === shapeId || s.toId === shapeId)))
          )}
          onToggleVideo={undefined}
        />

        {/* Orange recording border */}
        {isRecording && (
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 5,
            border: '3px solid rgba(232,120,10,0.75)',
            boxShadow: 'inset 0 0 0 1px rgba(232,120,10,0.3)',
          }} />
        )}

        {/* ── HUD — transparent overlay ── */}
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

        {/* Top bar — safe-area-inset-top pushes content below Dynamic Island / notch */}
        <div onClick={e => e.stopPropagation()} style={{
          position: 'absolute', top: 0, left: 0, right: 0,
          zIndex: 10, display: 'flex', alignItems: 'center',
          justifyContent: 'space-between',
          paddingTop: 'max(10px, env(safe-area-inset-top))',
          paddingLeft: 14, paddingRight: 14, paddingBottom: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button
              onClick={() => window.history.back()}
              style={{
                background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.15)',
                borderRadius: 8, color: 'rgba(255,255,255,0.7)',
                fontSize: 12, fontWeight: 600, padding: '5px 10px', cursor: 'pointer',
              }}
            >
              ← Zurück
            </button>
            {!isMobile && (
              <button
                onClick={() => chooseHud(false)}
                style={{
                  background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: 8, color: 'rgba(255,255,255,0.7)',
                  fontSize: 12, fontWeight: 600, padding: '5px 10px', cursor: 'pointer',
                }}
              >
                ▭ Normale Ansicht
              </button>
            )}
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
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif', fontSize: 14,
            color: 'rgba(255,255,255,0.9)', letterSpacing: '0.08em',
            textShadow: '0 1px 4px rgba(0,0,0,0.5)',
          }}>
            {game.title}
          </div>
          <div style={{
            fontFamily: 'Bebas Neue, sans-serif', fontSize: 16,
            color: '#fff', letterSpacing: '0.1em',
            background: 'rgba(255,255,255,0.12)', backdropFilter: 'blur(6px)',
            padding: '3px 12px', borderRadius: 7,
          }}>
            {annotations.filter(a => a.label === 'goal_home').length}
            {' – '}
            {annotations.filter(a => a.label === 'goal_away').length}
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
              onClick={() => seekTo(Math.max(0, currentTime + delta))}
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
              onClick={() => seekTo(Math.min(duration || 9999, currentTime + delta))}
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
          zIndex: 10,
          paddingTop: 0, paddingLeft: 12, paddingRight: 12,
          paddingBottom: 'max(10px, env(safe-area-inset-bottom))',
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
                {/* markIn range highlight */}
                {markIn !== null && duration > 0 && (
                  <div style={{
                    position: 'absolute', top: 0, bottom: 0,
                    left: `${(markIn / duration) * 100}%`,
                    width: `${((currentTime - markIn) / duration) * 100}%`,
                    background: 'rgba(232,120,10,0.6)',
                  }} />
                )}
              </div>
              {/* Playhead */}
              <div style={{
                position: 'absolute', top: '50%',
                transform: 'translate(-50%, -50%)',
                left: duration > 0 ? `${(currentTime / duration) * 100}%` : '0%',
                width: 14, height: 14, borderRadius: '50%',
                background: '#fff',
                boxShadow: isRecording ? '0 0 0 2px #E8780A, 0 0 8px rgba(232,120,10,0.6)' : '0 0 0 2px #E8780A',
              }} />
              {/* markIn pin */}
              {markIn !== null && duration > 0 && (
                <div style={{
                  position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)',
                  left: `${(markIn / duration) * 100}%`,
                  width: 3, height: 22, background: '#E8780A', borderRadius: 2,
                }} />
              )}
              {/* Annotation markers */}
              {annotations.map(ann => (
                <div
                  key={ann.id}
                  onClick={e => { e.stopPropagation(); seekTo(ann.timestamp_sec) }}
                  style={{
                    position: 'absolute', top: '50%',
                    transform: 'translate(-50%, -50%)',
                    left: duration > 0 ? `${(ann.timestamp_sec / duration) * 100}%` : '0%',
                    width: 10, height: 10, borderRadius: '50%',
                    background: labelColor(ann.label),
                    border: '2px solid rgba(0,0,0,0.4)',
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

          {/* Action row — drawing mode replaces normal controls */}
          {isDrawing ? (
            /* ── Mobile draw toolbar ── */
            <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
              {/* Tool picker row */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, overflowX: 'auto', paddingBottom: 2 }}>
                {([
                  { key: 'select',    icon: '↖', label: 'Auswahl' },
                  { key: 'zone',      icon: '▭', label: 'Zone' },
                  { key: 'curve',     icon: '↝', label: 'Linie' },
                  { key: 'connector', icon: '⌇', label: 'Pfeil' },
                  { key: 'cone',      icon: '◣', label: 'Spieler' },
                  { key: 'highlight', icon: '◎', label: 'Spot' },
                  { key: 'label',     icon: 'T', label: 'Text' },
                  { key: 'number',    icon: '①', label: 'Nr.' },
                ] as const).map(t => {
                  const isSelected = drawTool === t.key
                  return (
                    <button
                      key={t.key}
                      onClick={() => setDrawTool(t.key as typeof drawTool)}
                      title={t.label}
                      style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
                        padding: '5px 8px', borderRadius: 8, flexShrink: 0,
                        border: `1.5px solid ${isSelected ? '#4ade80' : 'rgba(255,255,255,0.2)'}`,
                        background: isSelected ? 'rgba(74,222,128,0.15)' : 'rgba(255,255,255,0.07)',
                        color: isSelected ? '#4ade80' : 'rgba(255,255,255,0.7)',
                        cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                      }}
                    >
                      <span style={{ fontSize: 14 }}>{t.icon}</span>
                      <span style={{ fontSize: 8, fontWeight: 600, letterSpacing: '0.03em' }}>{t.label}</span>
                    </button>
                  )
                })}
              </div>
              {/* Undo / Clear / Done row */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                <button
                  onClick={() => setDraftShapes(prev => prev.slice(0, -1))}
                  disabled={draftShapes.length === 0}
                  style={{
                    padding: '0 12px', height: 36,
                    background: 'rgba(255,255,255,0.07)',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: 9,
                    color: draftShapes.length > 0 ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.25)',
                    fontSize: 12, fontWeight: 600,
                    fontFamily: 'DM Sans, sans-serif', cursor: draftShapes.length > 0 ? 'pointer' : 'default',
                    flexShrink: 0,
                  }}
                >↩ Undo</button>
                <button
                  onClick={() => setDraftShapes([])}
                  style={{
                    padding: '0 12px', height: 36,
                    background: 'rgba(239,68,68,0.1)',
                    border: '1px solid rgba(239,68,68,0.3)',
                    borderRadius: 9, color: 'rgba(239,68,68,0.85)',
                    fontSize: 12, fontWeight: 600,
                    fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                  }}
                >✕ Clear</button>
                <div style={{ flex: 1 }} />
                {/* Hint */}
                <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', flexShrink: 0 }}>
                  {draftShapes.length} shapes
                </span>
                <button
                  onClick={() => {
                    mobileCanvasRef.current?.finalizePending()
                    desktopCanvasRef.current?.finalizePending()
                    setIsDrawing(false)
                    setDrawTool('select')
                    setEditingAnnotationId(null)
                    setMarkIn(null)
                    setAnnotationEndSec(null)
                    setClipEndSec(null)
                  }}
                  style={{
                    padding: '0 16px', height: 36,
                    background: '#0f2972', border: 'none',
                    borderRadius: 9, color: '#fff',
                    fontSize: 12, fontWeight: 700,
                    fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                  }}
                >✓ Fertig</button>

                {/* Fullscreen toggle (drawing mode) */}
                <button
                  onClick={() => {
                    if (!document.fullscreenElement) {
                      document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {})
                    } else {
                      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {})
                    }
                  }}
                  style={{
                    padding: '0 11px', height: 36,
                    background: 'rgba(255,255,255,0.08)',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    cursor: 'pointer', flexShrink: 0,
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    {isFullscreen
                      ? <path d="M5 2H2v3M9 2h3v3M5 12H2V9M9 12h3V9" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                      : <path d="M2 5V2h3M9 2h3v3M2 9v3h3M12 9v3H9" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    }
                  </svg>
                </button>
              </div>
            </div>
          ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            {!readOnly && (<>
            {/* Mark / Save clip button */}
            <button
              onClick={mobileMarkClip}
              style={{
                display: 'flex', alignItems: 'center', gap: 7,
                padding: '0 14px', height: 38,
                background: isRecording ? '#E8780A' : '#0f2972',
                border: 'none', borderRadius: 9,
                color: '#fff', fontSize: 13, fontWeight: 700,
                fontFamily: 'DM Sans, sans-serif', cursor: 'pointer',
                flexShrink: 0,
                boxShadow: isRecording ? '0 0 0 0 rgba(232,120,10,0.5)' : 'none',
                animation: isRecording ? 'mobilePulse 1.2s infinite' : 'none',
              }}
            >
              {isRecording
                ? <>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <path d="M2 7l3.5 3.5 6.5-7" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                    {saving ? 'Speichern…' : `Speichern · ${formatTime(markIn ?? 0)}–${formatTime(currentTime)}`}
                  </>
                : <>
                    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <circle cx="7" cy="7" r="5.5" stroke="white" strokeWidth="1.5"/>
                      <line x1="7" y1="4" x2="7" y2="10" stroke="white" strokeWidth="1.5"/>
                      <line x1="4" y1="7" x2="10" y2="7" stroke="white" strokeWidth="1.5"/>
                    </svg>
                    Clip markieren
                  </>
              }
            </button>

            {/* Cancel (only while recording) */}
            {isRecording && (
              <button
                onClick={() => { setMarkIn(null); setIsRecording(false); setSaved(false) }}
                style={{
                  padding: '0 10px', height: 38,
                  background: 'rgba(239,68,68,0.15)',
                  border: '1px solid rgba(239,68,68,0.35)',
                  borderRadius: 9, color: 'rgba(239,68,68,0.9)',
                  fontSize: 12, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                ✕
              </button>
            )}

            {/* Label pill */}
            <button
              onClick={() => setShowAnnotations(prev => !prev)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '0 11px', height: 38,
                background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(8px)',
                border: `1px solid ${activeLabelColor}66`,
                borderRadius: 9, color: '#fff',
                fontSize: 12, fontWeight: 600,
                fontFamily: 'DM Sans, sans-serif', cursor: 'pointer',
                flexShrink: 0,
              }}
              // We reuse showAnnotations to mean "show label sheet" on mobile HUD
            >
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: activeLabelColor, flexShrink: 0 }} />
              {activeLabelDisplay} ▾
            </button>

            </>)}

            {/* Tactical board — only when this game has board data */}
            {boardState.status === 'ready' && (
              <button
                onClick={() => setShowBoard(v => !v)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '0 11px', height: 38,
                  background: showBoard ? '#E8780A' : 'rgba(255,255,255,0.1)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.25)',
                  borderRadius: 9, color: '#fff',
                  fontSize: 12, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                ⚽ Board
              </button>
            )}

            {/* Match statistics — only when this game has them */}
            {statsState.status === 'ready' && (
              <button
                onClick={() => { setShowStats(true); setHudVisible(true) }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '0 11px', height: 38,
                  background: 'rgba(255,255,255,0.1)', backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255,255,255,0.25)',
                  borderRadius: 9, color: '#fff',
                  fontSize: 12, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                📊 Statistik
              </button>
            )}

            {/* Tactical draw button — coaches only */}
            {isCoach && (
              <button
                onClick={() => {
                  setSelectedLabel('tactical')
                  setHudVisible(true)
                  startDrawing()
                }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '0 11px', height: 38,
                  background: 'rgba(255,255,255,0.07)',
                  border: '1px solid rgba(255,255,255,0.18)',
                  borderRadius: 9, color: 'rgba(255,255,255,0.8)',
                  fontSize: 12, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', cursor: 'pointer', flexShrink: 0,
                }}
              >
                ✏ Draw
              </button>
            )}

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

            {/* Fullscreen / Add-to-Home-Screen button */}
            <div style={{ position: 'relative', flexShrink: 0 }}>
              <button
                onClick={() => {
                  const isIOS = /iP(hone|ad|od)/.test(navigator.userAgent)
                  if (isIOS) {
                    setShowFsHint(h => !h)
                  } else if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {})
                  } else {
                    document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {})
                  }
                }}
                style={{
                  padding: '0 11px', height: 38,
                  background: 'rgba(255,255,255,0.08)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  {isFullscreen
                    ? <path d="M5 2H2v3M9 2h3v3M5 12H2V9M9 12h3V9" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    : <path d="M2 5V2h3M9 2h3v3M2 9v3h3M12 9v3H9" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  }
                </svg>
              </button>
              {showFsHint && (
                <div
                  onClick={() => setShowFsHint(false)}
                  style={{
                    position: 'absolute', bottom: 46, right: 0,
                    background: 'rgba(15,41,114,0.97)', backdropFilter: 'blur(12px)',
                    border: '1px solid rgba(255,255,255,0.15)',
                    borderRadius: 10, padding: '10px 14px',
                    width: 220, zIndex: 30,
                    color: '#fff', fontSize: 12, fontWeight: 500,
                    fontFamily: 'DM Sans, sans-serif', lineHeight: 1.5,
                  }}
                >
                  <div style={{ fontWeight: 700, marginBottom: 4 }}>Vollbild auf iPhone</div>
                  Tippe auf <strong>Teilen ⎋</strong> → <strong>„Zum Home-Bildschirm"</strong> — dann startet die App ohne Browserleiste.
                  <div style={{ marginTop: 8, color: 'rgba(255,255,255,0.45)', fontSize: 11 }}>Tippe irgendwo zum Schließen</div>
                </div>
              )}
            </div>
          </div>
          )}
        </div>

        {boardOverlay}

        {/* ── Statistics sheet (slides up from bottom) ── */}
        {showStats && statsState.status === 'ready' && (
          <div
            onClick={() => setShowStats(false)}
            style={{ position: 'absolute', inset: 0, zIndex: 22, background: 'rgba(0,0,0,0.35)' }}
          >
            <div
              onClick={e => e.stopPropagation()}
              style={{
                position: 'absolute', bottom: 0, left: 0, right: 0, maxHeight: '88%',
                display: 'flex', flexDirection: 'column',
                background: 'rgba(8,12,24,0.96)', backdropFilter: 'blur(16px)',
                borderTop: '1px solid rgba(255,255,255,0.1)', borderRadius: '16px 16px 0 0',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px 6px' }}>
                <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 22, letterSpacing: '0.04em', color: '#fff' }}>Statistik</div>
                <button
                  onClick={() => setShowStats(false)} aria-label="Schließen"
                  style={{ width: 30, height: 30, borderRadius: '50%', border: '1px solid rgba(255,255,255,0.25)', background: 'rgba(255,255,255,0.08)', color: '#fff', fontSize: 14, cursor: 'pointer' }}
                >✕</button>
              </div>
              <div style={{ overflowY: 'auto', padding: '4px 14px max(22px, env(safe-area-inset-bottom))', WebkitOverflowScrolling: 'touch' }}>
                <StatsPanel
                  stats={statsState.stats} dark halfStart={halfStart} now={currentTime}
                  onSeek={sec => { seekTo(sec); setShowStats(false) }}
                />
              </div>
            </div>
          </div>
        )}

        {/* ── Label sheet (slides up from bottom) ── */}
        {!readOnly && showAnnotations && (
          <div
            onClick={() => setShowAnnotations(false)}
            style={{
              position: 'absolute', inset: 0, zIndex: 20,
              background: 'rgba(0,0,0,0.3)',
            }}
          >
            <div
              onClick={e => e.stopPropagation()}
              style={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                background: 'rgba(8,12,24,0.92)', backdropFilter: 'blur(16px)',
                borderTop: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '16px 16px 0 0',
                padding: '0 14px 28px',
              }}
            >
              <div style={{ width: 32, height: 4, background: 'rgba(255,255,255,0.2)', borderRadius: 2, margin: '10px auto 12px' }} />
              <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'rgba(255,255,255,0.4)', marginBottom: 10 }}>
                Kategorie wählen
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                {LABEL_DEFS.filter(l => isCoach || !l.coachOnly).map(l => {
                  const isSelected = selectedLabel === l.key
                  const color = labelColor(l.key)
                  return (
                    <button
                      key={l.key}
                      onClick={() => {
                        setSelectedLabel(l.key)
                        setShowAnnotations(false)
                      }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        padding: '9px 10px', borderRadius: 8,
                        border: `1.5px solid ${isSelected ? color : 'rgba(255,255,255,0.1)'}`,
                        background: isSelected ? `${color}30` : 'rgba(255,255,255,0.05)',
                        color: isSelected ? color : 'rgba(255,255,255,0.8)',
                        fontSize: 12, fontWeight: 600, cursor: 'pointer',
                        fontFamily: 'DM Sans, sans-serif',
                        textAlign: 'left',
                      }}
                    >
                      <div style={{ width: 8, height: 8, borderRadius: '50%', background: color, flexShrink: 0 }} />
                      {labelDisplay(l.key, game.home_team, game.away_team)}
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )}

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
                  Noch keine Clips
                </div>
              ) : annotations.map(ann => {
                const color = labelColor(ann.label)
                const isActive = Math.abs(currentTime - ann.timestamp_sec) < 3
                return (
                  <div
                    key={ann.id}
                    onClick={() => seekTo(ann.timestamp_sec)}
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
                      {labelDisplay(ann.label, game.home_team, game.away_team)}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        </div>{/* end HUD wrapper */}

        {/* Pulse animation for recording button */}
        <style>{`
          @keyframes mobilePulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(232,120,10,0.5); }
            50% { box-shadow: 0 0 0 6px rgba(232,120,10,0); }
          }
        `}</style>
      </div>
    )
  }


  // ── DESKTOP LAYOUT ─────────────────────────────────────────────
  return (
    <div style={{
      minHeight: '100vh', background: '#F8F8F6',
      fontFamily: 'DM Sans, sans-serif', color: '#111318',
    }}>
      {!focusMode && (readOnly
        ? <div style={{ pointerEvents: 'none' }}><Topbar backHref="/" /></div>
        : <Topbar role={userRole} backHref="/" />)}

      <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {/* Game title bar */}
        {!focusMode && (
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
              &nbsp;·&nbsp;
              <button
                onClick={() => chooseHud(true)}
                title="Fullscreen view with the controls over the video (remembered on this device)"
                style={{ border: 'none', background: 'none', padding: 0, cursor: 'pointer', color: '#0f2972', fontWeight: 600, fontSize: 11 }}
              >
                ⤢ Vollbild-Ansicht
              </button>
            </div>
          </div>
          {isCoach && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <div style={{
                display: 'flex', border: '1px solid #E4E6EE', borderRadius: 8,
                overflow: 'hidden', flexShrink: 0,
              }}>
                <button
                  onClick={() => setRecordMode('current')}
                  title="Record starting from wherever the video is now"
                  style={{
                    fontSize: 11, fontWeight: 600, padding: '6px 10px', border: 'none',
                    background: recordMode === 'current' ? '#0f2972' : '#fff',
                    color: recordMode === 'current' ? '#fff' : '#8A8F9E',
                    cursor: 'pointer',
                  }}
                >
                  From now
                </button>
                <button
                  onClick={() => setRecordMode('reel')}
                  disabled={filteredAnnotations.length === 0}
                  title={filteredAnnotations.length === 0 ? 'No annotations match the current filter' : 'Record a reel of the currently filtered annotations'}
                  style={{
                    fontSize: 11, fontWeight: 600, padding: '6px 10px', border: 'none',
                    background: recordMode === 'reel' ? '#0f2972' : '#fff',
                    color: filteredAnnotations.length === 0 ? '#C0C4CE' : recordMode === 'reel' ? '#fff' : '#8A8F9E',
                    cursor: filteredAnnotations.length === 0 ? 'default' : 'pointer',
                  }}
                >
                  Highlights reel
                </button>
              </div>
              {filteredAnnotations.length > 0 && (
                <label style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  fontSize: 11, color: '#8A8F9E', flexShrink: 0,
                }}>
                  <input
                    type="checkbox" checked={loopReel}
                    onChange={e => setLoopReel(e.target.checked)}
                    title="Automatically restart the highlights reel after it finishes"
                  />
                  Loop after
                  <input
                    type="number" min={0} max={120} value={loopDelaySec}
                    disabled={!loopReel}
                    onChange={e => setLoopDelaySec(Math.max(0, Number(e.target.value) || 0))}
                    style={{
                      width: 42, fontSize: 11, padding: '3px 5px',
                      border: '1px solid #E4E6EE', borderRadius: 5,
                      outline: 'none', fontFamily: 'DM Sans, sans-serif',
                      opacity: loopReel ? 1 : 0.5,
                    }}
                  />
                  sec
                </label>
              )}
              {filteredAnnotations.length > 0 && (
                <button
                  onClick={startReelForRecording}
                  title="Play the highlights reel now, without recording"
                  style={{
                    fontSize: 11, fontWeight: 600, padding: '6px 10px',
                    borderRadius: 8, border: '1px solid #E4E6EE',
                    background: reelAutoPlay ? '#22c55e' : '#fff',
                    color: reelAutoPlay ? '#fff' : '#0f2972',
                    cursor: 'pointer', flexShrink: 0,
                  }}
                >
                  {reelAutoPlay ? '▶ Playing reel...' : '▶ Play reel'}
                </button>
              )}
              <RecordExportButton
                onStart={recordMode === 'reel' ? startReelForRecording : () => seekTo(0)}
                isFinished={recordMode === 'reel' ? reelFinished : hasEnded}
                resetFinished={recordMode === 'reel' ? () => setReelFinished(false) : () => setHasEnded(false)}
                onEnterFocusMode={() => setFocusMode(true)}
                onExitFocusMode={() => setFocusMode(false)}
                captureRegionRef={videoWrapperRef}
              />
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
            </div>
          )}
          {readOnly && filteredAnnotations.length > 0 && (
            <button
              onClick={startReelForRecording}
              title="Play the highlights one after the other"
              style={{
                fontSize: 12, fontWeight: 600, padding: '7px 14px',
                borderRadius: 8, border: '1px solid #E4E6EE',
                background: reelAutoPlay ? '#22c55e' : '#fff',
                color: reelAutoPlay ? '#fff' : '#0f2972',
                cursor: 'pointer', flexShrink: 0,
              }}
            >
              {reelAutoPlay ? '▶ Playing reel...' : '▶ Play reel'}
            </button>
          )}
        </div>
        )}

        {showShare && (
          <ShareModal
            gameId={id}
            annotations={annotations}
            initialFilter={filter}
            onClose={() => setShowShare(false)}
          />
        )}

        {showVeoImport && isCoach && userId && (
          <VeoImportModal
            gameId={id}
            homeTeam={game.home_team}
            awayTeam={game.away_team}
            userId={userId}
            onClose={() => setShowVeoImport(false)}
            onImported={(count) => {
              setShowVeoImport(false)
              // Reload annotations to pick up the new batch
              supabase
                .from('annotations')
                .select('*, profiles(display_name)')
                .eq('game_id', id)
                .order('timestamp_sec')
                .then(({ data }) => { if (data) setAnnotations(data) })
            }}
          />
        )}

        <div style={{ display: 'grid', gridTemplateColumns: focusMode ? '1fr' : '1fr 300px', gap: 10 }}>
          {/* Left */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Video */}
            <div
              ref={videoWrapperRef}
              style={focusMode ? {
                background: '#000', position: 'fixed', inset: 0,
                width: '100vw', height: '100vh', zIndex: 9999,
              } : {
                background: '#091d52', borderRadius: 12,
                overflow: 'hidden', aspectRatio: '16/9', position: 'relative',
              }}
            >
              <iframe
                ref={playerRef}
                src={`${game.video_url}?enablejsapi=1&controls=0&rel=0&modestbranding=1&origin=${typeof window !== 'undefined' ? window.location.origin : ''}`}
                style={{ width: '100%', height: '100%', border: 'none' }}
                allowFullScreen
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              />
              {activeTacticalShapes.length > 0 && (
                <div style={{
                  position: 'absolute', inset: 0,
                  background: 'rgba(9, 29, 82, 0.4)', pointerEvents: 'none',
                }} />
              )}
              <AnnotationCanvas
                ref={desktopCanvasRef}
                shapes={activeTacticalShapes}
                editable={isDrawing}
                tool={drawTool}
                onAddShape={s => setDraftShapes(prev => [...prev, s])}
                onUpdateShape={(shapeId, patch) =>
                  setDraftShapes(prev => prev.map(s => s.id === shapeId ? { ...s, ...patch } as Shape : s))
                }
                onRemoveShape={shapeId => setDraftShapes(prev =>
                  prev.filter(s => s.id !== shapeId)
                    // cascade delete: drop any connector that referenced the removed shape
                    .filter(s => !(s.type === 'connector' && (s.fromId === shapeId || s.toId === shapeId)))
                )}
                onToggleVideo={() => (isPlaying ? pauseVideo() : resumeVideo())}
              />
              {boardOverlay}
              {boardState.status === 'ready' && !showBoard && (
                <button
                  onClick={() => setShowBoard(true)}
                  title="Show the tactics board on the video"
                  style={{
                    position: 'absolute', top: 8, right: 8, zIndex: 17,
                    fontSize: 12, fontWeight: 600, padding: '5px 10px', borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.3)', background: 'rgba(0,0,0,0.5)',
                    color: '#fff', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                  }}
                >
                  ⚽ Board
                </button>
              )}
              {isDrawing && (
                <FloatingDrawTools
                  tool={drawTool}
                  onChange={setDrawTool}
                  canUndo={draftShapes.length > 0}
                  shapeCount={draftShapes.length}
                  onUndo={() => setDraftShapes(prev => prev.slice(0, -1))}
                  onClear={() => setDraftShapes([])}
                  onDone={finishDrawing}
                />
              )}
            </div>

            {/* Match statistics: a card that opens under the video */}
            {!focusMode && statsState.status === 'ready' && (
              <StatsCard stats={statsState.stats} halfStart={halfStart} now={currentTime} onSeek={seekTo} />
            )}

            {/* Timeline card */}
            {!focusMode && (
            <div style={{
              background: '#fff', border: '1px solid #E4E6EE',
              borderRadius: 12, padding: '12px 14px',
            }}>
              {/* Time + hint */}
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: 8,
              }}>
                <TransportBar
                  isPlaying={isPlaying}
                  timeLabel={`${formatTime(currentTime)} / ${formatTime(duration)}`}
                  rate={mobilePlaybackRate}
                  onTogglePlay={() => (isPlaying ? pauseVideo() : resumeVideo())}
                  onSeekBy={seekBy}
                  onStep={stepFrame}
                  onRate={changeRate}
                />
                <span style={{ fontSize: 10, color: '#8A8F9E' }}>
                  {isCoach
                    ? markIn !== null ? `Mark in: ${formatTime(markIn)}` : 'Mark in/out for range'
                    : readOnly ? 'Shared highlights' : 'Pick a label then mark'}
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
              {!readOnly && (
              <div style={{ marginBottom: 10 }}>
                <LabelPicker />
              </div>
              )}

              {/* Draw tactical shapes on the current frame (coach, Tactical label only) */}
              {isCoach && selectedLabel === 'tactical' && (
                <div style={{ marginBottom: 8 }}>
                  {isDrawing ? (
                    <>
                      <div style={{ fontSize: 11, color: '#8A8F9E', padding: '2px 0 6px' }}>
                        Drawing tools are on the video (keys 1–8, Ctrl/⌘+Z undo, Esc = select).
                        <button onClick={finishDrawing} style={{
                          marginLeft: 8, fontSize: 11, fontWeight: 700, padding: '3px 10px', borderRadius: 99, border: 'none',
                          background: '#0f2972', color: '#fff', cursor: 'pointer',
                        }}>Done</button>
                      </div>
                      <div style={{
                        display: 'flex', alignItems: 'center', gap: 6,
                        fontSize: 11, color: '#8A8F9E', marginTop: 4,
                      }}>
                        <span>Auto-resume video after:</span>
                        <input
                          type="number" min={0} max={60} value={autoResumeSec}
                          onChange={e => setAutoResumeSec(Math.max(0, Number(e.target.value) || 0))}
                          style={{
                            width: 46, fontSize: 11, padding: '3px 6px',
                            border: '1px solid #E4E6EE', borderRadius: 5,
                            outline: 'none', fontFamily: 'DM Sans, sans-serif',
                          }}
                        />
                        <span>sec (0 = wait for manual play)</span>
                      </div>
                      {markIn !== null && (
                        <TimingStrip
                          timestampSec={markIn}
                          contextStart={contextStartSec ?? Math.max(0, markIn - 2)}
                          annotationEnd={annotationEndSec ?? (markIn + 1.5)}
                          clipEnd={clipEndSec ?? (markIn + 5)}
                          onChangeTimestamp={setMarkIn}
                          onChangeContextStart={setContextStartSec}
                          onChangeAnnotationEnd={setAnnotationEndSec}
                          onChangeClipEnd={setClipEndSec}
                          onScrub={scrubTo}
                        />
                      )}
                    </>
                  ) : (
                    <button
                      onClick={startDrawing}
                      style={{
                        width: '100%', fontSize: 11, fontWeight: 600,
                        padding: '6px 8px', borderRadius: 6,
                        border: '1.5px solid #E4E6EE', background: '#fff',
                        color: '#0f2972', cursor: 'pointer',
                      }}
                    >
                      {draftShapes.length > 0 ? 'Resume editing' : 'Add tactical drawing'}
                    </button>
                  )}
                </div>
              )}

              {/* Mark in/out (coach) or quick save (player) */}
              {!readOnly && (<>
              {isCoach ? (
                <>
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
                      onClick={() => {
                        mobileCanvasRef.current?.finalizePending()
                        desktopCanvasRef.current?.finalizePending()
                        handleRangeSave()
                      }}
                      disabled={markIn === null || saving}
                      style={{
                        flex: 1, fontSize: 11, fontWeight: 600,
                        padding: '6px 8px', borderRadius: 6, border: 'none',
                        background: markIn !== null ? '#0f2972' : '#E4E6EE',
                        color: markIn !== null ? '#fff' : '#8A8F9E',
                        cursor: markIn !== null ? 'pointer' : 'default',
                      }}
                    >
                      {editingAnnotationId ? 'Save changes' : 'Save annotation'}
                    </button>
                  </div>
                  {saveError && (
                    <div style={{
                      fontSize: 11, color: '#b91c1c', background: '#fef2f2',
                      border: '1px solid #fecaca', borderRadius: 6,
                      padding: '6px 10px', marginTop: 6,
                    }}>
                    Save failed: {saveError}
                  </div>
                  )}
                </>
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

              </>)}

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
            )}
          </div>

          {/* Right: tactical board + annotations */}
          {!focusMode && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <BoardPanel gameId={id} gameTitle={game.title} state={boardState} reload={reloadBoard} currentTime={currentTime} isCoach={isCoach} />
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
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  {isCoach && (
                    <button
                      onClick={() => exportGame(game.title, annotations, boardState.status === 'ready' ? boardState.board : null)}
                      title="Download the events and the board data as CSV files for analysis"
                      style={{ fontSize: 10, fontWeight: 600, padding: '3px 8px', borderRadius: 99, border: '1.5px solid #E4E6EE', background: '#fff', color: '#4A4F5C', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', whiteSpace: 'nowrap' }}
                    >Export</button>
                  )}
                  {isCoach && (
                    <button
                      onClick={() => setShowVeoImport(true)}
                      title="Import timestamps from a Veo highlights export"
                      style={{
                        fontSize: 10, fontWeight: 600,
                        padding: '3px 8px', borderRadius: 99,
                        border: '1.5px solid #E4E6EE',
                        background: '#fff', color: '#4A4F5C',
                        cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      + Veo import
                    </button>
                  )}
                  <span style={{
                    fontFamily: 'DM Sans, sans-serif',
                    fontSize: 10, fontWeight: 700,
                    padding: '2px 8px', borderRadius: 99,
                    background: '#e8edf8', color: '#0f2972',
                  }}>
                    {annotations.length}
                  </span>
                </div>
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
          )}
        </div>
      </div>
    </div>
  )
}
