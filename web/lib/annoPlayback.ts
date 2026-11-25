// lib/annoPlayback.ts
// The clock a drawing runs on, for the viewer and for the coach's editor.
//
// τ (tau) = seconds on the drawing's own clock. One rule covers every case:
//
//     while the video is frozen at the anchor:   τ = real seconds since the freeze began   (0 … F)
//     afterwards:                                 τ = F + (video time − anchor)
//
// F is the length of the freeze. F = 0 and no freeze at all is "live" (shapes move over the playing video).
// The freeze length is stored in the existing `auto_resume_sec` column, so nothing needs migrating:
//     > 0  freeze for that many seconds, then play on
//     0    freeze until the viewer presses play (an animated drawing plays out once first)
//     < 0  no freeze: the drawing animates over the playing video
import { useCallback, useEffect, useRef } from 'react'
import type { Shape } from '@/components/AnnotationCanvas'
import { isAnimated, spanEnd } from '@/lib/annoAnim'

export const LIVE_SEC = -1

export type Hold = { mode: 'freeze'; sec: number } | { mode: 'live' }

export function holdOf(autoResumeSec: number | null | undefined): Hold {
  const v = autoResumeSec ?? 0
  return v < 0 ? { mode: 'live' } : { mode: 'freeze', sec: v }
}
/** Stored as whole seconds: the column's type is not under this app's control, and a freeze of 2.5 s vs 3 s hardly matters. */
export function holdToColumn(h: Hold): number { return h.mode === 'live' ? LIVE_SEC : Math.max(0, Math.round(h.sec)) }

/** Length of the frozen phase on the drawing clock (a manual freeze lasts as long as the animation does). */
export function freezeLength(h: Hold, shapes: Shape[]): number {
  if (h.mode === 'live') return 0
  if (h.sec > 0) return h.sec
  return isAnimated(shapes) ? spanEnd(shapes) : 0
}

/** When the drawing stops showing: the coach's "removed" pin, or 1.5 s after the anchor for older drawings. */
export function annoEnd(a: { timestamp_sec: number; annotation_end_sec?: number | null }): number {
  return a.annotation_end_sec ?? a.timestamp_sec + 1.5
}

/** The whole timeline in τ seconds: the freeze, then the video running on until the drawing is removed. */
export function totalLength(F: number, shapes: Shape[], anchor: number, end: number): number {
  return Math.max(0.5, F + Math.max(0, end - anchor), spanEnd(shapes))
}

// ---- smooth video time ------------------------------------------------------------------------

/**
 * The YouTube iframe reports its time only ~4× a second; animation needs more. This returns a getter that
 * extrapolates from the last report using the wall clock. It reads refs, so it never causes a re-render.
 */
export function useSmoothVideoTime(videoTime: number, isPlaying: boolean, rate: number): () => number {
  const ref = useRef({ t: videoTime, ms: 0, playing: isPlaying, rate })
  useEffect(() => { ref.current = { t: videoTime, ms: performance.now(), playing: isPlaying, rate } }, [videoTime, isPlaying, rate])
  return useCallback(() => {
    const r = ref.current
    return r.playing ? r.t + ((performance.now() - r.ms) / 1000) * r.rate : r.t
  }, [])
}

// ---- viewer ------------------------------------------------------------------------------------

export type Holding = { id: string; startMs: number } | null

export type ActiveClockInfo = { id: string; anchor: number; F: number }

/** τ for the drawing that is on screen while somebody watches. `holding` is set only while the app itself has frozen the video for this drawing. */
export function viewerTau(info: ActiveClockInfo, holding: Holding, video: number, nowMs = performance.now()): number {
  if (holding && holding.id === info.id) return Math.min(info.F, (nowMs - holding.startMs) / 1000)
  return info.F + Math.max(0, video - info.anchor)
}

/** A stable getter for the canvas to call every frame. */
export function useViewerClock(info: ActiveClockInfo | null, holdingRef: { current: Holding }, getVideo: () => number): () => number {
  const infoRef = useRef(info)
  useEffect(() => { infoRef.current = info }, [info])
  return useCallback(() => {
    const i = infoRef.current
    return i ? viewerTau(i, holdingRef.current, getVideo()) : 0
  }, [getVideo, holdingRef])
}

// ---- editor ------------------------------------------------------------------------------------

export type ClockEnv = {
  anchor: number                         // video time the drawing is anchored to
  F: number                              // freeze length (τ seconds)
  total: number                          // where the drawing ends (τ seconds)
  maxScrub: number                       // how far the playhead may be dragged: past `total` too, so a keyframe can extend the drawing
  getVideo: () => number                 // smooth video time
  seekPaused: (videoSec: number) => void // show that video frame and stay paused
  seekPlay: (videoSec: number) => void   // jump there and play
  pause: () => void
}

/**
 * The coach's playhead. A small external store so scrubbing and previewing re-render only the timeline,
 * not the whole game view; the canvas reads `get()` every frame.
 *
 *   playhead inside the freeze  → the video sits on the anchor frame
 *   playhead after the freeze   → the video shows anchor + (τ − F), so what you see matches what viewers will see
 *   Play                        → runs the freeze on the wall clock, then plays the video on from the anchor
 */
export class EditorClock {
  private env: ClockEnv | null = null
  private tau = 0
  private playing = false
  private phase: 'hold' | 'video' = 'hold'
  private startMs = 0
  private startTau = 0
  private version = 0
  private subs = new Set<() => void>()

  configure(env: ClockEnv) { this.env = env }
  subscribe = (fn: () => void) => { this.subs.add(fn); return () => { this.subs.delete(fn) } }
  getVersion = () => this.version
  private emit() { this.version++; this.subs.forEach(f => f()) }

  isPlaying() { return this.playing }

  /** τ now. While previewing it advances by itself; the canvas and the timeline both call this every frame. */
  get = (): number => {
    const e = this.env
    if (!this.playing || !e) return this.tau
    if (this.phase === 'hold') {
      this.tau = this.startTau + (performance.now() - this.startMs) / 1000
      if (this.tau >= e.F) { this.tau = e.F; this.phase = 'video'; e.seekPlay(e.anchor) }
    }
    if (this.phase === 'video') {
      this.tau = Math.max(this.tau, e.F + Math.max(0, e.getVideo() - e.anchor))
    }
    if (this.tau >= e.total) { this.tau = e.total; this.stop() }
    return this.tau
  }

  /** Move the playhead (clamped to the timeline) and show the matching video frame. */
  scrub(tau: number) {
    const e = this.env
    if (!e) return
    if (this.playing) { this.playing = false; e.pause() }
    this.tau = Math.min(e.maxScrub, Math.max(0, tau))
    this.seekThrottled()
    this.emit()
  }

  // Dragging the playhead would otherwise send YouTube a seek for every mouse move. At most one per 120 ms, and always one for the final position.
  private lastSeek = 0
  private seekTimer: ReturnType<typeof setTimeout> | null = null
  private seekNow() {
    const e = this.env
    if (!e) return
    this.lastSeek = performance.now()
    e.seekPaused(e.anchor + Math.max(0, this.tau - e.F))
  }
  private seekThrottled() {
    if (this.seekTimer) clearTimeout(this.seekTimer)
    const wait = 120 - (performance.now() - this.lastSeek)
    if (wait <= 0) { this.seekTimer = null; this.seekNow() }
    else this.seekTimer = setTimeout(() => { this.seekTimer = null; this.seekNow() }, wait)
  }

  start() {
    const e = this.env
    if (!e || this.playing) return
    if (this.tau >= e.total - 0.05) this.tau = 0
    this.playing = true
    this.startMs = performance.now()
    this.startTau = this.tau
    if (this.tau < e.F) { this.phase = 'hold'; e.seekPaused(e.anchor) }
    else { this.phase = 'video'; e.seekPlay(e.anchor + (this.tau - e.F)) }
    this.emit()
  }

  stop() {
    if (!this.playing) return
    this.playing = false
    this.env?.pause()
    this.emit()
  }

  toggle() {
    if (this.playing) this.stop()
    else this.start()
  }

  /** New drawing, or the old one closed. */
  reset(tau = 0) { this.playing = false; this.tau = tau; this.emit() }
}
