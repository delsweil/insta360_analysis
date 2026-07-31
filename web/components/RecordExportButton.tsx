'use client'

// components/RecordExportButton.tsx
//
// Exports the annotated game by recording the browser tab itself, not by
// processing a file. This sidesteps the cross-origin YouTube-iframe problem
// entirely: getDisplayMedia() captures whatever the compositor has already
// rendered to the screen, which is unaffected by CORS/canvas-tainting rules
// that block drawImage()-based approaches.
//
// Cropping strategy: the native Fullscreen API was tried first (see git
// history) but doesn't survive contact with getDisplayMedia — browsers
// forcibly exit any active fullscreen the instant the "share your screen"
// permission prompt appears, as an anti-spoofing measure, so by the time
// recording actually starts the page is back to its normal, un-fullscreened
// layout. Instead, the parent page owns a "focus mode" — its own CSS-only
// state that hides everything except the video+annotations region and
// expands that region to fill the viewport. Nothing about it depends on the
// browser's fullscreen state, so no permission prompt can interrupt it.
//
// Flow: user clicks "Record & Export" → onEnterFocusMode() hides the rest of
// the page → browser prompts to share a screen/window/tab (now showing only
// the focused video region, regardless of what's picked) → video seeks to 0
// and plays → your existing pause-on-annotation feature fires naturally
// during playback → on reaching the end, recording stops, focus mode exits,
// and the file downloads.

import { useRef, useState, useCallback, useEffect } from 'react'

interface RecordExportButtonProps {
  /** Called to begin whatever playback should be recorded — e.g. seekTo(0), or a "play all" highlight-reel driver. */
  onStart: () => void
  /** True once the thing being recorded is done — native video end, or a highlight reel finishing its last clip. */
  isFinished: boolean
  /** Reset isFinished's underlying source back to false — called right before onStart, so a fresh finish can be detected next time. */
  resetFinished: () => void
  /** Hide surrounding page chrome and expand the video region to fill the viewport, via the parent's own CSS/state — not the browser Fullscreen API. */
  onEnterFocusMode: () => void
  /** Restore the normal page layout once recording stops. */
  onExitFocusMode: () => void
  /**
   * Optional: the video+annotations element, used only to apply CSS
   * `cursor: none` while recording (hides the pointer icon from captured
   * frames) and, on Chrome/Edge self-capture, as a secondary Element Capture
   * crop target — belt-and-braces on top of focus mode, not load-bearing.
   */
  captureRegionRef?: React.RefObject<HTMLElement | null>
}

type ShareWarning = 'not-this-tab' | null

function pickMimeType(): string {
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    // MP4 recording in MediaRecorder is newer and less battle-tested than WebM —
    // some Chrome builds report support via isTypeSupported() but then silently
    // fail to encode the video track (audio still works), so it's kept as a
    // last resort rather than the first choice.
    'video/mp4;codecs=avc1',
  ]
  for (const c of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(c)) return c
  }
  return 'video/webm'
}

export default function RecordExportButton({
  onStart, isFinished, resetFinished, onEnterFocusMode, onExitFocusMode, captureRegionRef,
}: RecordExportButtonProps) {
  const [status, setStatus] = useState<'idle' | 'requesting' | 'recording' | 'finishing'>('idle')
  const [warning, setWarning] = useState<ShareWarning>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const prevCursorRef = useRef<string>('')

  const restoreCursor = useCallback(() => {
    if (captureRegionRef?.current) captureRegionRef.current.style.cursor = prevCursorRef.current
  }, [captureRegionRef])

  const cleanup = useCallback(() => {
    restoreCursor()
    onExitFocusMode()
    setStatus('idle')
  }, [restoreCursor, onExitFocusMode])

  const stopAndSave = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      setStatus('finishing')
      recorderRef.current.stop()
    }
  }, [])

  const startRecording = useCallback(async () => {
    setStatus('requesting')
    setWarning(null)

    // Hide everything except the video region first — this is a plain state
    // change in the parent, synchronous and unaffected by anything the
    // browser does with fullscreen/permission prompts afterward.
    onEnterFocusMode()

    if (captureRegionRef?.current) {
      prevCursorRef.current = captureRegionRef.current.style.cursor
      captureRegionRef.current.style.cursor = 'none'
    }

    try {
      // preferCurrentTab is Chrome-only; falls back to the normal picker elsewhere.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser', cursor: 'never' } as MediaTrackConstraints,
        audio: true,
        // @ts-expect-error — Chrome-specific, not yet in lib.dom.d.ts
        preferCurrentTab: true,
      })
      streamRef.current = stream

      const [track] = stream.getVideoTracks()
      const settings = track.getSettings() as MediaTrackSettings & { displaySurface?: string }

      if (settings.displaySurface && settings.displaySurface !== 'browser') {
        // User picked "Entire Screen" or "Window" — other monitors/windows
        // could leak into the recording even with focus mode active on this tab.
        setWarning('not-this-tab')
      }

      if (track.readyState !== 'live') {
        console.error('Video track is not live after setup — recording would be audio-only. Aborting.')
        stream.getTracks().forEach(t => t.stop())
        cleanup()
        return
      }

      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(stream, { mimeType })
      chunksRef.current = []

      recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const url = URL.createObjectURL(blob)
        const ext = mimeType.includes('mp4') ? 'mp4' : 'webm'
        const a = document.createElement('a')
        a.href = url
        a.download = `game-export.${ext}`
        a.click()
        URL.revokeObjectURL(url)
        streamRef.current?.getTracks().forEach(t => t.stop())
        cleanup()
      }

      // If the user stops sharing via the browser's own "Stop sharing" control,
      // treat that the same as clicking stop ourselves.
      track.addEventListener('ended', stopAndSave)

      recorderRef.current = recorder
      recorder.start()
      setStatus('recording')

      resetFinished()
      onStart() // begin whatever playback should be captured — full playthrough or a highlight-reel sequence
    } catch (err) {
      console.error('Screen share was cancelled or failed:', err)
      cleanup()
    }
  }, [onStart, resetFinished, stopAndSave, captureRegionRef, onEnterFocusMode, cleanup])

  // Auto-stop the moment the tracked playback (full video or highlight reel) finishes.
  useEffect(() => {
    if (isFinished && status === 'recording') {
      stopAndSave()
    }
  }, [isFinished, status, stopAndSave])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button
          onClick={status === 'idle' ? startRecording : undefined}
          disabled={status !== 'idle'}
          style={{
            fontSize: 12, fontWeight: 600,
            padding: '7px 14px', borderRadius: 8, border: 'none',
            background: status === 'idle' ? '#0f2972' : '#E4E6EE',
            color: status === 'idle' ? '#fff' : '#8A8F9E',
            cursor: status === 'idle' ? 'pointer' : 'default',
          }}
        >
          {status === 'idle' && '● Record & Export'}
          {status === 'requesting' && 'Choose "This Tab" when prompted…'}
          {status === 'recording' && '● Recording — will save at end of video'}
          {status === 'finishing' && 'Saving…'}
        </button>
        {status === 'recording' && (
          <button
            onClick={stopAndSave}
            style={{ fontSize: 12, color: '#b91c1c', background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Stop now
          </button>
        )}
      </div>
      {warning === 'not-this-tab' && (
        <div style={{
          fontSize: 11, color: '#b45309', background: '#fffbeb',
          border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px',
          maxWidth: 320,
        }}>
          You shared the whole screen/window instead of this tab — other windows
          could show up in the recording. Stop, then try again and pick
          <strong> "Chrome Tab" → this tab</strong> in the share dialog.
        </div>
      )}
    </div>
  )
}
