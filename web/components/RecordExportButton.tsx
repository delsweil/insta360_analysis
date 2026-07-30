'use client'

// components/RecordExportButton.tsx
//
// Exports the annotated game by recording the browser tab itself, not by
// processing a file. This sidesteps the cross-origin YouTube-iframe problem
// entirely: getDisplayMedia() captures whatever the compositor has already
// rendered to the screen, which is unaffected by CORS/canvas-tainting rules
// that block drawImage()-based approaches.
//
// Cropping strategy: rather than relying on Chrome-only APIs (Element Capture,
// preferCurrentTab), the capture target is put into the standard Fullscreen
// API before recording starts — supported everywhere, including Safari and
// Firefox. If the video+annotations element fills the whole screen, there's
// nothing else on screen to accidentally capture, no cropping needed after
// the fact. The mouse cursor is hidden via CSS `cursor: none` on that same
// fullscreened element, which suppresses the OS pointer icon while hovering
// it — also universal, unlike the Chrome-only `cursor: 'never'` constraint.
// The old crop-after-capture approach (Element Capture) is kept as a fallback
// for the rare case fullscreen itself isn't available.
//
// Flow: user clicks "Record & Export" → target element goes fullscreen →
// browser prompts to share a screen/window (now showing only that element) →
// video seeks to 0 and plays → your existing pause-on-annotation feature
// fires naturally during playback, exactly as a normal viewer would see it →
// on reaching the end, recording stops, fullscreen exits, and the file downloads.

import { useRef, useState, useCallback, useEffect } from 'react'

interface RecordExportButtonProps {
  /** Called to begin whatever playback should be recorded — e.g. seekTo(0), or a "play all" highlight-reel driver. */
  onStart: () => void
  /** True once the thing being recorded is done — native video end, or a highlight reel finishing its last clip. */
  isFinished: boolean
  /** Reset isFinished's underlying source back to false — called right before onStart, so a fresh finish can be detected next time. */
  resetFinished: () => void
  /**
   * The element to fullscreen and record — typically the video + annotation
   * wrapper div. Required for the fullscreen-crop strategy; if fullscreening
   * fails or is unsupported, falls back to Chrome-only Element Capture
   * cropping on the same element, then to full-tab capture as a last resort.
   */
  captureRegionRef?: React.RefObject<HTMLElement | null>
}

type ShareWarning = 'not-this-tab' | 'crop-unsupported' | 'no-fullscreen' | null

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

function requestFs(el: HTMLElement): Promise<void> {
  const anyEl = el as any
  const fn = el.requestFullscreen || anyEl.webkitRequestFullscreen || anyEl.mozRequestFullScreen
  if (!fn) return Promise.reject(new Error('Fullscreen API not supported'))
  return fn.call(el)
}

function exitFs(): Promise<void> {
  const anyDoc = document as any
  const fn = document.exitFullscreen || anyDoc.webkitExitFullscreen || anyDoc.mozCancelFullScreen
  if (!fn || !(document.fullscreenElement || anyDoc.webkitFullscreenElement)) return Promise.resolve()
  return fn.call(document).catch(() => {})
}

export default function RecordExportButton({ onStart, isFinished, resetFinished, captureRegionRef }: RecordExportButtonProps) {
  const [status, setStatus] = useState<'idle' | 'requesting' | 'recording' | 'finishing'>('idle')
  const [warning, setWarning] = useState<ShareWarning>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const prevCursorRef = useRef<string>('')

  const restoreCursor = useCallback(() => {
    if (captureRegionRef?.current) captureRegionRef.current.style.cursor = prevCursorRef.current
  }, [captureRegionRef])

  const stopAndSave = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      setStatus('finishing')
      recorderRef.current.stop()
    }
  }, [])

  const startRecording = useCallback(async () => {
    setStatus('requesting')
    setWarning(null)
    let fullscreened = false

    console.log('[RecordExportButton] start clicked. captureRegionRef element:', captureRegionRef?.current)

    try {
      // 1. Fullscreen the capture target first, so there's nothing else on
      // screen for the user to accidentally share — works in every browser.
      if (captureRegionRef?.current) {
        try {
          console.log('[RecordExportButton] attempting requestFullscreen()...')
          await requestFs(captureRegionRef.current)
          fullscreened = true
          console.log('[RecordExportButton] fullscreen succeeded')
          prevCursorRef.current = captureRegionRef.current.style.cursor
          captureRegionRef.current.style.cursor = 'none' // hide the pointer while over the recorded area
        } catch (err) {
          console.warn('[RecordExportButton] Fullscreen request failed, falling back to Element Capture cropping:', err)
          setWarning('no-fullscreen')
        }
      } else {
        console.warn('[RecordExportButton] No captureRegionRef element — skipping fullscreen entirely. This prop may not be wired up.')
      }

      // preferCurrentTab is Chrome-only; falls back to the normal picker elsewhere.
      // cursor: 'never' is a Chrome-only best-effort hint; the CSS cursor:none
      // above is what actually does the work cross-browser once fullscreened.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser', cursor: 'never' } as MediaTrackConstraints,
        audio: true,
        // @ts-expect-error — Chrome-specific, not yet in lib.dom.d.ts
        preferCurrentTab: true,
      })
      streamRef.current = stream

      const [track] = stream.getVideoTracks()
      const settings = track.getSettings() as MediaTrackSettings & { displaySurface?: string }

      if (!fullscreened && settings.displaySurface && settings.displaySurface !== 'browser') {
        // Only relevant when fullscreen wasn't available — otherwise the whole
        // screen legitimately IS just the recorded element regardless of surface type.
        setWarning('not-this-tab')
      }

      // Fallback cropping (Chrome/Edge only) — only attempted if fullscreen
      // didn't happen, since fullscreen already solves the same problem
      // more reliably and cross-browser.
      if (!fullscreened && captureRegionRef?.current && settings.displaySurface === 'browser') {
        let cropped = false
        try {
          const RestrictionTargetCtor = (window as any).RestrictionTarget
          if (RestrictionTargetCtor?.fromElement && typeof (track as any).restrictTo === 'function') {
            const target = await RestrictionTargetCtor.fromElement(captureRegionRef.current)
            await (track as any).restrictTo(target)
            cropped = true
          }
        } catch (err) {
          console.warn('RestrictionTarget crop failed, trying CropTarget fallback:', err)
        }
        if (!cropped) {
          try {
            const CropTargetCtor = (window as any).CropTarget
            if (CropTargetCtor?.fromElement && typeof (track as any).cropTo === 'function') {
              const target = await CropTargetCtor.fromElement(captureRegionRef.current)
              await (track as any).cropTo(target)
              cropped = true
            }
          } catch (err) {
            console.warn('CropTarget crop also failed:', err)
          }
        }
        if (!cropped) setWarning('crop-unsupported')
      }

      if (track.readyState !== 'live') {
        console.error('Video track is not live after setup — recording would be audio-only. Aborting.')
        stream.getTracks().forEach(t => t.stop())
        restoreCursor()
        await exitFs()
        setStatus('idle')
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
        restoreCursor()
        exitFs()
        setStatus('idle')
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
      restoreCursor()
      if (fullscreened) await exitFs()
      setStatus('idle')
    }
  }, [onStart, resetFinished, stopAndSave, captureRegionRef, restoreCursor])

  // If the user exits fullscreen manually (Esc key) mid-recording, stop and save
  // rather than continuing to record whatever's now visible outside fullscreen.
  useEffect(() => {
    const handler = () => {
      const anyDoc = document as any
      const stillFullscreen = document.fullscreenElement || anyDoc.webkitFullscreenElement
      if (!stillFullscreen && status === 'recording') stopAndSave()
    }
    document.addEventListener('fullscreenchange', handler)
    document.addEventListener('webkitfullscreenchange', handler)
    return () => {
      document.removeEventListener('fullscreenchange', handler)
      document.removeEventListener('webkitfullscreenchange', handler)
    }
  }, [status, stopAndSave])

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
      {warning === 'no-fullscreen' && (
        <div style={{
          fontSize: 11, color: '#b45309', background: '#fffbeb',
          border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px',
          maxWidth: 320,
        }}>
          Couldn't enter fullscreen for a clean recording — falling back to
          cropping, which works best in Chrome or Edge.
        </div>
      )}
      {warning === 'not-this-tab' && (
        <div style={{
          fontSize: 11, color: '#b45309', background: '#fffbeb',
          border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px',
          maxWidth: 320,
        }}>
          You shared the whole screen/window instead of this tab — the video won't be
          cropped and the cursor may still show. Stop, then try again and pick
          <strong> "Chrome Tab" → this tab</strong> in the share dialog.
        </div>
      )}
      {warning === 'crop-unsupported' && (
        <div style={{
          fontSize: 11, color: '#b45309', background: '#fffbeb',
          border: '1px solid #fde68a', borderRadius: 6, padding: '6px 10px',
          maxWidth: 320,
        }}>
          Couldn't crop to just the video area in this browser — recording the full tab
          instead. Cropping needs a recent Chrome or Edge.
        </div>
      )}
    </div>
  )
}
