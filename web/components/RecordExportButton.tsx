'use client'

// components/RecordExportButton.tsx
//
// Exports the annotated game by recording the browser tab itself, not by
// processing a file. This sidesteps the cross-origin YouTube-iframe problem
// entirely: getDisplayMedia() captures whatever the compositor has already
// rendered to the screen, which is unaffected by CORS/canvas-tainting rules
// that block drawImage()-based approaches.
//
// Flow: user clicks "Record & Export" → browser prompts to share this tab →
// video seeks to 0 and plays → your existing pause-on-annotation feature
// fires naturally during playback, exactly as a normal viewer would see it →
// on reaching the end, recording stops and downloads automatically.

import { useRef, useState, useCallback, useEffect } from 'react'

interface RecordExportButtonProps {
  /** Called to begin whatever playback should be recorded — e.g. seekTo(0), or a "play all" highlight-reel driver. */
  onStart: () => void
  /** True once the thing being recorded is done — native video end, or a highlight reel finishing its last clip. */
  isFinished: boolean
  /** Reset isFinished's underlying source back to false — called right before onStart, so a fresh finish can be detected next time. */
  resetFinished: () => void
  /**
   * Optional: crop the recording to just this element (e.g. the video + annotation
   * wrapper div) instead of capturing the whole tab. Uses the Element Capture API
   * (RestrictionTarget), which only works for self-capture (sharing "this tab") —
   * Chrome/Edge only as of writing. Falls back to full-tab capture silently if
   * unsupported, so this is always safe to pass.
   */
  captureRegionRef?: React.RefObject<HTMLElement | null>
}

function pickMimeType(): string {
  const candidates = [
    'video/mp4;codecs=avc1',       // Safari, and newer Chrome builds
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
  ]
  for (const c of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(c)) return c
  }
  return 'video/webm'
}

export default function RecordExportButton({ onStart, isFinished, resetFinished, captureRegionRef }: RecordExportButtonProps) {
  const [status, setStatus] = useState<'idle' | 'requesting' | 'recording' | 'finishing'>('idle')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const stopAndSave = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      setStatus('finishing')
      recorderRef.current.stop()
    }
  }, [])

  const startRecording = useCallback(async () => {
    setStatus('requesting')
    try {
      // preferCurrentTab is Chrome-only; falls back to the normal picker elsewhere.
      // cursor: 'never' asks the browser to omit the mouse pointer from captured frames
      // (support varies by OS/browser — best-effort, not guaranteed everywhere).
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: 'browser', cursor: 'never' } as MediaTrackConstraints,
        audio: true,
        // @ts-expect-error — Chrome-specific, not yet in lib.dom.d.ts
        preferCurrentTab: true,
      })
      streamRef.current = stream

      // Crop to a specific element if requested and the browser supports it
      // (Element Capture — Chrome/Edge, self-capture only). Best-effort: if this
      // fails or isn't supported, we silently keep the full-tab recording.
      const [track] = stream.getVideoTracks()
      if (captureRegionRef?.current) {
        try {
          const RestrictionTargetCtor = (window as any).RestrictionTarget
          if (RestrictionTargetCtor?.fromElement && typeof (track as any).restrictTo === 'function') {
            const target = await RestrictionTargetCtor.fromElement(captureRegionRef.current)
            await (track as any).restrictTo(target)
          } else {
            // Older Chrome versions shipped the same capability under CropTarget/cropTo
            const CropTargetCtor = (window as any).CropTarget
            if (CropTargetCtor?.fromElement && typeof (track as any).cropTo === 'function') {
              const target = await CropTargetCtor.fromElement(captureRegionRef.current)
              await (track as any).cropTo(target)
            }
          }
        } catch (cropErr) {
          console.warn('Could not restrict recording to the video region — recording full tab instead:', cropErr)
        }
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
      setStatus('idle')
    }
  }, [onStart, resetFinished, stopAndSave, captureRegionRef])

  // Auto-stop the moment the tracked playback (full video or highlight reel) finishes.
  useEffect(() => {
    if (isFinished && status === 'recording') {
      stopAndSave()
    }
  }, [isFinished, status, stopAndSave])

  return (
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
        {status === 'requesting' && 'Choose this tab to share…'}
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
  )
}
