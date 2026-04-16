'use client'

// components/ClipRecorder.tsx
// Mobile clip recording with auto-save, extend/shorten controls

import { useEffect, useState, useRef } from 'react'

const AUTO_SAVE_SEC = 30      // auto-save after this many seconds
const WARNING_SEC = 5         // show countdown in last N seconds

interface Props {
  currentTime: number
  onSave: (startTime: number, endTime: number) => void
  onCancel: () => void
  saving: boolean
  saved: boolean
}

export default function ClipRecorder({ currentTime, onSave, onCancel, saving, saved }: Props) {
  const [markIn, setMarkIn] = useState<number>(currentTime)
  const [offset, setOffset] = useState(0)          // manual +/- adjustment in seconds
  const [elapsed, setElapsed] = useState(0)         // seconds since mark in
  const startWallTime = useRef(Date.now())

  // Count up elapsed wall-clock time
  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startWallTime.current) / 1000))
    }, 250)
    return () => clearInterval(interval)
  }, [])

  // Auto-save when time runs out
  useEffect(() => {
    if (elapsed >= AUTO_SAVE_SEC && !saving && !saved) {
      handleSave()
    }
  }, [elapsed])

  const clipDuration = elapsed + offset
  const endTime = markIn + clipDuration
  const timeLeft = AUTO_SAVE_SEC - elapsed
  const isWarning = timeLeft <= WARNING_SEC && timeLeft > 0

  const handleSave = () => {
    onSave(markIn, Math.max(markIn + 1, endTime))
  }

  const formatDur = (sec: number) => {
    const s = Math.max(0, Math.round(sec))
    return `${s}s`
  }

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60)
    const s = Math.floor(sec % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div style={{
      padding: '0 14px 20px',
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      {/* Clip progress bar */}
      <div style={{
        background: '#F8F8F6', borderRadius: 8,
        border: '1px solid #E4E6EE', overflow: 'hidden',
        height: 6,
      }}>
        <div style={{
          height: '100%',
          width: `${Math.min(100, (clipDuration / AUTO_SAVE_SEC) * 100)}%`,
          background: isWarning ? '#ef4444' : '#E8780A',
          borderRadius: 8,
          transition: 'width 0.25s, background 0.3s',
        }} />
      </div>

      {/* Status row */}
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
        background: isWarning ? '#fef2f2' : '#FEF0E0',
        borderRadius: 10, padding: '10px 14px',
        border: `1px solid ${isWarning ? '#fca5a5' : '#fdba74'}`,
        transition: 'all 0.3s',
      }}>
        <div>
          <div style={{
            fontSize: 13, fontWeight: 700,
            color: isWarning ? '#ef4444' : '#E8780A',
          }}>
            {saved ? '✓ Clip saved' : saving ? 'Saving...' : `Recording — ${formatDur(clipDuration)}`}
          </div>
          <div style={{ fontSize: 10, color: '#8A8F9E', marginTop: 1 }}>
            {formatTime(markIn)} → {formatTime(endTime)}
            {isWarning && !saving && !saved && (
              <span style={{ color: '#ef4444', fontWeight: 600, marginLeft: 6 }}>
                Auto-saves in {timeLeft}s
              </span>
            )}
          </div>
        </div>
        <button
          onClick={onCancel}
          style={{
            fontSize: 11, color: '#8A8F9E',
            background: 'none', border: 'none',
            cursor: 'pointer', padding: '4px 6px',
          }}
        >
          Cancel
        </button>
      </div>

      {/* Adjust + save buttons */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={() => setOffset(o => o - 5)}
          disabled={clipDuration <= 2}
          style={{
            width: 52, height: 52,
            fontSize: 13, fontWeight: 700,
            borderRadius: 10, border: '1.5px solid #E4E6EE',
            background: '#fff', color: clipDuration > 2 ? '#4A4F5C' : '#C0C4CE',
            cursor: clipDuration > 2 ? 'pointer' : 'default',
            fontFamily: 'DM Sans, sans-serif', flexShrink: 0,
          }}
        >
          −5s
        </button>

        <button
          onClick={handleSave}
          disabled={saving || saved}
          style={{
            flex: 1, height: 52,
            fontSize: 15, fontWeight: 700,
            borderRadius: 10, border: 'none',
            background: saved ? '#22c55e' : saving ? '#E4E6EE' : '#0f2972',
            color: saving ? '#8A8F9E' : '#fff',
            cursor: saving || saved ? 'default' : 'pointer',
            fontFamily: 'DM Sans, sans-serif',
            transition: 'background 0.2s',
          }}
        >
          {saved ? '✓ Saved' : saving ? 'Saving...' : 'End clip'}
        </button>

        <button
          onClick={() => setOffset(o => o + 5)}
          style={{
            width: 52, height: 52,
            fontSize: 13, fontWeight: 700,
            borderRadius: 10, border: '1.5px solid #E4E6EE',
            background: '#fff', color: '#4A4F5C',
            cursor: 'pointer',
            fontFamily: 'DM Sans, sans-serif', flexShrink: 0,
          }}
        >
          +5s
        </button>
      </div>
    </div>
  )
}
