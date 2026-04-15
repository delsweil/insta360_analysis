'use client'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'
import type { Annotation } from '@/lib/supabase'

const LABELS = [
  { key: 'goal',     label: 'Goal',     color: '#ef4444' },
  { key: 'shot',     label: 'Shot',     color: '#E8780A' },
  { key: 'chance',   label: 'Chance',   color: '#22c55e' },
  { key: 'tactical', label: 'Tactical', color: '#0f2972' },
  { key: 'other',    label: 'Other',    color: '#8A8F9E' },
]

function formatTime(sec: number) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function labelColor(label: string) {
  return LABELS.find(l => l.key === label)?.color ?? '#8A8F9E'
}

interface Props {
  gameId: string
  annotations: Annotation[]
  onClose: () => void
}

export default function ShareModal({ gameId, annotations, onClose }: Props) {
  const [selected, setSelected] = useState<Set<string>>(
    new Set(annotations.map(a => a.id))
  )
  const [generating, setGenerating] = useState(false)
  const [shareUrl, setShareUrl] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const toggleAll = () => {
    if (selected.size === annotations.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(annotations.map(a => a.id)))
    }
  }

  const toggleOne = (id: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleGenerate = async () => {
    if (selected.size === 0) return
    setGenerating(true)

    const { data, error } = await supabase
      .from('share_links')
      .insert({
        game_id: gameId,
        annotation_ids: Array.from(selected),
      })
      .select('token')
      .single()

    if (data?.token) {
      const url = `${window.location.origin}/share/${data.token}`
      setShareUrl(url)
    }
    setGenerating(false)
  }

  const handleCopy = async () => {
    if (!shareUrl) return
    await navigator.clipboard.writeText(shareUrl)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(15,41,114,0.5)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 20,
    }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div style={{
        background: '#fff', borderRadius: 16,
        width: '100%', maxWidth: 520,
        maxHeight: '80vh', display: 'flex', flexDirection: 'column',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid #E4E6EE',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          flexShrink: 0,
        }}>
          <div>
            <div style={{
              fontFamily: 'Bebas Neue, sans-serif',
              fontSize: 20, color: '#0f2972', letterSpacing: '0.04em',
            }}>
              Share Highlights
            </div>
            <div style={{ fontSize: 11, color: '#8A8F9E', marginTop: 2 }}>
              Select annotations to include in the share link
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              fontSize: 18, color: '#8A8F9E',
              background: 'none', border: 'none', cursor: 'pointer',
            }}
          >
            ✕
          </button>
        </div>

        {/* Select all */}
        <div style={{
          padding: '10px 20px',
          borderBottom: '1px solid #E4E6EE',
          display: 'flex', alignItems: 'center', gap: 10,
          flexShrink: 0,
        }}>
          <input
            type="checkbox"
            checked={selected.size === annotations.length}
            onChange={toggleAll}
            style={{ cursor: 'pointer', width: 16, height: 16 }}
          />
          <span style={{ fontSize: 13, color: '#4A4F5C' }}>
            Select all ({selected.size} / {annotations.length} selected)
          </span>
        </div>

        {/* Annotations list */}
        <div style={{ overflowY: 'auto', flex: 1, padding: '8px 20px' }}>
          {annotations.map(ann => (
            <div
              key={ann.id}
              onClick={() => toggleOne(ann.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 0',
                borderBottom: '1px solid #F8F8F6',
                cursor: 'pointer',
              }}
            >
              <input
                type="checkbox"
                checked={selected.has(ann.id)}
                onChange={() => toggleOne(ann.id)}
                onClick={e => e.stopPropagation()}
                style={{ cursor: 'pointer', width: 16, height: 16, flexShrink: 0 }}
              />
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
              <span style={{ fontSize: 13, color: '#111318', flex: 1 }}>
                {LABELS.find(l => l.key === ann.label)?.label ?? ann.label}
              </span>
              {ann.end_timestamp_sec && (
                <span style={{ fontSize: 10, color: '#8A8F9E', flexShrink: 0 }}>
                  {formatTime(ann.end_timestamp_sec - ann.timestamp_sec)}s clip
                </span>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid #E4E6EE',
          flexShrink: 0,
        }}>
          {shareUrl ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{
                background: '#F8F8F6', borderRadius: 8,
                padding: '10px 14px',
                fontSize: 12, color: '#4A4F5C',
                wordBreak: 'break-all',
                border: '1px solid #E4E6EE',
              }}>
                {shareUrl}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={handleCopy}
                  style={{
                    flex: 1, padding: '10px',
                    fontSize: 13, fontWeight: 600,
                    borderRadius: 8, border: 'none',
                    background: copied ? '#22c55e' : '#0f2972',
                    color: '#fff', cursor: 'pointer',
                    fontFamily: 'DM Sans, sans-serif',
                    transition: 'background 0.2s',
                  }}
                >
                  {copied ? '✓ Copied!' : 'Copy link'}
                </button>
                <button
                  onClick={() => setShareUrl(null)}
                  style={{
                    padding: '10px 16px',
                    fontSize: 13, fontWeight: 600,
                    borderRadius: 8,
                    border: '1px solid #E4E6EE',
                    background: '#fff', color: '#4A4F5C',
                    cursor: 'pointer',
                    fontFamily: 'DM Sans, sans-serif',
                  }}
                >
                  New link
                </button>
              </div>
            </div>
          ) : (
            <button
              onClick={handleGenerate}
              disabled={selected.size === 0 || generating}
              style={{
                width: '100%', padding: '12px',
                fontSize: 14, fontWeight: 600,
                borderRadius: 8, border: 'none',
                background: selected.size === 0 ? '#E4E6EE' : '#0f2972',
                color: selected.size === 0 ? '#8A8F9E' : '#fff',
                cursor: selected.size === 0 ? 'default' : 'pointer',
                fontFamily: 'DM Sans, sans-serif',
              }}
            >
              {generating
                ? 'Generating...'
                : `Generate link (${selected.size} highlight${selected.size !== 1 ? 's' : ''})`
              }
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
