'use client'
export const dynamic = 'force-dynamic'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import PitchCanvas, { Point } from '@/components/PitchCanvas'

// ─── Tokens ───────────────────────────────────────────────────────
const navy   = '#0f2972'
const orange = '#E8780A'
const border = '#E4E6EE'
const muted  = '#8A8F9E'
const bgPage = '#F8F8F6'

const WORKER  = 'http://localhost:8765'
const IMAGE_W = 2880
const IMAGE_H = 1440

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '10px 14px', border: `1px solid ${border}`,
  borderRadius: 8, fontSize: 14, fontFamily: 'DM Sans, sans-serif',
  color: '#111318', outline: 'none', background: '#fff', boxSizing: 'border-box',
}
const labelStyle: React.CSSProperties = {
  fontSize: 12, fontWeight: 600, color: '#4A4F5C', display: 'block', marginBottom: 4,
}

// ─── Types ────────────────────────────────────────────────────────
interface InsvFile {
  name: string
  path: string
  size_mb: number
}

interface Recording {
  sequence: string
  recorded_at: string
  test_clip: boolean
  estimated_duration: string
  files: InsvFile[]
  selected_file?: string   // path of chosen file
}

type ItemStatus = 'queued' | 'extracting' | 'calibrating' | 'review' | 'accepted' | 'skipped' | 'error'

interface QueueItem {
  id: string
  recording: Recording
  file: InsvFile
  status: ItemStatus
  error?: string
  frame_url?: string
  frame_path?: string
  polygon?: Point[]
  confidence?: number
  rotation?: string
  timestamp: string
}

interface Venue { id: string; name: string }

// ─── Helpers ─────────────────────────────────────────────────────
const normToPixel = (pts: { x: number; y: number }[]): Point[] =>
  pts.map(p => [Math.round(p.x * IMAGE_W), Math.round(p.y * IMAGE_H)])
const pixelToNorm = (pts: Point[]) =>
  pts.map(([x, y]) => ({ x: +(x / IMAGE_W).toFixed(4), y: +(y / IMAGE_H).toFixed(4) }))

function ConfidencePill({ value }: { value: number }) {
  const s = value >= 0.75 ? { bg: '#dcfce7', color: '#166534' }
          : value >= 0.5  ? { bg: '#fef9c3', color: '#854d0e' }
          :                  { bg: '#fee2e2', color: '#991b1b' }
  return <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, ...s }}>{Math.round(value * 100)}% confidence</span>
}

function StatusPill({ status }: { status: ItemStatus }) {
  const map: Record<ItemStatus, { label: string; bg: string; color: string }> = {
    queued:      { label: 'Queued',       bg: '#f3f4f6', color: muted },
    extracting:  { label: 'Extracting…',  bg: '#ede9fe', color: '#5b21b6' },
    calibrating: { label: 'Calibrating…', bg: '#fef9c3', color: '#854d0e' },
    review:      { label: 'Review',       bg: '#ffedd5', color: '#9a3412' },
    accepted:    { label: 'Accepted ✓',   bg: '#dcfce7', color: '#166534' },
    skipped:     { label: 'Skipped',      bg: '#f3f4f6', color: muted },
    error:       { label: 'Error',        bg: '#fee2e2', color: '#991b1b' },
  }
  const s = map[status]
  return <span style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, background: s.bg, color: s.color }}>{s.label}</span>
}

function Spinner() {
  return <>
    <style>{`@keyframes _spin { to { transform:rotate(360deg) } }`}</style>
    <div style={{ width: 18, height: 18, borderRadius: '50%', border: `2px solid ${border}`, borderTopColor: navy, animation: '_spin 0.7s linear infinite', display: 'inline-block' }} />
  </>
}

function smallBtn(bg: string, color: string, borderColor?: string): React.CSSProperties {
  return { padding: '7px 14px', fontSize: 12, fontWeight: 600, borderRadius: 8, cursor: 'pointer', background: bg, color, border: borderColor ? `2px solid ${borderColor}` : 'none', fontFamily: 'DM Sans, sans-serif' }
}

// ─── Page ─────────────────────────────────────────────────────────
export default function CalibratePage() {
  const router = useRouter()
  const [userRole, setUserRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [venues, setVenues]     = useState<Venue[]>([])
  const [workerOk, setWorkerOk] = useState<boolean | null>(null)

  // Step 1: folder + session setup
  const [folderPath, setFolderPath]   = useState('')
  const [scanning, setScanning]       = useState(false)
  const [recordings, setRecordings]   = useState<Recording[]>([])
  const [scanError, setScanError]     = useState('')
  const [date, setDate]               = useState(() => new Date().toISOString().slice(0, 10))
  const [opponent, setOpponent]       = useState('')
  const [venueId, setVenueId]         = useState('')
  const [newVenueName, setNewVenueName] = useState('')
  const [frameTime, setFrameTime]     = useState('00:04:00')
  const [formError, setFormError]     = useState('')

  // Step 2: processing queue
  const [queue, setQueue]         = useState<QueueItem[]>([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [step, setStep]           = useState<'form' | 'processing'>('form')

  // Editor
  const [points, setPoints] = useState<Point[]>([])
  const [closed, setClosed] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState('')

  const current = queue[currentIdx] ?? null

  // ── Auth ────────────────────────────────────────────────────────
  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      const { data: role } = await supabase.from('user_roles').select('role').eq('user_id', user.id).single()
      if (role?.role !== 'admin' && role?.role !== 'coach') { router.push('/'); return }
      setUserRole(role?.role ?? null)
      const { data } = await supabase.from('venues').select('id, name').order('name')
      setVenues(data ?? [])
    }
    load()
  }, [router])

  // ── Worker + auto-detect volumes ───────────────────────────────
  useEffect(() => {
    async function init() {
      try {
        const r = await fetch(`${WORKER}/health`, { signal: AbortSignal.timeout(2000) })
        if (!r.ok) { setWorkerOk(false); return }
        setWorkerOk(true)
        // Auto-detect SD card
        const vr = await fetch(`${WORKER}/scan-volumes`)
        const vdata = await vr.json()
        if (vdata.volumes?.length > 0) {
          setFolderPath(vdata.volumes[0].path)
        }
      } catch {
        setWorkerOk(false)
      }
    }
    init()
  }, [])

  // ── Keyboard shortcuts ─────────────────────────────────────────
  useEffect(() => {
    if (step !== 'processing') return
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'z' || e.key === 'Z') undoPoint()
      if (e.key === 's' || e.key === 'S') closePolygon()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  // ── Sync editor when currentIdx changes ───────────────────────
  useEffect(() => {
    const item = queue[currentIdx]
    if (!item) return
    if (item.polygon) { setPoints(item.polygon); setClosed(true) }
    else { setPoints([]); setClosed(false) }
    setSaveMsg('')
  }, [currentIdx, queue])

  // Sync edited points back into the queue item
  useEffect(() => {
    if (currentIdx < 0 || !queue[currentIdx]) return
    if (queue[currentIdx].status !== 'review') return
    setQueue(prev => prev.map((q, i) =>
      i === currentIdx ? { ...q, polygon: points } : q
    ))
  }, [points])
  
  // ── Scan folder ────────────────────────────────────────────────
  const handleScan = async () => {
    if (!folderPath.trim()) return
    setScanning(true); setScanError(''); setRecordings([])
    try {
      const r = await fetch(`${WORKER}/scan-folder?path=${encodeURIComponent(folderPath.trim())}`)
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail ?? 'Scan failed') }
      const data = await r.json()
      // Pre-select the first (largest) file in each recording
      const withSelection = data.recordings.map((rec: Recording) => ({
        ...rec,
        selected_file: rec.files[0]?.path,
      }))
      setRecordings(withSelection)
      if (data.recordings.length === 0) setScanError(data.warning ?? 'No recordings found')
    } catch (err: any) {
      setScanError(err.message)
    } finally {
      setScanning(false)
    }
  }

  const toggleFileSelection = (seqKey: string, filePath: string) => {
    setRecordings(prev => prev.map(r =>
      r.sequence === seqKey ? { ...r, selected_file: filePath } : r
    ))
  }

  // ── Process a queue item ───────────────────────────────────────
  const processItem = useCallback(async (idx: number, q: QueueItem[]) => {
    const item = q[idx]
    if (!item || item.status !== 'queued') return

    const update = (patch: Partial<QueueItem>) =>
      setQueue(prev => prev.map((qi, i) => i === idx ? { ...qi, ...patch } : qi))

    update({ status: 'extracting' })
    try {
      const r = await fetch(`${WORKER}/extract-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ insv_path: item.file.path, timestamp: item.timestamp }),
      })
      if (!r.ok) throw new Error((await r.json()).detail ?? 'Extract failed')
      const ex = await r.json()
      update({ frame_url: ex.frame_url, frame_path: ex.frame_path, rotation: ex.rotation })

      update({ status: 'calibrating' })
      const r2 = await fetch(`${WORKER}/auto-calibrate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_path: ex.frame_path }),
      })
      if (!r2.ok) throw new Error((await r2.json()).detail ?? 'Calibration failed')
      const cal = await r2.json()
      const polygon = normToPixel(cal.polygon)
      update({ status: 'review', polygon, confidence: cal.confidence })
      setPoints(polygon); setClosed(true)
    } catch (err: any) {
      update({ status: 'error', error: err.message })
    }
  }, [])

  // ── Start session ──────────────────────────────────────────────
  const handleStart = async () => {
    setFormError('')
    if (!opponent.trim())  { setFormError('Bitte Gegner eingeben'); return }
    if (!date)             { setFormError('Bitte Datum wählen'); return }
    if (!venueId && !newVenueName.trim()) { setFormError('Bitte Spielstätte wählen'); return }

    const selected = recordings.filter(r => r.selected_file && !(r as any)._excluded)
    if (selected.length === 0) { setFormError('Bitte mindestens eine Aufnahme auswählen'); return }

    let finalVenueId = venueId === '__new__' ? '' : venueId
    if (!finalVenueId && newVenueName.trim()) {
      const { data } = await supabase.from('venues').insert({ name: newVenueName.trim() }).select('id').single()
      if (data) {
        finalVenueId = data.id
        setVenueId(data.id)
        setVenues(prev => [...prev, { id: data.id, name: newVenueName.trim() }])
      }
    }

    const items: QueueItem[] = selected.map(rec => {
      const file = rec.files.find(f => f.path === rec.selected_file) ?? rec.files[0]
      return { id: crypto.randomUUID(), recording: rec, file, status: 'queued', timestamp: frameTime }
    })

    setQueue(items)
    setCurrentIdx(0)
    setStep('processing')
    setTimeout(() => processItem(0, items), 50)
  }

  // ── Navigation ─────────────────────────────────────────────────
  const goTo = (idx: number) => {
    if (idx < 0 || idx >= queue.length) return
    setCurrentIdx(idx)
    if (queue[idx]?.status === 'queued') setTimeout(() => processItem(idx, queue), 50)
  }

  const skipCurrent = () => {
    setQueue(prev => prev.map((q, i) => i === currentIdx ? { ...q, status: 'skipped' } : q))
    goTo(currentIdx + 1)
  }

  // Core re-extract — accepts optional overrides for file and rotation
  const reExtractWith = async (overrides: { file?: InsvFile; rotation?: string; timestamp?: string } = {}) => {
    if (!current) return
    const update = (patch: Partial<QueueItem>) =>
      setQueue(prev => prev.map((q, i) => i === currentIdx ? { ...q, ...patch } : q))

    const targetFile     = overrides.file      ?? current.file
    const targetRotation = overrides.rotation  ?? current.rotation
    const targetTs       = overrides.timestamp ?? current.timestamp

    // Apply file/rotation overrides to queue item immediately
    if (overrides.file || overrides.rotation) {
      update({
        ...(overrides.file     ? { file: overrides.file }         : {}),
        ...(overrides.rotation ? { rotation: overrides.rotation } : {}),
      })
    }

    update({ status: 'extracting', polygon: undefined, frame_url: undefined })
    setPoints([]); setClosed(false)
    try {
      const r = await fetch(`${WORKER}/extract-frame`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          insv_path: targetFile.path,
          timestamp: targetTs,
          ...(targetRotation ? { rotation: targetRotation } : {}),
        }),
      })
      if (!r.ok) throw new Error((await r.json()).detail ?? 'Extract failed')
      const ex = await r.json()
      update({ frame_url: ex.frame_url, frame_path: ex.frame_path, rotation: ex.rotation, status: 'calibrating' })
      const r2 = await fetch(`${WORKER}/auto-calibrate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_path: ex.frame_path }),
      })
      if (!r2.ok) throw new Error((await r2.json()).detail ?? 'Calibration failed')
      const cal = await r2.json()
      const polygon = normToPixel(cal.polygon)
      update({ status: 'review', polygon, confidence: cal.confidence })
      setPoints(polygon); setClosed(true)
    } catch (err: any) {
      update({ status: 'error', error: err.message })
    }
  }

  const reExtract       = () => reExtractWith()
  const switchFile      = (file: InsvFile) => { if (file.path !== current?.file.path) reExtractWith({ file }) }
  const switchRotation  = (rotation: string) => { if (rotation !== current?.rotation) reExtractWith({ rotation }) }

  // ── Editor ─────────────────────────────────────────────────────
  const undoPoint    = () => { if (closed) setClosed(false); else setPoints(p => p.slice(0, -1)); setSaveMsg('') }
  const closePolygon = () => { if (points.length >= 3) setClosed(true) }

  // ── Accept + save ──────────────────────────────────────────────
  const handleAccept = async () => {
    const effectiveVenueId = venueId === '__new__' ? '' : venueId
    if (!current || points.length < 4 || !effectiveVenueId) return
    setSaving(true); setSaveMsg('')
    try {
      let imageStorageUrl: string | null = null
      if (current.frame_url) {
        const blob = await fetch(current.frame_url).then(r => r.blob())
        const path = `calibration/${effectiveVenueId}/equirect_frame.jpg`
        const { error: upErr } = await supabase.storage.from('venues').upload(path, blob, { upsert: true })
        if (!upErr) {
          const { data: { publicUrl } } = supabase.storage.from('venues').getPublicUrl(path)
          imageStorageUrl = publicUrl
        }
      }
      const { error } = await supabase.from('venues').update({
        pitch_polygon: pixelToNorm(points),
        ...(imageStorageUrl ? { calibration_image_url: imageStorageUrl } : {}),
        updated_at: new Date().toISOString(),
      }).eq('id', effectiveVenueId)
      if (error) throw error
      setQueue(prev => prev.map((q, i) => i === currentIdx ? { ...q, status: 'accepted' } : q))
      setSaveMsg('Gespeichert ✓')
      setTimeout(() => goTo(currentIdx + 1), 700)
    } catch (err: any) {
      setSaveMsg(`Fehler: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDownload = () => {
    if (!current || points.length < 4) return
    const venue = venues.find(v => v.id === venueId)
    const payload = {
      version: 1,
      ground_id: venue?.name.toLowerCase().replace(/\s+/g, '_') ?? 'unknown',
      venue_id: venueId,
      created_at: new Date().toISOString(),
      source_frame: { width: IMAGE_W, height: IMAGE_H, timestamp: current.timestamp },
      auto_polygon: pixelToNorm(points),
      pixel_polygon: points,
    }
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
    a.download = `pitch_${venue?.name ?? 'venue'}.json`.toLowerCase().replace(/\s+/g, '_')
    a.click()
  }

  const acceptedCount = queue.filter(q => q.status === 'accepted').length
  const venue = venues.find(v => v.id === venueId)

  // ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ minHeight: '100vh', background: bgPage, fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={userRole} backHref="/" />
      <div style={{ padding: '24px 20px', maxWidth: step === 'form' ? 620 : 1100, margin: '0 auto' }}>

        {/* ── STEP 1: Setup ──────────────────────────────────────── */}
        {step === 'form' && (
          <>
            <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 28, color: navy, letterSpacing: '0.02em', marginBottom: 4 }}>
              Kalibrierung starten
            </div>
            <div style={{ fontSize: 12, color: muted, marginBottom: 20 }}>
              SD-Karte einlegen, Ordner scannen, Aufnahmen auswählen.
            </div>

            {/* Worker status */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px',
              borderRadius: 8, marginBottom: 20,
              background: workerOk === true ? '#dcfce7' : workerOk === false ? '#fee2e2' : '#f3f4f6',
              border: `1px solid ${workerOk === true ? '#bbf7d0' : workerOk === false ? '#fca5a5' : border}`,
            }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: workerOk === true ? '#166534' : workerOk === false ? '#991b1b' : muted }}>
                {workerOk === true ? '● Worker online' : workerOk === false ? '● Worker offline' : '● Verbinde…'}
              </span>
              {workerOk === false && (
                <span style={{ fontSize: 12, color: '#991b1b' }}>
                  — starte: <code style={{ background: '#fee2e2', padding: '1px 4px', borderRadius: 4 }}>uvicorn worker_server:app --port 8765</code>
                </span>
              )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

              {/* Folder path + scan */}
              <div>
                <label style={labelStyle}>Ordner (SD-Karte oder lokal)</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    value={folderPath}
                    onChange={e => setFolderPath(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleScan()}
                    placeholder="/Volumes/Insta360/DCIM/Camera01"
                    style={{ ...inputStyle, flex: 1 }}
                  />
                  <button
                    onClick={handleScan}
                    disabled={!folderPath.trim() || scanning || workerOk !== true}
                    style={{
                      padding: '10px 18px', borderRadius: 8, border: 'none',
                      background: !folderPath.trim() || workerOk !== true ? '#E4E6EE' : navy,
                      color: !folderPath.trim() || workerOk !== true ? muted : '#fff',
                      fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {scanning ? 'Scanne…' : 'Scannen'}
                  </button>
                </div>
                {scanError && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 4 }}>{scanError}</div>}
              </div>

              {/* Recordings list */}
              {recordings.length > 0 && (
                <div>
                  <label style={labelStyle}>Aufnahmen ({recordings.filter(r => !(r as any)._excluded).length} ausgewählt)</label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {recordings.map(rec => {
                      const excluded = (rec as any)._excluded
                      return (
                        <div
                          key={rec.sequence}
                          style={{
                            background: excluded ? bgPage : '#fff',
                            border: `1px solid ${excluded ? border : navy}`,
                            borderRadius: 10, padding: '10px 14px',
                            opacity: excluded ? 0.5 : 1,
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: rec.files.length > 1 ? 8 : 0 }}>
                            <input
                              type="checkbox"
                              checked={!excluded}
                              onChange={() => setRecordings(prev => prev.map(r =>
                                r.sequence === rec.sequence ? { ...r, _excluded: !excluded } as any : r
                              ))}
                              style={{ width: 16, height: 16, cursor: 'pointer' }}
                            />
                            <div style={{ flex: 1 }}>
                              <span style={{ fontSize: 13, fontWeight: 600, color: navy }}>{rec.recorded_at}</span>
                              <span style={{ fontSize: 12, color: muted, marginLeft: 8 }}>{rec.estimated_duration}</span>
                              {rec.test_clip && (
                                <span style={{ fontSize: 10, fontWeight: 600, marginLeft: 8, padding: '1px 6px', borderRadius: 99, background: '#fef9c3', color: '#854d0e' }}>
                                  Testclip
                                </span>
                              )}
                            </div>
                          </div>

                          {/* File picker — only shown when multiple files in recording */}
                          {rec.files.length > 1 && (
                            <div style={{ paddingLeft: 26, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {rec.files.map(file => (
                                <button
                                  key={file.path}
                                  onClick={() => toggleFileSelection(rec.sequence, file.path)}
                                  style={{
                                    padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                                    cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                                    background: rec.selected_file === file.path ? navy : '#fff',
                                    color: rec.selected_file === file.path ? '#fff' : muted,
                                    border: `1px solid ${rec.selected_file === file.path ? navy : border}`,
                                  }}
                                >
                                  {file.name} <span style={{ opacity: 0.7 }}>{file.size_mb}MB</span>
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Session details — only shown after scan */}
              {recordings.length > 0 && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    <div>
                      <label style={labelStyle}>Datum</label>
                      <input type="date" value={date} onChange={e => setDate(e.target.value)} style={inputStyle} />
                    </div>
                    <div>
                      <label style={labelStyle}>Gegner</label>
                      <input type="text" placeholder="FC Gegner" value={opponent} onChange={e => setOpponent(e.target.value)} style={inputStyle} />
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    <div>
                      <label style={labelStyle}>Spielstätte</label>
                      <select value={venueId} onChange={e => setVenueId(e.target.value)} style={{ ...inputStyle, background: bgPage }}>
                        <option value="">— Spielstätte wählen —</option>
                        {venues.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
                        <option value="__new__">+ Neu anlegen</option>
                      </select>
                    </div>
                    <div>
                      <label style={labelStyle}>Frame-Zeitpunkt</label>
                      <input value={frameTime} onChange={e => setFrameTime(e.target.value)} placeholder="00:04:00" style={inputStyle} />
                    </div>
                  </div>

                  {venueId === '__new__' && (
                    <div>
                      <label style={labelStyle}>Name der neuen Spielstätte</label>
                      <input type="text" placeholder="ASN Sportplatz Nürnberg" value={newVenueName} onChange={e => setNewVenueName(e.target.value)} style={inputStyle} />
                    </div>
                  )}

                  {formError && (
                    <div style={{ fontSize: 12, color: '#ef4444', background: '#fef2f2', padding: '8px 12px', borderRadius: 8 }}>
                      {formError}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: 8 }}>
                    <button onClick={() => router.push('/')} style={{ flex: 1, padding: 11, background: '#fff', color: navy, border: `2px solid ${navy}`, borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>
                      Abbrechen
                    </button>
                    <button
                      onClick={handleStart}
                      style={{ flex: 1, padding: 11, background: navy, color: '#fff', border: 'none', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}
                    >
                      Kalibrierung starten →
                    </button>
                  </div>
                </>
              )}
            </div>
          </>
        )}

        {/* ── STEP 2: Processing ─────────────────────────────────── */}
        {step === 'processing' && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
              <div>
                <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 28, color: navy, letterSpacing: '0.02em' }}>
                  {date} — {opponent}
                </div>
                <div style={{ fontSize: 12, color: muted }}>
                  {venue?.name ?? '—'} &nbsp;·&nbsp; {acceptedCount}/{queue.length} abgeschlossen
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <button onClick={() => setStep('form')} style={{ background: '#fff', color: navy, border: `2px solid ${navy}`, borderRadius: 8, padding: '8px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>
                ← Neue Session
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 16, alignItems: 'start' }}>

              {/* Queue */}
              <div style={{ background: '#fff', border: `1px solid ${border}`, borderRadius: 12, padding: '14px 14px' }}>
                <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 14, color: navy, marginBottom: 10, letterSpacing: 0.5 }}>
                  Warteschlange
                </div>
                {queue.map((item, i) => (
                  <div key={item.id} onClick={() => goTo(i)} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '7px 8px', borderRadius: 8, cursor: 'pointer', marginBottom: 2, background: i === currentIdx ? '#e8edf8' : 'transparent' }}>
                    <span style={{ fontSize: 11, color: muted, minWidth: 16 }}>{i + 1}</span>
                    <div style={{ flex: 1, overflow: 'hidden' }}>
                      <div style={{ fontSize: 12, color: '#111318', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.recording.recorded_at}</div>
                      <div style={{ fontSize: 10, color: muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.file.name}</div>
                    </div>
                    <StatusPill status={item.status} />
                  </div>
                ))}
              </div>

              {/* Editor */}
              <div>
                {!current ? (
                  <div style={{ background: '#fff', border: `1px solid ${border}`, borderRadius: 12, padding: 40, textAlign: 'center', color: muted }}>
                    Alle Dateien verarbeitet
                  </div>
                ) : (
                  <>
                    {/* File header */}
                    <div style={{ background: '#fff', border: `1px solid ${border}`, borderRadius: 12, padding: '12px 16px', marginBottom: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: navy }}>{current.recording.recorded_at} — {current.recording.estimated_duration}</div>
                          <div style={{ fontSize: 11, color: muted, marginTop: 1 }}>{current.file.name} · {current.file.size_mb}MB</div>
                        </div>
                        <StatusPill status={current.status} />
                        {current.confidence !== undefined && <ConfidencePill value={current.confidence} />}
                        <span style={{ fontSize: 11, color: muted }}>{currentIdx + 1}/{queue.length}</span>
                      </div>

                      {/* Lens + rotation controls */}
                      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', borderTop: `1px solid ${border}`, paddingTop: 10 }}>

                        {/* Lens switcher — only shown when recording has multiple files */}
                        {current.recording.files.length > 1 && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: 11, color: muted, fontWeight: 600 }}>Linse</span>
                            {current.recording.files.map(f => (
                              <button
                                key={f.path}
                                onClick={() => switchFile(f)}
                                style={{
                                  padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                                  cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                                  background: current.file.path === f.path ? navy : '#fff',
                                  color: current.file.path === f.path ? '#fff' : muted,
                                  border: `1px solid ${current.file.path === f.path ? navy : border}`,
                                }}
                              >
                                {f.name}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Rotation toggle */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ fontSize: 11, color: muted, fontWeight: 600 }}>Rotation</span>
                          {([['cw', '↻ CW'], ['ccw', '↺ CCW'], ['180', '↕ 180°']] as const).map(([val, label]) => (
                            <button
                              key={val}
                              onClick={() => switchRotation(val)}
                              style={{
                                padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                                cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
                                background: current.rotation === val ? navy : '#fff',
                                color: current.rotation === val ? '#fff' : muted,
                                border: `1px solid ${current.rotation === val ? navy : border}`,
                              }}
                            >
                              {label}
                            </button>
                          ))}
                        </div>

                      </div>
                    </div>

                    {/* Spinner */}
                    {['extracting', 'calibrating'].includes(current.status) && (
                      <div style={{ background: '#fff', border: `1px solid ${border}`, borderRadius: 12, padding: 40, textAlign: 'center', color: muted, marginBottom: 12 }}>
                        <Spinner />
                        <div style={{ fontSize: 14, marginTop: 12 }}>
                          {current.status === 'extracting'  && 'Keyframe wird extrahiert…'}
                          {current.status === 'calibrating' && 'Auto-Kalibrierung läuft…'}
                        </div>
                        <div style={{ fontSize: 11, color: muted, marginTop: 4 }}>{current.file.name}</div>
                      </div>
                    )}

                    {/* Error */}
                    {current.status === 'error' && (
                      <div style={{ background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 12, padding: '16px 20px', marginBottom: 12, fontSize: 13, color: '#991b1b' }}>
                        <strong>Fehler:</strong> {current.error}
                        <div style={{ marginTop: 10, display: 'flex', gap: 8 }}>
                          <button onClick={reExtract} style={smallBtn(navy, '#fff')}>Wiederholen</button>
                          <button onClick={skipCurrent} style={smallBtn('#fff', navy, navy)}>Überspringen</button>
                        </div>
                      </div>
                    )}

                    {/* Canvas */}
                    {current.frame_url && !['extracting', 'calibrating'].includes(current.status) && (
                      <>
                        <div style={{ marginBottom: 12 }}>
                          <PitchCanvas
                            imageUrl={current.frame_url}
                            imageNaturalWidth={IMAGE_W}
                            imageNaturalHeight={IMAGE_H}
                            points={points}
                            onAddPoint={p => { if (!closed) setPoints(prev => [...prev, p]); setSaveMsg('') }}
                            onUpdatePoint={(i, p) => { setPoints(prev => prev.map((pt, idx) => idx === i ? p : pt)); setSaveMsg('') }}
                            closed={closed}
                          />
                        </div>

                        {/* Toolbar */}
                        <div style={{ background: '#fff', border: `1px solid ${border}`, borderRadius: 12, padding: '12px 16px', display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                          <button onClick={undoPoint} disabled={points.length === 0} style={smallBtn('#fff', navy, navy)}>↩ Undo</button>
                          <button onClick={closePolygon} disabled={points.length < 3 || closed} style={smallBtn('#fff', navy, navy)}>Schließen</button>
                          <button onClick={() => { setPoints([]); setClosed(false); setSaveMsg('') }} disabled={points.length === 0} style={smallBtn('#ef4444', '#fff')}>Leeren</button>

                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 4 }}>
                            <input
                              value={current.timestamp}
                              onChange={e => setQueue(prev => prev.map((q, i) => i === currentIdx ? { ...q, timestamp: e.target.value } : q))}
                              style={{ width: 84, padding: '5px 8px', border: `1px solid ${border}`, borderRadius: 6, fontSize: 12, fontFamily: 'DM Sans, sans-serif' }}
                            />
                            <button onClick={reExtract} style={smallBtn('#fff', muted, border)}>Neues Frame</button>
                          </div>

                          <div style={{ flex: 1 }} />
                          <button onClick={skipCurrent} style={smallBtn('#fff', muted, border)}>Überspringen →</button>
                          <button onClick={handleDownload} disabled={points.length < 4} style={smallBtn('#fff', navy, navy)}>↓ JSON</button>
                          <button onClick={handleAccept} disabled={points.length < 4 || saving || !venueId} style={smallBtn(points.length < 4 || saving || !venueId ? '#E4E6EE' : navy, points.length < 4 || saving || !venueId ? muted : '#fff')}>
                            {saving ? 'Speichern…' : '✓ Akzeptieren + Weiter'}
                          </button>
                        </div>

                        {!venueId && points.length >= 4 && (
                          <div style={{ fontSize: 12, color: orange, marginTop: 8 }}>Keine Spielstätte ausgewählt</div>
                        )}
                        {saveMsg && (
                          <div style={{ marginTop: 8, fontSize: 13, padding: '8px 12px', borderRadius: 8, background: saveMsg.startsWith('Fehler') ? '#fee2e2' : '#dcfce7', color: saveMsg.startsWith('Fehler') ? '#991b1b' : '#166534' }}>
                            {saveMsg}
                          </div>
                        )}
                        <div style={{ marginTop: 8, fontSize: 11, color: muted }}>
                          <kbd style={kbdStyle}>Z</kbd> Undo &nbsp;<kbd style={kbdStyle}>S</kbd> Schließen &nbsp;Punkte ziehen zum Anpassen
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

const kbdStyle: React.CSSProperties = {
  display: 'inline-block', background: bgPage, border: `1px solid ${border}`,
  borderRadius: 4, padding: '1px 5px', fontFamily: 'DM Sans, sans-serif', fontSize: 10, marginRight: 4,
}
