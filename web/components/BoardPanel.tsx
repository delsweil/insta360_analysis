'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  type Board, type BoardState, type Box, type Raw,
  boxOf, frameAt, parseBoard, partAt, playersOf, uploadBoard,
} from '@/lib/board'

const NAVY = '#0f2972'
const PAD = 3 // metres of grass around the pitch
const OTHER_COLOUR = '#aab3c5'

function fmt(sec: number) {
  const s = Math.max(0, Math.floor(sec))
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
}

/** 10th-90th percentile spread of a list of numbers. */
function spread(v: number[]) {
  const a = [...v].sort((x, y) => x - y)
  const q = (p: number) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))]
  return q(0.9) - q(0.1)
}

function Pitch({ box, grass = 1 }: { box: Box; grass?: number }) {
  const [x0, x1, y0, y1] = box
  const L = x1 - x0
  const W = y1 - y0
  const sx = L / 105
  const sy = W / 68
  const cx = (x0 + x1) / 2
  const cy = (y0 + y1) / 2
  const line = 'rgba(255,255,255,0.85)'
  const lw = 0.28
  const bands = Array.from({ length: 10 }, (_, i) => i)
  return (
    <g>
      <rect x={x0 - PAD} y={y0 - PAD} width={L + 2 * PAD} height={W + 2 * PAD} fill="#2a7344" fillOpacity={grass} />
      {bands.map(i => (
        <rect key={i} x={x0 + (L / 10) * i} y={y0} width={L / 10} height={W} fill={i % 2 ? '#2f7d49' : '#2a7344'} fillOpacity={grass} />
      ))}
      <g fill="none" stroke={line} strokeWidth={lw}>
        <rect x={x0} y={y0} width={L} height={W} />
        <line x1={cx} y1={y0} x2={cx} y2={y1} />
        <ellipse cx={cx} cy={cy} rx={9.15 * sx} ry={9.15 * sy} />
        {[0, 1].map(end => {
          const dir = end ? -1 : 1
          const gx = end ? x1 : x0
          const rect = (depth: number, half: number) => (
            <rect x={end ? gx - depth * sx : gx} y={cy - half * sy} width={depth * sx} height={2 * half * sy} />
          )
          return (
            <g key={end}>
              {rect(16.5, 20.16)}
              {rect(5.5, 9.16)}
              <circle cx={gx + dir * 11 * sx} cy={cy} r={0.25} fill={line} stroke="none" />
            </g>
          )
        })}
        <circle cx={cx} cy={cy} r={0.3} fill={line} stroke="none" />
      </g>
    </g>
  )
}

interface ViewProps {
  board: Board
  t: number            // video time the board shows
  trails: boolean
  svgRef?: React.Ref<SVGSVGElement>
  maxHeight?: string
  translucent?: boolean   // small see-through version for over the video: pitch lines and dots only
}

function BoardView({ board, t, trails, svgRef, maxHeight, translucent }: ViewProps) {
  const part = partAt(board, t)
  const [x0, x1, y0, y1] = boxOf(board, part)
  const idx = frameAt(board, t)
  const frame = idx >= 0 ? board.frames[idx] : null
  const colours = [board.teams.A.colour, board.teams.B.colour, OTHER_COLOUR]
  const players = frame ? playersOf(frame) : []
  const wedge = useMemo(() => {
    if (!frame || frame[1] == null || frame[2] == null) return null
    const k = board.kAngle
    const a1 = ((frame[1] - frame[2] / 2) * k * Math.PI) / 180
    const a2 = ((frame[1] + frame[2] / 2) * k * Math.PI) / 180
    const R = 150
    return `M0,0 L${R * Math.sin(a1)},${R * Math.cos(a1)} L${R * Math.sin(a2)},${R * Math.cos(a2)} Z`
  }, [frame, board.kAngle])

  const old: { x: number; y: number; team: number; age: number }[] = []
  if (trails && idx > 0) {
    for (let k = 1; k <= 6 && idx - k >= 0; k++) {
      if (board.times[idx] - board.times[idx - k] > 2.4) break
      for (const p of playersOf(board.frames[idx - k])) old.push({ ...p, age: k })
    }
  }

  // the view takes in the pitch and everyone ever seen beside it (near touchline, benches)
  const r = board.reach
  const vx0 = Math.min(x0 - PAD, r[0]), vx1 = Math.max(x1 + PAD, r[1])
  const vy0 = Math.min(y0 - PAD, r[2]), vy1 = Math.max(y1 + PAD, r[3])
  const vbW = vx1 - vx0
  const vbH = vy1 - vy0
  return (
    <svg
      ref={svgRef}
      viewBox={`${vx0} ${-vy1} ${vbW} ${vbH}`}
      style={{ display: 'block', width: '100%', maxHeight, borderRadius: 8, background: translucent ? 'transparent' : '#2a7344' }}
      role="img"
      aria-label="Top-down board of the players"
    >
      <g transform="scale(1,-1)">
        <Pitch box={[x0, x1, y0, y1]} grass={translucent ? 0.42 : 1} />
        {wedge && <path d={wedge} fill="rgba(255,255,255,0.10)" stroke="rgba(255,255,255,0.45)" strokeWidth={0.2} strokeDasharray="1.2 0.8" />}
        {old.map((p, i) => (
          <circle key={`o${i}`} cx={p.x} cy={p.y} r={0.55} fill={colours[p.team] ?? OTHER_COLOUR} opacity={0.4 - p.age * 0.05} />
        ))}
        {[2, 1, 0].map(team =>
          players.filter(p => p.team === team).map((p, i) => (
            <circle key={`${team}-${i}`} cx={p.x} cy={p.y} r={team === 2 ? (translucent ? 1.2 : 0.95) : (translucent ? 1.8 : 1.35)}
              fill={colours[team]} stroke="#0a1a3a" strokeWidth={0.3} opacity={team === 2 ? 0.8 : 1} />
          )),
        )}
      </g>
      {!frame && (
        <text x={vx0 + vbW / 2} y={-vy1 + vbH / 2} textAnchor="middle" fontSize={vbW / 22} fill="#fff" opacity={0.85}>
          No players tracked at this moment
        </text>
      )}
    </svg>
  )
}

async function savePicture(svg: SVGSVGElement, caption: string, filename: string) {
  const vb = svg.viewBox.baseVal
  const W = 1400
  const H = Math.round((W * vb.height) / vb.width)
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  clone.setAttribute('width', String(W))
  clone.setAttribute('height', String(H))
  clone.style.maxHeight = ''
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml;charset=utf-8' }))
  try {
    const img = new Image()
    await new Promise((ok, fail) => { img.onload = () => ok(null); img.onerror = fail; img.src = url })
    const cv = document.createElement('canvas')
    cv.width = W
    cv.height = H + 64
    const ctx = cv.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = NAVY
    ctx.fillRect(0, 0, cv.width, cv.height)
    ctx.drawImage(img, 0, 0, W, H)
    ctx.fillStyle = '#fff'
    ctx.font = '600 28px DM Sans, sans-serif'
    ctx.textBaseline = 'middle'
    ctx.fillText(caption, 24, H + 32)
    const blob: Blob | null = await new Promise(ok => cv.toBlob(ok, 'image/png'))
    if (!blob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = filename
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 2000)
  } finally {
    URL.revokeObjectURL(url)
  }
}

interface Props {
  gameId: string
  gameTitle?: string
  state: BoardState
  reload: () => void
  currentTime: number
  isCoach: boolean
  compact?: boolean     // only the see-through pitch, no card, text or buttons (mobile HUD)
}

const btn: React.CSSProperties = {
  fontSize: 11, fontWeight: 600, color: NAVY, background: '#e8edf8', border: 'none',
  borderRadius: 99, padding: '3px 10px', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
}

export default function BoardPanel({ gameId, gameTitle, state, reload, currentTime, isCoach, compact }: Props) {
  const board = state.status === 'ready' ? state.board : null
  const raw = state.status === 'ready' ? state.raw : null
  const [trails, setTrails] = useState(false)
  const [big, setBig] = useState(false)
  const [shift, setShift] = useState(0)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const bigRef = useRef<SVGSVGElement>(null)

  useEffect(() => { if (board) setShift(board.shift) }, [board])
  useEffect(() => {
    if (!big) return
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') setBig(false) }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [big])

  async function onFile(file: File | undefined) {
    if (!file) return
    setBusy(true)
    setMsg(null)
    try {
      const data = JSON.parse(await file.text()) as Raw
      parseBoard(data) // checks that it is a board file
      const err = await uploadBoard(gameId, data)
      if (err) {
        setMsg(/bucket/i.test(err) ? `${err} - run web/supabase/boards.sql in the Supabase SQL editor first.` : err)
      } else {
        reload()
      }
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function saveSync() {
    if (!raw) return
    setBusy(true)
    setMsg(null)
    const err = await uploadBoard(gameId, { ...raw, sync_shift_s: shift })
    setBusy(false)
    if (err) setMsg(err)
    else reload()
  }

  const card: React.CSSProperties = { background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '12px 14px' }
  const head: React.CSSProperties = {
    fontFamily: 'Bebas Neue, sans-serif', fontSize: 16, color: NAVY, letterSpacing: '0.04em',
    display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8,
  }

  // Players do not see an empty box; coaches see how to fill it.
  if (!board) {
    if (!isCoach || state.status === 'loading') return null
    return (
      <div style={{ ...card, borderStyle: 'dashed' }}>
        <div style={head}>Board</div>
        <div style={{ fontSize: 12, color: '#5b6377', marginBottom: 8 }}>
          {state.status === 'error' ? `Board data could not be read: ${state.message}` :
            'No board data for this game yet. Upload the *_board.json from the analysis app to show every player on a top-down board next to the video.'}
        </div>
        <label style={{ ...btn, display: 'inline-block', opacity: busy ? 0.6 : 1 }}>
          {busy ? 'Uploading...' : 'Upload board data'}
          <input type="file" accept=".json,application/json" disabled={busy} style={{ display: 'none' }}
            onChange={e => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        </label>
        {msg && <div style={{ fontSize: 11, color: '#dc2626', marginTop: 8 }}>{msg}</div>}
      </div>
    )
  }

  const t = currentTime + shift
  if (compact) return <BoardView board={board} t={t} trails={false} translucent />
  const part = partAt(board, t)
  const idx = frameAt(board, t)
  const players = idx >= 0 ? playersOf(board.frames[idx]) : []
  const teams = [board.teams.A, board.teams.B]
  const attacksRight = part?.a_attacks_right
  const info = [0, 1].map(k => {
    const pts = players.filter(p => p.team === k)
    return { n: pts.length, length: pts.length >= 4 ? spread(pts.map(p => p.x)) : null, width: pts.length >= 4 ? spread(pts.map(p => p.y)) : null }
  })
  const other = players.filter(p => p.team === 2).length
  const caption = `${gameTitle ?? board.title} · ${fmt(currentTime)}`
  const changed = Math.abs(shift - board.shift) > 1e-6

  const legend = (dark: boolean) => (
    <div style={{ marginTop: 8 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 12px', fontSize: 12, color: dark ? '#fff' : '#1a2340' }}>
        {teams.map((tm, k) => (
          <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: tm.colour, border: '1px solid #0a1a3a' }} />
            <strong>{tm.name}</strong> {info[k].n}
            {attacksRight != null && (k === 0) === attacksRight && ' →'}
            {attacksRight != null && (k === 0) !== attacksRight && ' ←'}
          </span>
        ))}
        {other > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, color: dark ? '#cbd5ee' : '#5b6377' }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: OTHER_COLOUR }} /> other {other}
          </span>
        )}
      </div>
      {info.some(i => i.length != null) && (
        <div style={{ fontSize: 11, color: dark ? '#cbd5ee' : '#5b6377', marginTop: 4 }}>
          {teams.map((tm, k) => info[k].length != null && (
            <div key={k}>{tm.name}: {Math.round(info[k].length!)} m long, {Math.round(info[k].width!)} m wide</div>
          ))}
        </div>
      )}
    </div>
  )

  return (
    <div style={card}>
      <div style={head}>
        <span>Board <span style={{ fontFamily: 'DM Sans, sans-serif', fontSize: 11, color: '#8A8F9E', letterSpacing: 0 }}>{fmt(currentTime)}</span></span>
        <span style={{ display: 'flex', gap: 6 }}>
          <button style={{ ...btn, background: trails ? NAVY : '#e8edf8', color: trails ? '#fff' : NAVY }} onClick={() => setTrails(v => !v)} title="Show the last two seconds">Trails</button>
          <button style={btn} onClick={() => svgRef.current && savePicture(svgRef.current, caption, `board-${Math.floor(currentTime)}s.png`)} title="Save this moment as a picture">Picture</button>
          <button style={btn} onClick={() => setBig(true)} title="Enlarge">⤢</button>
        </span>
      </div>
      <BoardView board={board} t={t} trails={trails} svgRef={svgRef} />
      {legend(false)}
      <div style={{ fontSize: 10, color: '#8A8F9E', marginTop: 8 }}>
        Positions are estimated from the video (a few metres off, more on the far side): read the shape, not the exact distances.
      </div>
      {isCoach && (
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 6, marginTop: 8, fontSize: 11, color: '#5b6377' }}>
          <span>Sync</span>
          <button style={btn} onClick={() => setShift(s => Math.round((s - 0.5) * 10) / 10)} title="Show an earlier moment">◀ 0.5 s</button>
          <span style={{ minWidth: 34, textAlign: 'center' }}>{shift > 0 ? '+' : ''}{shift.toFixed(1)} s</span>
          <button style={btn} onClick={() => setShift(s => Math.round((s + 0.5) * 10) / 10)} title="Show a later moment">0.5 s ▶</button>
          {changed && <button style={{ ...btn, background: '#E8780A', color: '#fff' }} disabled={busy} onClick={saveSync}>Save for everyone</button>}
          <label style={{ ...btn, marginLeft: 'auto', opacity: busy ? 0.6 : 1 }}>
            Replace data
            <input type="file" accept=".json,application/json" disabled={busy} style={{ display: 'none' }}
              onChange={e => { onFile(e.target.files?.[0]); e.target.value = '' }} />
          </label>
        </div>
      )}
      {msg && <div style={{ fontSize: 11, color: '#dc2626', marginTop: 8 }}>{msg}</div>}

      {big && (
        <div onClick={() => setBig(false)} style={{
          position: 'fixed', inset: 0, zIndex: 10000, background: 'rgba(9,29,82,0.98)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 16,
        }}>
          <div onClick={e => e.stopPropagation()} style={{ width: 'min(96vw, 1200px)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#fff', marginBottom: 8 }}>
              <span style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 20, letterSpacing: '0.04em' }}>{caption}</span>
              <span style={{ display: 'flex', gap: 6 }}>
                <button style={btn} onClick={() => setTrails(v => !v)}>{trails ? 'Trails on' : 'Trails off'}</button>
                <button style={btn} onClick={() => bigRef.current && savePicture(bigRef.current, caption, `board-${Math.floor(currentTime)}s.png`)}>Picture</button>
                <button style={btn} onClick={() => setBig(false)}>Close</button>
              </span>
            </div>
            <BoardView board={board} t={t} trails={trails} svgRef={bigRef} maxHeight="78vh" />
            {legend(true)}
          </div>
        </div>
      )}
    </div>
  )
}
