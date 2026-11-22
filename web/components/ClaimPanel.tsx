'use client'

import { useState } from 'react'
import { type Board, boxOf, partAt } from '@/lib/board'
import type { Me } from '@/lib/self'
import { CAND_COLOURS } from '@/components/VideoTags'
import { Pitch, PAD } from '@/components/BoardPanel'
import {
  type Question, type Tracks,
  pointAt, parseTracks, uploadTracks, WINDOW_S,
} from '@/lib/tracks'

const NAVY = '#0f2972'
const GOLD = '#ffd400'
const CAND = CAND_COLOURS

const btn: React.CSSProperties = {
  fontSize: 12, fontWeight: 600, color: NAVY, background: '#e8edf8', border: 'none',
  borderRadius: 99, padding: '5px 12px', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif',
}

function fmt(sec: number) {
  const s = Math.max(0, Math.floor(sec))
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`
}

type Pos = { x: number; y: number }

/** The positions of a piece in the few moments next to its end (dir = next) or start (dir = prev). */
function trail(board: Board, tracks: Tracks, pid: number, f: number, step: 1 | -1, n = 15): Pos[] {
  const out: Pos[] = []
  for (let g = f, c = 0; g >= 0 && g < board.frames.length && c < n; g += step, c++) {
    const k = tracks.fp[g].indexOf(pid)
    if (k >= 0) { const p = pointAt(board, g, k); out.push({ x: p.x, y: p.y }) }
  }
  return out
}

export function QuestionMap({ board, tracks, q }: { board: Board; tracks: Tracks; q: Question }) {
  const part = partAt(board, q.at)
  const [x0, x1, y0, y1] = boxOf(board, part)
  const fEnd = (p: number, which: 2 | 3) => tracks.pieces[p][which]
  // you: the piece that is yours, looking back from its end (next) or forward from its start (prev)
  const mine = q.dir === 'next' ? trail(board, tracks, q.from, fEnd(q.from, 3), -1) : trail(board, tracks, q.from, fEnd(q.from, 2), 1)
  const cands = q.cands.map(c => q.dir === 'next' ? trail(board, tracks, c[0], fEnd(c[0], 2), 1) : trail(board, tracks, c[0], fEnd(c[0], 3), -1))
  const path = (pts: Pos[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p.x},${p.y}`).join(' ')
  const anchor = mine[0]
  const vbW = x1 - x0 + 2 * PAD, vbH = y1 - y0 + 2 * PAD
  return (
    <svg viewBox={`${x0 - PAD} ${-(y1 + PAD)} ${vbW} ${vbH}`} style={{ display: 'block', width: '100%', borderRadius: 8 }} role="img" aria-label="Where you were and where the candidates are">
      <g transform="scale(1,-1)">
        <Pitch box={[x0, x1, y0, y1]} grass={0.9} />
        {mine.length > 1 && <path d={path(mine)} fill="none" stroke={GOLD} strokeWidth={0.5} strokeLinecap="round" />}
        {cands.map((pts, i) => pts.length > 1 && <path key={`p${i}`} d={path(pts)} fill="none" stroke={CAND[i % 4]} strokeWidth={0.5} strokeLinecap="round" opacity={0.9} />)}
        {anchor && <circle cx={anchor.x} cy={anchor.y} r={1.9} fill="none" stroke={GOLD} strokeWidth={0.6} />}
        {cands.map((pts, i) => pts[0] && (
          <g key={`c${i}`}>
            <circle cx={pts[0].x} cy={pts[0].y} r={1.7} fill={CAND[i % 4]} stroke="#0a1a3a" strokeWidth={0.3} />
            <g transform={`translate(${pts[0].x},${pts[0].y}) scale(1,-1)`}>
              <text x={0} y={0.75} textAnchor="middle" fontSize={2.2} fontWeight={700} fill="#fff">{i + 1}</text>
            </g>
          </g>
        ))}
      </g>
    </svg>
  )
}

interface Props {
  gameId: string
  board: Board
  me: Me
  video?: { seek: (sec: number) => void; pause: () => void }
  isCoach: boolean
}

export default function ClaimPanel({ gameId, board, me, video, isCoach }: Props) {
  const { tracksState, claim, followed, picking, setPicking, question: q } = me
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [map, setMap] = useState(false)
  const tracks = tracksState.status === 'ready' ? tracksState.tracks : null

  const box: React.CSSProperties = { marginTop: 10, paddingTop: 10, borderTop: '1px solid #E4E6EE', fontSize: 12, color: '#1a2340' }
  const small: React.CSSProperties = { fontSize: 11, color: '#5b6377' }

  async function onFile(file: File | undefined) {
    if (!file) return
    setBusy(true); setMsg(null)
    try {
      const raw = JSON.parse(await file.text()) as Record<string, unknown>
      parseTracks(raw, board)
      const err = await uploadTracks(gameId, raw)
      if (err) setMsg(err); else me.reloadTracks()
    } catch (e) { setMsg(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }

  if (tracksState.status === 'loading') return null
  if (!tracks) {
    if (!isCoach) return null
    return (
      <div style={box}>
        <div style={{ fontWeight: 600 }}>This is me</div>
        <div style={small}>
          {tracksState.status === 'error' ? `The tracks file could not be read: ${tracksState.message}` :
            'Players can mark themselves once the tracks file is here. Make it with board_tracks.py from the board data and upload it.'}
        </div>
        <label style={{ ...btn, display: 'inline-block', marginTop: 6, opacity: busy ? 0.6 : 1 }}>
          {busy ? 'Uploading...' : 'Upload tracks'}
          <input type="file" accept=".json,application/json" disabled={busy} style={{ display: 'none' }} onChange={e => { onFile(e.target.files?.[0]); e.target.value = '' }} />
        </label>
        {msg && <div style={{ ...small, color: '#dc2626', marginTop: 6 }}>{msg}</div>}
      </div>
    )
  }
  if (claim.signedIn !== true) return null   // players who are not signed in do not see it

  const hasClaim = claim.claim.seeds.length > 0
  const nOpen = me.openQuestions

  function endPt(p: number, which: 2 | 3) {
    const f = tracks!.pieces[p][which]
    return pointAt(board, f, tracks!.fp[f].indexOf(p))
  }
  function answer(q: Question, to: number | null) {
    claim.answer({ dir: q.dir, from: endPt(q.from, q.dir === 'next' ? 3 : 2), to: to === null ? null : endPt(to, q.dir === 'next' ? 2 : 3) })
  }
  function watch(q: Question) {
    if (!video) return
    video.seek(Math.max(0, q.at - board.shift - (q.dir === 'next' ? 4 : 10)))
  }
  function start() {
    video?.pause()
    setPicking(true)
  }

  return (
    <div style={box}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>This is me</div>
      {!hasClaim && !picking && (
        <>
          <div style={small}>Mark yourself on the video and the tracker follows you for a stretch of the game, asking only where it is unsure. Nobody else sees your answers.</div>
          <button style={{ ...btn, background: NAVY, color: '#fff', marginTop: 6 }} onClick={start}>Find myself</button>
        </>
      )}
      {picking && (
        <div>
          <div style={{ marginBottom: 6 }}>The video is paused. Tap the dot that is nearest to you on the video (or your dot on the board).{hasClaim ? ' Tap a gold one to say it is not you.' : ''} If the dots are not near the players, pause on another moment first.</div>
          <button style={btn} onClick={() => setPicking(false)}>Cancel</button>
        </div>
      )}
      {hasClaim && !picking && followed && (
        <>
          <div style={{ marginBottom: 6 }}>
            You are the <span style={{ background: GOLD, color: '#0a1a3a', borderRadius: 4, padding: '0 5px', fontWeight: 600 }}>gold</span> tag on the video and board: followed for <strong>{fmt(followed.seconds)}</strong>, up to {WINDOW_S} s either side of where you tapped.
            {nOpen > 0 ? ` ${nOpen} question${nOpen === 1 ? '' : 's'} left.` : ' No open questions.'}
          </div>
          {q && (
            <div style={{ background: '#f4f6fb', borderRadius: 10, padding: 8, marginBottom: 8 }}>
              <div style={{ marginBottom: 6 }}>
                Around <strong>{fmt(q.at - board.shift)}</strong> the tracker lost you. Watch the video: when you are back on the picture, which numbered tag is you?
              </div>
              <button style={{ ...btn, background: NAVY, color: '#fff' }} onClick={() => watch(q)} disabled={!video}>▶ Watch this moment</button>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {q.cands.map((c, i) => (
                  <button key={c[0]} style={{ ...btn, background: CAND[i % 4], color: '#fff' }} onClick={() => answer(q, c[0])}>{i + 1}</button>
                ))}
                <button style={btn} onClick={() => answer(q, null)}>None of them</button>
                <button style={btn} onClick={() => me.skip(q.key)}>Skip</button>
              </div>
              <div style={{ ...small, marginTop: 6 }}>
                The numbered tags appear on the video next to the players the tracker thinks could be you.
                <button style={{ ...btn, marginLeft: 6, padding: '2px 8px', fontSize: 11 }} onClick={() => setMap(v => !v)}>{map ? 'Hide board' : 'Board'}</button>
              </div>
              {map && <div style={{ marginTop: 6 }}><QuestionMap board={board} tracks={tracks} q={q} /></div>}
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            <button style={btn} onClick={start}>Mark myself again</button>
            <button style={btn} onClick={() => { if (window.confirm('Remove everything you marked in this game?')) { claim.reset(); me.resetSkips() } }}>Start over</button>
          </div>
        </>
      )}
      <label style={{ display: 'flex', gap: 6, alignItems: 'flex-start', marginTop: 8, ...small }}>
        <input type="checkbox" checked={claim.optIn} onChange={e => claim.setOptIn(e.target.checked)} style={{ marginTop: 2 }} />
        <span>Let the team use my answers to improve the tracking. Off by default. It is then used without your name.</span>
      </label>
      {followed && followed.lost > 0 && <div style={{ ...small, marginTop: 4 }}>{followed.lost} of your answers no longer fit the current tracks and are ignored.</div>}
      {claim.error && <div style={{ ...small, color: '#dc2626', marginTop: 6 }}>{claim.error}</div>}
    </div>
  )
}
