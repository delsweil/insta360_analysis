'use client'

import { boxOf, frameAt, partAt } from '@/lib/board'
import type { Me } from '@/lib/self'

const GOLD = '#ffd400'
export const CAND_COLOURS = ['#e8780a', '#2b9bd8', '#c64fb0', '#2fa36b']

/** Tags on the video for "This is me": your own tag (gold), the candidates of an open question (numbered), and, while you tap
 *  yourself, a dot for every player. The sideways position comes from the pan and field of view of the moment, so it is about
 *  right; the height on the picture is only a guess from the distance (nearer = lower), so a tag is near its player, not on his feet. */
export default function VideoTags({ me, currentTime }: { me: Me; currentTime: number }) {
  const { board, tracks, followed, question } = me
  if (!board || !tracks || !followed || me.claim.signedIn !== true) return null
  const showAll = !!me.onPick
  if (!showAll && followed.status.size === 0) return null
  const t = currentTime + board.shift
  const idx = frameAt(board, t)
  if (idx < 0) return null
  const fr = board.frames[idx]
  if (fr[1] == null || fr[2] == null) return null
  const k = board.kAngle
  const pan = fr[1] as number, fov = fr[2] as number
  const half = Math.tan((k * fov * Math.PI) / 360)
  const [, , y0, y1] = boxOf(board, partAt(board, t))
  const candNo = new Map<number, number>()
  question?.cands.forEach((c, i) => candNo.set(c[0], i))
  const colours = [board.teams.A?.colour ?? '#d62828', board.teams.B?.colour ?? '#1d4ed8']
  const tags: React.ReactNode[] = []
  for (let j = 0; 3 + 3 * j + 2 < fr.length; j++) {
    const x = (fr[3 + 3 * j] as number) / 10, y = (fr[4 + 3 * j] as number) / 10, team = fr[5 + 3 * j] as number
    if (team === 2 || team === 3) continue
    const pid = tracks.fp[idx][j]
    const mine = followed.status.has(pid)
    const cand = candNo.get(pid)
    if (!mine && cand === undefined && !showAll) continue
    const lon = (Math.atan2(x, y) * 180) / Math.PI / k
    const off = (k * (lon - pan) * Math.PI) / 180
    if (Math.abs(off) >= Math.PI / 2.2) continue
    const u = 0.5 + Math.tan(off) / (2 * half)
    if (u < 0.02 || u > 0.98) continue
    const near = Math.min(1, Math.max(0, (y - y0) / Math.max(1, y1 - y0)))
    const v = 0.9 - 0.42 * near
    const size = mine ? 30 : cand !== undefined ? 26 : 18
    const onClick = me.onPick ? (e: React.MouseEvent) => { e.stopPropagation(); me.onPick!(idx, j) } : undefined
    tags.push(
      <div key={j} onClick={onClick} style={{
        position: 'absolute', left: `${u * 100}%`, top: `${v * 100}%`, width: size, height: size, marginLeft: -size / 2, marginTop: -size / 2,
        borderRadius: '50%', boxSizing: 'border-box', display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: mine ? `3px solid ${GOLD}` : '2px solid #fff', fontFamily: 'DM Sans, sans-serif', fontWeight: 700, fontSize: 13, color: '#fff',
        background: cand !== undefined ? CAND_COLOURS[cand % 4] : mine ? 'rgba(255,212,0,0.18)' : colours[team === 1 || team === 5 ? 1 : 0],
        boxShadow: '0 0 0 1px rgba(0,0,0,0.5)', cursor: onClick ? 'pointer' : 'default', pointerEvents: onClick ? 'auto' : 'none',
        opacity: mine || cand !== undefined ? 1 : 0.85,
      }}>{cand !== undefined ? cand + 1 : ''}</div>,
    )
  }
  return <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 3 }}>{tags}</div>
}
