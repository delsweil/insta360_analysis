'use client'

import { frameAt } from '@/lib/board'
import type { Me } from '@/lib/self'

const GOLD = '#ffd400'
const HORIZON = 0.17
const CAMERA_M = 4.8     // camera height above the pitch (m); read off one Buch frame, adjustable with 'spread'
export const CAND_COLOURS = ['#e8780a', '#2b9bd8', '#c64fb0', '#2fa36b']

/** Tags on the video for "This is me": your own tag (gold), the candidates of an open question (numbered), and, while you tap
 *  yourself, a dot for every player. The sideways position comes from the pan and field of view of the moment; the height comes from a camera model with
 *  two guessed numbers (horizon, camera height), so a tag is near its player, not on his feet. The player can nudge them. */
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
  const half = Math.tan((fov * Math.PI) / 360)           // the picture is rendered in the detections' own angle units: no k here
  const focal = (0.5 / Math.tan((k * fov * Math.PI) / 360)) * (16 / 9)   // picture heights per unit of height/depth (ground-map units, with k)
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
    const dl = ((((lon - pan) % 360) + 540) % 360) - 180   // degrees from the centre of the picture, in the video's angle units
    if (Math.abs(dl) >= 80) continue
    const u = 0.5 + Math.tan((dl * Math.PI) / 180) / (2 * half)
    const off = (k * dl * Math.PI) / 180
    if (u < 0.02 || u > 0.98) continue
    // pinhole camera above the pitch: a foot at depth z along the view axis is c / z below the horizon (fractions of the picture height).
    // The defaults were read off one Buch frame; the player can nudge them (Me.adj) for other cameras.
    const z = Math.max(4, Math.hypot(x, y) * Math.cos(off))
    const v = Math.min(0.97, Math.max(0.06, HORIZON - me.adj.up + (CAMERA_M * me.adj.spread * focal) / z))
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
