// Tactical board: the player positions the analysis app exports (`<game>_board.json`, see board_data.py),
// loaded from the public Supabase storage bucket `boards` as `<game id>.json`.
import { useCallback, useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

export const BOARD_BUCKET = 'boards'
export const BOARD_FORMAT = 'football-autopan-board'

export type Box = [number, number, number, number] // x0, x1, y0, y1 in metres (x along the pitch, y = distance from the camera)

export type BoardPart = {
  half: number
  video_start: number
  duration: number | null
  box: Box | null
  a_attacks_right: boolean | null
}

export type Board = {
  title: string
  teams: { A: { name: string; colour: string }; B: { name: string; colour: string } }
  parts: BoardPart[]
  /** frames[i] = [t, pan, fov, x1, y1, team1, x2, y2, team2, ...] (x, y in decimetres; team 0 = A, 1 = B, 2 = other) */
  frames: (number | null)[][]
  times: Float64Array
  box: Box              // whole-game fallback when a part carries none
  reach: Box            // where anyone was ever drawn (0.3-99.7 %): the view never clips a player outside the pitch box
  kAngle: number
  shift: number         // seconds added to the video time before looking a moment up (saved by the coach)
}

export type Raw = Record<string, unknown>

export function parseBoard(raw: Raw): Board {
  if (raw?.format !== BOARD_FORMAT) throw new Error('This is not a board data file (made by the analysis app).')
  if (Number(raw.version) !== 1) throw new Error('This board data file comes from a newer version of the analysis app.')
  const frames = raw.frames as (number | null)[][]
  if (!Array.isArray(frames) || frames.length === 0) throw new Error('The board data file has no moments.')
  const times = new Float64Array(frames.length)
  for (let i = 0; i < frames.length; i++) times[i] = Number(frames[i][0])
  const parts = (raw.parts as BoardPart[]) ?? []
  return {
    title: String(raw.title ?? ''),
    teams: raw.teams as Board['teams'],
    parts,
    frames,
    times,
    box: wholeBox(frames),
    reach: reachBox(frames),
    kAngle: typeof raw.k_angle === 'number' ? raw.k_angle : 0.75,
    shift: typeof raw.sync_shift_s === 'number' ? raw.sync_shift_s : 0,
  }
}

/** Pitch rectangle from where the two teams were seen (the 1st-99th percentile), when the export gave none. */
function wholeBox(frames: (number | null)[][]): Box {
  const xs: number[] = []
  const ys: number[] = []
  const step = Math.max(1, Math.floor(frames.length / 3000))
  for (let i = 0; i < frames.length; i += step) {
    const f = frames[i]
    for (let k = 3; k + 2 < f.length; k += 3) {
      if ((f[k + 2] as number) < 2) { xs.push((f[k] as number) / 10); ys.push((f[k + 1] as number) / 10) }
    }
  }
  if (xs.length < 20) return [-52, 52, 15, 83]
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b)
  const q = (v: number[], p: number) => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))]
  return [q(xs, 0.01) - 1.5, q(xs, 0.99) + 1.5, Math.max(0, q(ys, 0.01) - 1.5), q(ys, 0.99) + 1.5]
}

/** The area people were seen in, every team and the referee included, so a player beside the pitch is not cut off. */
function reachBox(frames: (number | null)[][]): Box {
  const xs: number[] = []
  const ys: number[] = []
  const step = Math.max(1, Math.floor(frames.length / 3000))
  for (let i = 0; i < frames.length; i += step) {
    const f = frames[i]
    for (let k = 3; k + 2 < f.length; k += 3) { xs.push((f[k] as number) / 10); ys.push((f[k + 1] as number) / 10) }
  }
  if (xs.length < 20) return [0, 0, 0, 0]
  xs.sort((a, b) => a - b); ys.sort((a, b) => a - b)
  const q = (v: number[], p: number) => v[Math.min(v.length - 1, Math.max(0, Math.round(p * (v.length - 1))))]
  return [q(xs, 0.003) - 1, q(xs, 0.997) + 1, Math.max(0, q(ys, 0.003) - 1), q(ys, 0.997) + 1]
}

/** Index of the moment nearest to video time t, or -1 when the board has nothing within `maxGap` seconds. */
export function frameAt(b: Board, t: number, maxGap = 1.2): number {
  const n = b.times.length
  let lo = 0
  let hi = n - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (b.times[mid] < t) lo = mid + 1
    else hi = mid
  }
  let best = lo
  if (lo > 0 && Math.abs(b.times[lo - 1] - t) <= Math.abs(b.times[lo] - t)) best = lo - 1
  return Math.abs(b.times[best] - t) <= maxGap ? best : -1
}

export function partAt(b: Board, t: number): BoardPart | null {
  let found: BoardPart | null = null
  for (const p of b.parts) if (p.video_start <= t + 0.01) found = p
  return found
}

export function boxOf(b: Board, part: BoardPart | null): Box {
  return part?.box ?? b.box
}

export type Spots = { x: number; y: number; team: number }

/** Players of one moment, in metres. */
export function playersOf(frame: (number | null)[]): Spots[] {
  const out: Spots[] = []
  for (let k = 3; k + 2 < frame.length; k += 3) {
    out.push({ x: (frame[k] as number) / 10, y: (frame[k + 1] as number) / 10, team: frame[k + 2] as number })
  }
  return out
}

export type BoardState =
  | { status: 'loading' }
  | { status: 'none' }
  | { status: 'error'; message: string }
  | { status: 'ready'; board: Board; raw: Raw }

export function boardPath(gameId: string) {
  return `${gameId}.json`
}

/** Loads the board data of a game; `reload()` fetches it again (after an upload). */
export function useBoard(gameId: string | null) {
  const [state, setState] = useState<BoardState>({ status: 'loading' })
  const [version, setVersion] = useState(0)

  useEffect(() => {
    if (!gameId) return
    let dead = false
    ;(async () => {
      try {
        const { data } = supabase.storage.from(BOARD_BUCKET).getPublicUrl(boardPath(gameId))
        const res = await fetch(version ? `${data.publicUrl}?v=${version}` : data.publicUrl)
        if (!res.ok) { if (!dead) setState({ status: 'none' }); return }   // no file (yet): nothing to show
        const raw = (await res.json()) as Raw
        const board = parseBoard(raw)
        if (!dead) setState({ status: 'ready', board, raw })
      } catch (e) {
        if (!dead) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
      }
    })()
    return () => { dead = true }
  }, [gameId, version])

  const reload = useCallback(() => setVersion(Date.now()), [])
  return { state, reload }
}

/** Coach: puts a board data file (or the same one with a new sync shift) in the bucket. */
export async function uploadBoard(gameId: string, raw: Raw): Promise<string | null> {
  const body = new Blob([JSON.stringify(raw)], { type: 'application/json' })
  const { error } = await supabase.storage.from(BOARD_BUCKET).upload(boardPath(gameId), body, {
    upsert: true, contentType: 'application/json', cacheControl: '60',
  })
  return error ? error.message : null
}
