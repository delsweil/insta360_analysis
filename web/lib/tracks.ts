// "This is me": pieces of players that the tracker followed (`<game>_tracks.json`, written by board_tracks.py), and the
// logic that follows one player through them from the points the player marked. Nothing here names a person.
import { useCallback, useEffect, useMemo, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { BOARD_BUCKET, type Board } from '@/lib/board'

export const TRACKS_FORMAT = 'football-autopan-tracks'

/** [piece, chance, distance m, gap s] */
export type Cand = [number, number, number, number]
export type Link = { c: Cand[]; s: number }          // s = 1: the tracker is sure of the best candidate
export type Tracks = {
  tracker: string
  pieces: [number, number, number, number][]          // id, side (0 A, 1 B, 2 other, 3 referee), first moment, last moment
  fp: number[][]                                      // fp[moment][k] = piece of the k-th player of that moment of the board file
  next: Record<string, Link>
  prev: Record<string, Link>
}

/** A place and moment the player marked: board time (seconds, the first column of the board file) and metres. */
export type Pt = { t: number; x: number; y: number }
export type LinkAnswer = { dir: 'next' | 'prev'; from: Pt; to: Pt | null }   // to = null: none of the candidates is me
export type Claim = { seeds: Pt[]; rejects: Pt[]; links: LinkAnswer[] }
export const EMPTY_CLAIM: Claim = { seeds: [], rejects: [], links: [] }

export function tracksPath(gameId: string) {
  return `${gameId}.tracks.json`
}

export function parseTracks(raw: Record<string, unknown>, board: Board): Tracks {
  if (raw?.format !== TRACKS_FORMAT) throw new Error('This is not a tracks file (made by board_tracks.py).')
  if (Number(raw.version) !== 1) throw new Error('This tracks file comes from a newer version of the analysis app.')
  const fp = raw.fp as number[][]
  if (!Array.isArray(fp) || fp.length !== board.frames.length) throw new Error('The tracks file does not belong to this board data (different number of moments).')
  return {
    tracker: String(raw.tracker ?? ''),
    pieces: raw.pieces as Tracks['pieces'],
    fp,
    next: (raw.next as Tracks['next']) ?? {},
    prev: (raw.prev as Tracks['prev']) ?? {},
  }
}

export type TracksState = { status: 'loading' } | { status: 'none' } | { status: 'error'; message: string } | { status: 'ready'; tracks: Tracks; raw: Record<string, unknown> }

export function useTracks(gameId: string | null, board: Board | null) {
  const [state, setState] = useState<TracksState>({ status: 'loading' })
  const [version, setVersion] = useState(0)
  useEffect(() => {
    if (!gameId || !board) return
    let dead = false
    ;(async () => {
      try {
        const { data } = supabase.storage.from(BOARD_BUCKET).getPublicUrl(tracksPath(gameId))
        const res = await fetch(version ? `${data.publicUrl}?v=${version}` : data.publicUrl)
        if (!res.ok) { if (!dead) setState({ status: 'none' }); return }
        const raw = (await res.json()) as Record<string, unknown>
        const tracks = parseTracks(raw, board)
        if (!dead) setState({ status: 'ready', tracks, raw })
      } catch (e) {
        if (!dead) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
      }
    })()
    return () => { dead = true }
  }, [gameId, board, version])
  const reload = useCallback(() => setVersion(Date.now()), [])
  return { state, reload }
}

/** Coach: puts the tracks file next to the board file. */
export async function uploadTracks(gameId: string, raw: Record<string, unknown>): Promise<string | null> {
  const body = new Blob([JSON.stringify(raw)], { type: 'application/json' })
  const { error } = await supabase.storage.from(BOARD_BUCKET).upload(tracksPath(gameId), body, {
    upsert: true, contentType: 'application/json', cacheControl: '60',
  })
  return error ? error.message : null
}

// ---- looking things up ----

export type Index = { first: Map<number, [number, number]>; last: Map<number, [number, number]> }

/** For each piece, the (moment, player index) where it starts and where it ends. */
export function buildIndex(tracks: Tracks): Index {
  const first = new Map<number, [number, number]>()
  const last = new Map<number, [number, number]>()
  tracks.fp.forEach((row, f) => row.forEach((pid, k) => {
    if (!first.has(pid)) first.set(pid, [f, k])
    last.set(pid, [f, k])
  }))
  return { first, last }
}

export function pointAt(board: Board, f: number, k: number): Pt {
  const fr = board.frames[f]
  return { t: board.times[f], x: (fr[3 + 3 * k] as number) / 10, y: (fr[4 + 3 * k] as number) / 10 }
}

/** The piece whose player was at (or within 2 m of) the marked point; -1 when there is none. */
export function resolve(board: Board, tracks: Tracks, pt: Pt): number {
  let lo = 0, hi = board.times.length - 1
  while (lo < hi) { const m = (lo + hi) >> 1; if (board.times[m] < pt.t) lo = m + 1; else hi = m }
  let best = lo
  if (lo > 0 && Math.abs(board.times[lo - 1] - pt.t) <= Math.abs(board.times[lo] - pt.t)) best = lo - 1
  if (Math.abs(board.times[best] - pt.t) > 1.0) return -1
  const fr = board.frames[best]
  let bd = 2.0, bp = -1
  for (let k = 0; 3 + 3 * k + 2 < fr.length; k++) {
    const d = Math.hypot((fr[3 + 3 * k] as number) / 10 - pt.x, (fr[4 + 3 * k] as number) / 10 - pt.y)
    if (d < bd) { bd = d; bp = tracks.fp[best][k] }
  }
  return bp
}

/** How far from a point the player marked the tracker follows them (seconds each way): a whole game would mean far too many questions. */
export const WINDOW_S = 90

export type Status = 'seed' | 'confirmed' | 'auto'
export type Question = { key: string; dir: 'next' | 'prev'; from: number; at: number; cands: Cand[] }
export type Followed = { status: Map<number, Status>; questions: Question[]; lost: number; seconds: number }

/** The pieces that are the player: those they marked, those they confirmed, and those the tracker is sure about; plus the open questions. */
export function follow(board: Board, tracks: Tracks, claim: Claim): Followed {
  const status = new Map<number, Status>()
  const t0 = (p: number) => board.times[tracks.pieces[p][2]]
  const t1 = (p: number) => board.times[tracks.pieces[p][3]]
  const rejected = new Set(claim.rejects.map(r => resolve(board, tracks, r)).filter(p => p >= 0))
  const answers = new Map<string, number>()
  let lost = 0
  for (const l of claim.links) {
    const p = resolve(board, tracks, l.from)
    const q = l.to ? resolve(board, tracks, l.to) : -1
    if (p < 0 || (l.to && q < 0)) { lost++; continue }
    answers.set(`${l.dir}:${p}`, q)
  }
  const queue: number[] = []
  const seedT = claim.seeds.map(s => s.t)
  const inWindow = (q: number) => seedT.some(t => t0(q) <= t + WINDOW_S && t1(q) >= t - WINDOW_S)
  const overlaps = (q: number) => { for (const c of status.keys()) if (Math.min(t1(c), t1(q)) - Math.max(t0(c), t0(q)) > 1.0) return true; return false }
  const add = (q: number, st: Status) => {
    if (q < 0 || status.has(q) || rejected.has(q) || tracks.pieces[q][1] > 1 || !inWindow(q) || overlaps(q)) return false
    status.set(q, st); queue.push(q); return true
  }
  for (const s of claim.seeds) { const p = resolve(board, tracks, s); if (p >= 0) add(p, 'seed'); else lost++ }
  const questions: Question[] = []
  while (queue.length) {
    const p = queue.shift()!
    for (const dir of ['next', 'prev'] as const) {
      const key = `${dir}:${p}`
      const ans = answers.get(key)
      if (ans !== undefined) { if (ans >= 0) add(ans, 'confirmed'); continue }
      const L = (dir === 'next' ? tracks.next : tracks.prev)[String(p)]
      if (!L) continue
      if (L.s === 1) { add(L.c[0][0], 'auto'); continue }
      questions.push({ key, dir, from: p, at: dir === 'next' ? t1(p) : t0(p), cands: L.c })
    }
  }
  const open = questions.filter(q => status.has(q.from)
    && !q.cands.some(c => status.has(c[0]))
    && q.cands.filter(c => !rejected.has(c[0]) && c[0] >= 0).length > 0)
  open.sort((a, b) => a.at - b.at)
  let seconds = 0
  for (const p of status.keys()) seconds += t1(p) - t0(p)
  return { status, questions: open, lost, seconds }
}

/** Hook-friendly memo of follow(). */
export function useFollowed(board: Board | null, tracks: Tracks | null, claim: Claim) {
  return useMemo(() => (board && tracks ? follow(board, tracks, claim) : null), [board, tracks, claim])
}
