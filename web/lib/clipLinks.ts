import { supabase } from '@/lib/supabase'
import type { Annotation } from '@/lib/supabase'

// How far outside a shared highlight a viewer may cut their own clip (seconds). The SQL function create_clip_link uses the same numbers.
export const CLIP_MARGIN_SEC = 15
export const CLIP_MAX_SEC = 120

// A highlight's own stretch: the build-up before it (context start, if the coach set one) to its end.
export function highlightStart(a: Annotation) {
  return a.label === 'tactical' && a.context_start_sec != null ? a.context_start_sec : a.timestamp_sec
}
export function highlightEnd(a: Annotation) {
  return a.end_timestamp_sec ?? a.timestamp_sec + 10
}

// The stretches of the game a viewer may clip from: every shared highlight with a margin, overlapping ones merged.
export function allowedWindows(annotations: Annotation[]): [number, number][] {
  const raw = annotations
    .map(a => [Math.max(0, highlightStart(a) - CLIP_MARGIN_SEC), highlightEnd(a) + CLIP_MARGIN_SEC] as [number, number])
    .sort((x, y) => x[0] - y[0])
  const merged: [number, number][] = []
  for (const w of raw) {
    const last = merged[merged.length - 1]
    if (last && w[0] <= last[1]) last[1] = Math.max(last[1], w[1])
    else merged.push([w[0], w[1]])
  }
  return merged
}

// The allowed stretch that holds time t, or the nearest one.
export function windowAround(annotations: Annotation[], t: number): [number, number] | null {
  const ws = allowedWindows(annotations)
  if (ws.length === 0) return null
  const inside = ws.find(w => t >= w[0] && t <= w[1])
  if (inside) return inside
  return ws.reduce((best, w) => {
    const d = (x: [number, number]) => (t < x[0] ? x[0] - t : t - x[1])
    return d(w) < d(best) ? w : best
  })
}

// Coach drawings (tactical annotations with shapes) that are on screen at some point inside [start, end].
export function drawingsIn(annotations: Annotation[], start: number, end: number) {
  return annotations.filter(a => {
    if (a.label !== 'tactical' || !a.shapes?.length) return false
    const to = a.annotation_end_sec ?? a.timestamp_sec + 1.5
    return to >= start && a.timestamp_sec <= end
  })
}

type Made = { url?: string; error?: string }

// Makes a clip link. Signed in: the link row is written directly. From a shared link (parentToken): the server function checks that
// the stretch lies inside what was shared and writes the row, so nobody can cut clips from parts of the game that were not shared.
export async function createClipLink(o: {
  gameId: string
  parentToken?: string
  annotations: Annotation[]
  start: number
  end: number
  includeDrawings: boolean
}): Promise<Made> {
  const start = Math.round(o.start * 10) / 10
  const end = Math.round(o.end * 10) / 10
  if (o.parentToken) {
    const { data, error } = await supabase.rpc('create_clip_link', {
      parent_token: o.parentToken, start_sec: start, end_sec: end, include_drawings: o.includeDrawings,
    })
    if (error || !data) return { error: error?.message ?? 'Could not create the link.' }
    return { url: `${window.location.origin}/share/${data}` }
  }
  const ids = o.includeDrawings ? drawingsIn(o.annotations, start, end).map(a => a.id) : []
  const { data, error } = await supabase
    .from('share_links')
    .insert({ game_id: o.gameId, annotation_ids: ids, clip_start_sec: start, clip_end_sec: end, clip_only: true })
    .select('token')
    .single()
  if (data?.token) return { url: `${window.location.origin}/share/${data.token}` }
  return { error: error?.message ?? 'Could not create the link.' }
}

// Phones get the share sheet; everything else gets the link copied.
export async function shareOrCopy(url: string, title: string): Promise<'shared' | 'copied' | 'failed'> {
  try {
    if (typeof navigator !== 'undefined' && navigator.share) {
      await navigator.share({ title, url })
      return 'shared'
    }
  } catch (e) {
    if ((e as Error)?.name === 'AbortError') return 'failed' // the person closed the share sheet
  }
  try {
    await navigator.clipboard.writeText(url)
    return 'copied'
  } catch {
    return 'failed'
  }
}
