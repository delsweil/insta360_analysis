// Match statistics: the "*_chartdata.json" files the analysis app writes (one per half), merged into one
// `<game id>.json` in the public Supabase storage bucket `stats`.
import { useCallback, useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

export const STATS_BUCKET = 'stats'
export const CHART_FORMAT = 'football-autopan-chart-data'
export const STATS_FORMAT = 'football-autopan-stats'

export type TeamSeries = {
  name: string
  colour: string
  spread_m: (number | null)[]
  width_m: (number | null)[]
  depth_m: (number | null)[]
  tempo_ms: (number | null)[]
}

/** One half, as the analysis app wrote it (only what the player uses). */
export type HalfStats = {
  title: string
  duration_s: number
  pitch_box_m: [number, number, number, number]
  team_a_attacks_right: boolean
  teams: { A: TeamSeries; B: TeamSeries }
  heat_map: { cell_m: number; x_edges_m: number[]; y_edges_m: number[]; counts: number[][] }
  territory: { minutes: (number | null)[]; value: (number | null)[] }
  minute_mid: (number | null)[]
}

export type Stats = {
  format: typeof STATS_FORMAT
  version: 1
  halves: Record<string, HalfStats>
}

type Obj = Record<string, unknown>
const isObj = (v: unknown): v is Obj => typeof v === 'object' && v !== null && !Array.isArray(v)
const nums = (v: unknown): (number | null)[] =>
  Array.isArray(v) ? v.map(x => (typeof x === 'number' && Number.isFinite(x) ? x : null)) : []

function team(v: unknown): TeamSeries {
  const t = isObj(v) ? v : {}
  return {
    name: String(t.name ?? ''), colour: String(t.colour ?? '#888'),
    spread_m: nums(t.spread_m), width_m: nums(t.width_m), depth_m: nums(t.depth_m), tempo_ms: nums(t.tempo_ms),
  }
}

/** Checks one chart data file and keeps what the player needs. Throws a readable message when it is something else. */
export function parseHalf(raw: unknown): HalfStats {
  if (!isObj(raw) || raw.format !== CHART_FORMAT) throw new Error('Das ist keine Statistik-Datei (…_chartdata.json) der Analyse-App.')
  if (Number(raw.version) !== 1) throw new Error('Diese Statistik-Datei stammt von einer neueren Version der Analyse-App.')
  const teams = isObj(raw.teams) ? raw.teams : {}
  const heat = isObj(raw.heat_map) ? raw.heat_map : {}
  const terr = isObj(raw.territory) ? raw.territory : {}
  const per = isObj(raw.per_minute) ? raw.per_minute : {}
  const box = Array.isArray(raw.pitch_box_m) && raw.pitch_box_m.length === 4 ? (raw.pitch_box_m as number[]) : [-52, 52, 15, 83]
  const counts = Array.isArray(heat.counts) ? (heat.counts as unknown[]).map(r => (Array.isArray(r) ? r.map(Number) : [])) : []
  const half: HalfStats = {
    title: String(raw.title ?? ''),
    duration_s: Number(raw.duration_s) || 0,
    pitch_box_m: [box[0], box[1], box[2], box[3]],
    team_a_attacks_right: raw.team_a_attacks_right !== false,
    teams: { A: team(teams.A), B: team(teams.B) },
    heat_map: {
      cell_m: Number(heat.cell_m) || 1,
      x_edges_m: nums(heat.x_edges_m) as number[],
      y_edges_m: nums(heat.y_edges_m) as number[],
      counts,
    },
    territory: { minutes: nums(terr.minutes), value: nums(terr.value) },
    minute_mid: nums(per.minute_mid),
  }
  if (half.territory.value.length < 2 && half.teams.A.spread_m.length < 2) throw new Error('Die Statistik-Datei enthält keine Daten.')
  return half
}

/** Which half a file is, from its name: "..._half2_chartdata.json" → "2". */
export function halfFromName(name: string): string | null {
  const m = name.match(/half[_\-\s]?([1-4])/i)
  return m ? m[1] : null
}

export function mergeStats(base: Stats | null, half: string, data: HalfStats): Stats {
  return { format: STATS_FORMAT, version: 1, halves: { ...(base?.halves ?? {}), [half]: data } }
}

export function parseStats(raw: unknown): Stats {
  if (!isObj(raw) || raw.format !== STATS_FORMAT || !isObj(raw.halves)) throw new Error('Unbekanntes Statistik-Format.')
  const halves: Record<string, HalfStats> = {}
  for (const [k, v] of Object.entries(raw.halves)) halves[k] = parseHalf(isObj(v) ? { ...v, format: CHART_FORMAT } : v)
  return { format: STATS_FORMAT, version: 1, halves }
}

// The stored file keeps the trimmed halves; parseHalf wants the app's field names, so store them in that shape.
function toStored(s: Stats): Obj {
  const halves: Obj = {}
  for (const [k, h] of Object.entries(s.halves)) {
    halves[k] = {
      title: h.title, duration_s: h.duration_s, pitch_box_m: h.pitch_box_m, team_a_attacks_right: h.team_a_attacks_right,
      teams: h.teams, heat_map: h.heat_map, territory: h.territory, per_minute: { minute_mid: h.minute_mid }, version: 1,
    }
  }
  return { format: STATS_FORMAT, version: 1, halves }
}

export const statsPath = (gameId: string) => `${gameId}.json`

export type StatsState =
  | { status: 'loading' }
  | { status: 'none' }
  | { status: 'error'; message: string }
  | { status: 'ready'; stats: Stats }

/** Loads the statistics of a game; `reload()` fetches them again (after an upload). */
export function useStats(gameId: string | null) {
  const [state, setState] = useState<StatsState>({ status: 'loading' })
  const [version, setVersion] = useState(0)

  useEffect(() => {
    if (!gameId) return
    let dead = false
    ;(async () => {
      try {
        const { data } = supabase.storage.from(STATS_BUCKET).getPublicUrl(statsPath(gameId))
        const res = await fetch(version ? `${data.publicUrl}?v=${version}` : data.publicUrl)
        if (!res.ok) { if (!dead) setState({ status: 'none' }); return }
        const stats = parseStats(await res.json())
        if (!dead) setState(Object.keys(stats.halves).length ? { status: 'ready', stats } : { status: 'none' })
      } catch (e) {
        if (!dead) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
      }
    })()
    return () => { dead = true }
  }, [gameId, version])

  const reload = useCallback(() => setVersion(Date.now()), [])
  return { state, reload }
}

/** The statistics stored for a game right now, or null when there are none. */
export async function fetchStats(gameId: string): Promise<Stats | null> {
  try {
    const { data } = supabase.storage.from(STATS_BUCKET).getPublicUrl(statsPath(gameId))
    const res = await fetch(`${data.publicUrl}?v=${Date.now()}`)
    return res.ok ? parseStats(await res.json()) : null
  } catch { return null }
}

/** Coach: stores the statistics of a game (replaces what was there). Returns an error message or null. */
export async function saveStats(gameId: string, stats: Stats): Promise<string | null> {
  const body = new Blob([JSON.stringify(toStored(stats))], { type: 'application/json' })
  const { error } = await supabase.storage.from(STATS_BUCKET).upload(statsPath(gameId), body, {
    upsert: true, contentType: 'application/json', cacheControl: '60',
  })
  return error ? error.message : null
}

export async function removeStats(gameId: string): Promise<string | null> {
  const { error } = await supabase.storage.from(STATS_BUCKET).remove([statsPath(gameId)])
  return error ? error.message : null
}

/** Reads chart data files chosen by the coach into halves. `taken` = halves already used, to guess a free one. */
export async function readChartFiles(files: File[]): Promise<{ file: string; half: string | null; data?: HalfStats; error?: string }[]> {
  const out: { file: string; half: string | null; data?: HalfStats; error?: string }[] = []
  for (const f of files) {
    try {
      out.push({ file: f.name, half: halfFromName(f.name), data: parseHalf(JSON.parse(await f.text())) })
    } catch (e) {
      out.push({ file: f.name, half: null, error: e instanceof SyntaxError ? 'Keine gültige JSON-Datei.' : (e as Error).message })
    }
  }
  return out
}
