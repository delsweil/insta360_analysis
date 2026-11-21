// lib/matchStrip.ts
// What the timeline shows besides the playhead: who had the play (the "territory" curve of the statistics,
// laid out on the video's own time axis) and the team colours of the goal / shot / set-piece dots.
//
// The statistics call the teams A and B; the game calls them home and away. They are matched by name. When
// no name matches, A is taken as home and `guessed` says so, so the caller can tell the coach.
import type { Game } from '@/lib/supabase'
import type { Stats } from '@/lib/stats'

export type StripColours = { home: string; away: string; fromStats: boolean; guessed: boolean }
/** v > 0: home had the play in the other half, v < 0: away. -1 … 1. */
export type MomentumPoint = { t: number; v: number }
export type Strip = { colours: StripColours; runs: MomentumPoint[][] }

export const HOME_FALLBACK = '#0f2972'
export const AWAY_FALLBACK = '#E8780A'

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9äöüß]/g, '')
function sameTeam(a: string, b: string): boolean {
  const x = norm(a), y = norm(b)
  return !!x && !!y && (x === y || x.includes(y) || y.includes(x))
}

/** Is the statistics' team A the game's home team? null when the names say nothing. */
export function aIsHome(aName: string, bName: string, game: Pick<Game, 'home_team' | 'away_team'>): boolean | null {
  if (sameTeam(aName, game.home_team)) return true
  if (sameTeam(aName, game.away_team)) return false
  if (sameTeam(bName, game.away_team)) return true
  if (sameTeam(bName, game.home_team)) return false
  return null
}

/** Near-white shirts are darkened on a light bar, near-black ones lightened on a dark one (same rule as the statistics card). */
export function visible(c: string, dark: boolean): string {
  const m = c.match(/^#?([0-9a-f]{6})$/i)
  if (!m) return c
  const n = parseInt(m[1], 16)
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  const mix = (t: number, to: number) => '#' + [r, g, b].map(v => Math.round(v + (to - v) * t).toString(16).padStart(2, '0')).join('')
  if (!dark && lum > 0.78) return mix(0.5, 90)
  if (dark && lum < 0.2) return mix(0.5, 200)
  return c
}

/** Colours and momentum for a game; the colours fall back to navy / orange without statistics. */
export function buildStrip(
  stats: Stats | null,
  game: Pick<Game, 'home_team' | 'away_team'> | null,
  halfStart: (half: string) => number | null,
  duration: number,
): Strip {
  const none: Strip = { colours: { home: HOME_FALLBACK, away: AWAY_FALLBACK, fromStats: false, guessed: false }, runs: [] }
  if (!stats || !game) return none
  const keys = Object.keys(stats.halves).sort()
  if (!keys.length) return none
  const first = stats.halves[keys[0]]
  const mapped = aIsHome(first.teams.A.name, first.teams.B.name, game)
  const homeIsA = mapped ?? true
  const colours: StripColours = {
    home: homeIsA ? first.teams.A.colour : first.teams.B.colour,
    away: homeIsA ? first.teams.B.colour : first.teams.A.colour,
    fromStats: true,
    guessed: mapped === null,
  }
  const runs: MomentumPoint[][] = []
  for (const k of keys) {
    const start = halfStart(k)
    if (start == null) continue
    const h = stats.halves[k]
    const run: MomentumPoint[] = []
    h.territory.minutes.forEach((m, i) => {
      const v = h.territory.value[i]
      if (m == null || v == null) return
      const t = start + m * 60
      if (duration > 0 && t > duration) return
      run.push({ t, v: Math.max(-1, Math.min(1, homeIsA ? v : -v)) })
    })
    if (run.length > 1) runs.push(run)
  }
  return { colours, runs }
}

/** Dot colour of an annotation label: the team's own colour, grey for tactical notes. */
export function eventColour(labelKey: string, c: StripColours, dark: boolean): string {
  if (labelKey.endsWith('_home')) return visible(c.home, dark)
  if (labelKey.endsWith('_away')) return visible(c.away, dark)
  return '#4A4F5C'
}

/** Diameter of an event dot: goals stand out, shots and set pieces stay small. */
export function eventSize(labelKey: string, base: number): number {
  if (labelKey.startsWith('goal')) return Math.round(base * 1.5)
  if (labelKey.startsWith('shot')) return base
  return Math.round(base * 0.8)
}
