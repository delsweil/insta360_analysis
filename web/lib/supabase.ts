import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export type Game = {
  id: string
  title: string
  home_team: string
  away_team: string
  date: string
  video_url: string
  duration_sec: number
  uploaded_by: string
  created_at: string
}

export type Annotation = {
  id: string
  game_id: string
  user_id: string
  timestamp_sec: number
  end_timestamp_sec?: number
  label: string
  note?: string
  is_public: boolean
  created_at: string
  profiles?: {
    display_name: string
  }
}

export type UserRole = 'admin' | 'coach' | 'player'

// ── Label system ────────────────────────────────────────────────

export type LabelKey =
  | 'goal_home' | 'goal_away'
  | 'shot_home' | 'shot_away'
  | 'setpiece_home' | 'setpiece_away'
  | 'tactical'

export type LabelDef = {
  key: LabelKey
  display: string     // short name shown in UI
  team: 'home' | 'away' | 'none'
  shape: 'circle' | 'square' | 'triangle' | 'cross'
  coachOnly: boolean
}

export const LABEL_DEFS: LabelDef[] = [
  { key: 'goal_home',     display: 'Goal',      team: 'home', shape: 'circle',   coachOnly: false },
  { key: 'goal_away',     display: 'Goal',      team: 'away', shape: 'circle',   coachOnly: false },
  { key: 'shot_home',     display: 'Shot',      team: 'home', shape: 'square',   coachOnly: false },
  { key: 'shot_away',     display: 'Shot',      team: 'away', shape: 'square',   coachOnly: false },
  { key: 'setpiece_home', display: 'Set piece', team: 'home', shape: 'triangle', coachOnly: false },
  { key: 'setpiece_away', display: 'Set piece', team: 'away', shape: 'triangle', coachOnly: false },
  { key: 'tactical',      display: 'Tactical',  team: 'none', shape: 'cross',    coachOnly: true  },
]

// Navy for home, orange for away, grey for tactical
export function labelColor(key: string): string {
  if (key.endsWith('_home')) return '#0f2972'
  if (key.endsWith('_away')) return '#E8780A'
  return '#4A4F5C'
}

export function labelDef(key: string): LabelDef | undefined {
  return LABEL_DEFS.find(l => l.key === key)
}

export function labelDisplay(key: string, homeTeam?: string, awayTeam?: string): string {
  const def = labelDef(key)
  if (!def) return key
  if (def.team === 'home') return `${def.display} · ${homeTeam ?? 'Home'}`
  if (def.team === 'away') return `${def.display} · ${awayTeam ?? 'Away'}`
  return def.display
}

// ── Role helpers ────────────────────────────────────────────────

export async function getCurrentUserRole(): Promise<UserRole | null> {
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return null
  const { data } = await supabase
    .from('user_roles')
    .select('role')
    .eq('user_id', user.id)
    .single()
  return (data?.role ?? null) as UserRole | null
}

export function isCoachOrAdmin(role: UserRole | null): boolean {
  return role === 'coach' || role === 'admin'
}

export function isAdmin(role: UserRole | null): boolean {
  return role === 'admin'
}
