// Training sessions: types and the small pure helpers the pages use.

export type SessionVisibility = 'private' | 'shared'

export type TrainingSession = {
  id: string
  title: string
  session_date: string | null
  focus: string
  visibility: SessionVisibility
  tags: string[]
  created_by: string
  created_at: string
  updated_at: string
}

export const MAX_TAGS = 12
export const MAX_TAG_LEN = 30

/** Turn what the coach typed ("U13, #Pressing  u13 ; Spielform") into a clean, de-duplicated tag list. */
export function parseTags(input: string): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of input.split(/[,;\n]+/)) {
    const t = raw.trim().replace(/^#+/, '').replace(/\s+/g, ' ').trim().slice(0, MAX_TAG_LEN)
    if (!t) continue
    const k = t.toLocaleLowerCase('de-DE')
    if (seen.has(k)) continue
    seen.add(k)
    out.push(t)
    if (out.length >= MAX_TAGS) break
  }
  return out
}

export type SessionScope = 'all' | 'mine' | 'shared'

export function filterSessions(
  list: TrainingSession[],
  opts: { scope: SessionScope; tag: string | null; userId: string | null; query?: string },
): TrainingSession[] {
  const q = (opts.query ?? '').trim().toLocaleLowerCase('de-DE')
  return list.filter(s => {
    if (opts.scope === 'mine' && s.created_by !== opts.userId) return false
    if (opts.scope === 'shared' && s.visibility !== 'shared') return false
    if (opts.tag && !s.tags.some(t => t.toLocaleLowerCase('de-DE') === opts.tag!.toLocaleLowerCase('de-DE'))) return false
    if (q) {
      const hay = `${s.title} ${s.focus} ${s.tags.join(' ')}`.toLocaleLowerCase('de-DE')
      if (!hay.includes(q)) return false
    }
    return true
  })
}

/** All tags used across the sessions with how often, most used first (ties alphabetical). Case-insensitive. */
export function tagCounts(list: TrainingSession[]): { tag: string; count: number }[] {
  const m = new Map<string, { tag: string; count: number }>()
  for (const s of list) {
    for (const t of s.tags) {
      const k = t.toLocaleLowerCase('de-DE')
      const cur = m.get(k)
      if (cur) cur.count++
      else m.set(k, { tag: t, count: 1 })
    }
  }
  return [...m.values()].sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag, 'de'))
}

/** Newest planned date first; sessions without a date go by when they were created. */
export function sortSessions(list: TrainingSession[]): TrainingSession[] {
  const key = (s: TrainingSession) => s.session_date ?? s.created_at.slice(0, 10)
  return [...list].sort((a, b) => key(b).localeCompare(key(a)) || b.created_at.localeCompare(a.created_at))
}

// ── Forms (exercises and small-sided games), sections and scenes ───────────────

import type { SceneBackground, SceneShape, Scene } from './scene.ts'

export type FormKind = 'exercise' | 'spielform'
export const FORM_KIND_LABELS: Record<FormKind, string> = { exercise: 'Übung', spielform: 'Spielform' }

/** A titled block of text in a form. A form has as many as the coach wants, in any order. */
export type Section = { id: string; title: string; text: string }

export type SessionForm = {
  id: string
  session_id: string
  position: number
  kind: FormKind
  title: string
  duration_min: number | null
  players: string
  sections: Section[]
  created_at: string
  updated_at: string
}

export type FormSceneRow = {
  id: string
  form_id: string
  position: number
  background: SceneBackground
  shapes: SceneShape[]
  caption: string
  created_at?: string
  updated_at?: string
}

/** Titles offered as one-tap starters. Nothing forces them. */
export const STARTER_SECTIONS = ['Warum', 'Idee', 'Aufbau', 'Ablauf', 'Fragen', 'Variationen']

const sid = () => Math.random().toString(36).slice(2, 10)
export function newSection(title = ''): Section { return { id: sid(), title, text: '' } }

/** Starter titles not yet used in this form (case-insensitive). */
export function unusedStarters(sections: Section[]): string[] {
  const used = new Set(sections.map(s => s.title.trim().toLocaleLowerCase('de-DE')))
  return STARTER_SECTIONS.filter(t => !used.has(t.toLocaleLowerCase('de-DE')))
}

export function moveItem<T>(arr: T[], from: number, to: number): T[] {
  if (from === to || from < 0 || to < 0 || from >= arr.length || to >= arr.length) return arr
  const out = arr.slice()
  const [x] = out.splice(from, 1)
  out.splice(to, 0, x)
  return out
}

/** Renumber positions 0..n-1 in the current order. */
export function reindex<T extends { position: number }>(arr: T[]): T[] { return arr.map((x, i) => (x.position === i ? x : { ...x, position: i })) }

/** Parse a duration the coach typed: whole minutes between 1 and 240, otherwise none. */
export function parseMinutes(input: string): number | null {
  const n = parseInt(input.replace(/[^\d]/g, ''), 10)
  return Number.isFinite(n) && n >= 1 && n <= 240 ? n : null
}

export function totalMinutes(forms: Pick<SessionForm, 'duration_min'>[]): number { return forms.reduce((a, f) => a + (f.duration_min ?? 0), 0) }

export function sceneOf(r: FormSceneRow): Scene { return { background: r.background, shapes: r.shapes ?? [], caption: r.caption ?? '' } }
