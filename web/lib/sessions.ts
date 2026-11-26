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
