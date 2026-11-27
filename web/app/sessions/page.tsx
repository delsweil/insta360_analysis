'use client'

export const dynamic = 'force-dynamic'

import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import {
  filterSessions, sortSessions, tagCounts,
  type TrainingSession, type SessionScope,
} from '@/lib/sessions'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

export default function SessionsPage() {
  const router = useRouter()
  const [user, setUser] = useState<{ id: string; email?: string } | null>(null)
  const [isCoach, setIsCoach] = useState(false)
  const [role, setRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [sessions, setSessions] = useState<TrainingSession[]>([])
  const [names, setNames] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scope, setScope] = useState<SessionScope>('all')
  const [tag, setTag] = useState<string | null>(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      setUser({ id: user.id, email: user.email })
      const r = await getCurrentUserRole()
      setRole(r)
      const coach = isCoachOrAdmin(r)
      setIsCoach(coach)
      if (coach) {
        const { data, error } = await supabase.from('training_sessions').select('*')
        if (error) setError(error.message)
        else {
          const list = (data ?? []) as TrainingSession[]
          setSessions(list)
          const ids = [...new Set(list.map(s => s.created_by))]
          if (ids.length) {
            const { data: profs } = await supabase.from('profiles').select('id, display_name').in('id', ids)
            const m: Record<string, string> = {}
            for (const p of (profs ?? []) as { id: string; display_name?: string | null }[]) if (p.display_name) m[p.id] = p.display_name
            setNames(m)
          }
        }
      }
      setLoading(false)
    }
    load()
  }, [router])

  const shown = useMemo(
    () => sortSessions(filterSessions(sessions, { scope, tag, userId: user?.id ?? null, query })),
    [sessions, scope, tag, query, user],
  )
  const tags = useMemo(() => tagCounts(sessions), [sessions])

  async function createSession() {
    setCreating(true); setError(null)
    const { data, error } = await supabase.from('training_sessions').insert({}).select().single()
    if (error || !data) { setError(error?.message ?? 'Konnte nicht angelegt werden.'); setCreating(false); return }
    router.push(`/sessions/${(data as TrainingSession).id}`)
  }

  const handleSignOut = async () => { await supabase.auth.signOut(); router.push('/login') }

  if (loading) return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C' }}>
      Lädt …
    </div>
  )

  const pill = (active: boolean): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '6px 14px', borderRadius: 99, cursor: 'pointer',
    border: `1px solid ${active ? NAVY : '#E4E6EE'}`, background: active ? NAVY : '#fff', color: active ? '#fff' : '#4A4F5C',
    fontFamily: 'DM Sans, sans-serif',
  })

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={role} email={user?.email} onSignOut={handleSignOut} />
      <div style={{ padding: 20, maxWidth: 900, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, gap: 12, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 28, color: NAVY, letterSpacing: '0.02em', lineHeight: 1 }}>
              Trainingseinheiten
            </div>
            <div style={{ fontSize: 12, color: MUTED, marginTop: 2 }}>
              {isCoach ? `${sessions.length} ${sessions.length === 1 ? 'Einheit' : 'Einheiten'}` : ''}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <button onClick={() => router.push('/')} style={{ ...pill(false), borderRadius: 8 }}>← Spiele</button>
            {isCoach && <button data-testid="to-library" onClick={() => router.push('/library')} style={{ ...pill(false), borderRadius: 8 }}>Bibliothek</button>}
            {isCoach && (
              <button
                data-testid="new-session"
                onClick={createSession}
                disabled={creating}
                style={{ fontSize: 12, fontWeight: 600, padding: '7px 16px', borderRadius: 8, border: 'none', background: NAVY, color: '#fff', whiteSpace: 'nowrap', cursor: creating ? 'default' : 'pointer', opacity: creating ? 0.6 : 1, fontFamily: 'DM Sans, sans-serif' }}
              >
                + Neue Einheit
              </button>
            )}
          </div>
        </div>

        {error && (
          <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', color: '#991b1b', borderRadius: 10, padding: '10px 14px', fontSize: 13, marginBottom: 12 }}>
            {error}
          </div>
        )}

        {!isCoach ? (
          <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '40px 20px', textAlign: 'center', color: MUTED, fontSize: 14 }}>
            Trainingseinheiten sind für Trainerinnen und Trainer. Spielerinnen und Spieler erhalten sie über einen Link.
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', marginBottom: 10 }}>
              {([['all', 'Alle'], ['mine', 'Meine'], ['shared', 'Geteilt']] as [SessionScope, string][]).map(([k, label]) => (
                <button key={k} data-testid={`scope-${k}`} onClick={() => setScope(k)} style={pill(scope === k)}>{label}</button>
              ))}
              <input
                data-testid="search"
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Suchen …"
                style={{ flex: 1, minWidth: 140, fontSize: 13, padding: '7px 12px', borderRadius: 8, border: '1px solid #E4E6EE', fontFamily: 'DM Sans, sans-serif', outline: 'none' }}
              />
            </div>

            {tags.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 14 }}>
                {tags.map(t => {
                  const active = tag !== null && tag.toLocaleLowerCase('de-DE') === t.tag.toLocaleLowerCase('de-DE')
                  return (
                    <button
                      key={t.tag}
                      data-testid="tag-chip"
                      onClick={() => setTag(active ? null : t.tag)}
                      style={{ fontSize: 11, fontWeight: 600, padding: '3px 10px', borderRadius: 99, cursor: 'pointer', border: 'none', background: active ? '#E8780A' : '#e8edf8', color: active ? '#fff' : NAVY, fontFamily: 'DM Sans, sans-serif' }}
                    >
                      {t.tag} <span style={{ opacity: 0.6 }}>{t.count}</span>
                    </button>
                  )
                })}
              </div>
            )}

            {shown.length === 0 ? (
              <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '40px 20px', textAlign: 'center', color: MUTED, fontSize: 14 }}>
                {sessions.length === 0 ? 'Noch keine Trainingseinheiten. Lege die erste an.' : 'Keine Einheit passt zu diesem Filter.'}
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {shown.map(s => {
                  const mine = s.created_by === user?.id
                  return (
                    <div
                      key={s.id}
                      data-testid="session-card"
                      onClick={() => router.push(`/sessions/${s.id}`)}
                      style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '14px 18px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}
                      onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 4px 20px rgba(15,41,114,0.1)' }}
                      onMouseLeave={e => { e.currentTarget.style.boxShadow = 'none' }}
                    >
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 18, color: NAVY, letterSpacing: '0.02em', lineHeight: 1.1 }}>
                          {s.title}
                        </div>
                        <div style={{ fontSize: 12, color: MUTED, marginTop: 3 }}>
                          {s.session_date
                            ? new Date(s.session_date + 'T12:00:00').toLocaleDateString('de-DE', { day: '2-digit', month: 'long', year: 'numeric' })
                            : 'Ohne Datum'}
                          {' · '}{mine ? 'von dir' : (names[s.created_by] ?? 'von einem Kollegen')}
                          {s.focus ? <> · {s.focus}</> : null}
                        </div>
                        {s.tags.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 7 }}>
                            {s.tags.map(t => (
                              <span key={t} style={{ fontSize: 10.5, fontWeight: 600, color: NAVY, background: '#e8edf8', padding: '2px 8px', borderRadius: 99 }}>{t}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      <span
                        data-testid="visibility-pill"
                        style={{ fontSize: 11, fontWeight: 600, padding: '4px 12px', borderRadius: 99, flexShrink: 0, color: s.visibility === 'shared' ? '#166534' : '#6b7280', background: s.visibility === 'shared' ? '#dcfce7' : '#f1f2f5' }}
                      >
                        {s.visibility === 'shared' ? 'Geteilt' : 'Privat'}
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
