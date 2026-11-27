'use client'

export const dynamic = 'force-dynamic'

import React, { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin, isAdmin } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import SceneCanvas from '@/components/SceneCanvas'
import { ConceptChips, phaseColor } from '@/components/ConceptPicker'
import { PHASES, groupsOfPhase, matchesFilter, conceptById, type PhaseId } from '@/lib/gameModel'
import { copyOfForm, copyOfScene } from '@/lib/library'
import { FORM_KIND_LABELS, sceneOf, type FormSceneRow, type SessionForm, type TrainingSession } from '@/lib/sessions'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

type Tab = 'library' | 'review'

export default function LibraryPage() {
  const router = useRouter()
  const [user, setUser] = useState<{ id: string; email?: string } | null>(null)
  const [role, setRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [loading, setLoading] = useState(true)
  const [forms, setForms] = useState<SessionForm[]>([])
  const [first, setFirst] = useState<Record<string, FormSceneRow>>({})
  const [mySessions, setMySessions] = useState<TrainingSession[]>([])
  const [error, setError] = useState('')
  const [tab, setTab] = useState<Tab>('library')
  const [phase, setPhase] = useState<PhaseId | null>(null)
  const [concept, setConcept] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [target, setTarget] = useState('')
  const [done, setDone] = useState<{ title: string; sessionId: string } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    const qs = new URLSearchParams(window.location.search)
    const c = qs.get('concept'), s = qs.get('session')
    // eslint-disable-next-line react-hooks/set-state-in-effect -- initial filters come from the address once
    if (c && conceptById(c)) { setConcept(c); setPhase(conceptById(c)!.phase) }
    if (s) setTarget(s)
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      setUser({ id: user.id, email: user.email })
      const r = await getCurrentUserRole(); setRole(r)
      if (!isCoachOrAdmin(r)) { setLoading(false); return }
      const { data: f, error: e } = await supabase.from('session_forms').select('*').in('library_status', ['library', 'proposed', 'archived'])
      if (e) setError(e.message)
      const list = (f ?? []) as SessionForm[]
      setForms(list)
      if (list.length) {
        const { data: sc } = await supabase.from('form_scenes').select('*').in('form_id', list.map(x => x.id))
        const m: Record<string, FormSceneRow> = {}
        for (const row of ((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position)) if (!m[row.form_id]) m[row.form_id] = row
        setFirst(m)
      }
      const { data: ms } = await supabase.from('training_sessions').select('*').eq('created_by', user.id).order('created_at', { ascending: false })
      setMySessions((ms ?? []) as TrainingSession[])
      setLoading(false)
    }
    load()
  }, [router])

  const admin = isAdmin(role)
  const inLibrary = useMemo(() => forms.filter(f => f.library_status === 'library'), [forms])
  const proposed = useMemo(() => forms.filter(f => f.library_status === 'proposed'), [forms])
  const archived = useMemo(() => forms.filter(f => f.library_status === 'archived'), [forms])
  const q = query.trim().toLocaleLowerCase('de-DE')
  const shown = useMemo(() => inLibrary.filter(f => matchesFilter(f.concepts ?? [], { phase, concept }) && (!q || `${f.title} ${f.players}`.toLocaleLowerCase('de-DE').includes(q))), [inLibrary, phase, concept, q])
  const countOf = (id: string) => inLibrary.filter(f => (f.concepts ?? []).includes(id)).length

  async function adopt(f: SessionForm) {
    if (!target) return
    setBusy(f.id); setError(''); setDone(null)
    const { data: sib } = await supabase.from('session_forms').select('position').eq('session_id', target)
    const pos = ((sib ?? []) as { position: number }[]).reduce((m, r) => Math.max(m, r.position), -1) + 1
    const { data: nf, error: e } = await supabase.from('session_forms').insert(copyOfForm(f, target, pos)).select().single()
    if (e || !nf) { setError(e?.message ?? 'Konnte nicht übernommen werden.'); setBusy(null); return }
    const { data: sc } = await supabase.from('form_scenes').select('*').eq('form_id', f.id)
    for (const row of ((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position)) {
      const { error: e2 } = await supabase.from('form_scenes').insert(copyOfScene(row, (nf as SessionForm).id))
      if (e2) { setError(e2.message); break }
    }
    setBusy(null); setDone({ title: f.title, sessionId: target })
  }

  const handleSignOut = async () => { await supabase.auth.signOut(); router.push('/login') }
  const chip = (active: boolean, color = NAVY): React.CSSProperties => ({ fontSize: 12, fontWeight: 600, padding: '6px 11px', minHeight: 32, borderRadius: 99, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', border: `1.5px solid ${active ? color : '#E4E6EE'}`, background: active ? color : '#fff', color: active ? '#fff' : '#33384a' })

  if (loading) return <div style={{ minHeight: '100vh', background: '#F8F8F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C' }}>Lädt …</div>

  const card = (f: SessionForm, mode: 'library' | 'review') => {
    const fs = first[f.id]
    return (
      <div key={f.id} data-testid="library-card-item" style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 10, display: 'flex', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div onClick={() => router.push(`/sessions/${f.session_id}/forms/${f.id}`)} style={{ width: 120, flex: '0 0 120px', borderRadius: 8, overflow: 'hidden', border: '1px solid #eceef3', background: '#fbfbf8', cursor: 'pointer' }}>
          {fs ? <SceneCanvas scene={sceneOf(fs)} look="clean" testId="lib-thumb" /> : <div style={{ height: 80 }} />}
        </div>
        <div style={{ flex: 1, minWidth: 180 }}>
          <div onClick={() => router.push(`/sessions/${f.session_id}/forms/${f.id}`)} style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 19, color: NAVY, letterSpacing: '0.02em', lineHeight: 1.1, cursor: 'pointer' }}>{f.title}</div>
          <div style={{ fontSize: 12, color: MUTED, margin: '3px 0 6px' }}>{FORM_KIND_LABELS[f.kind]}{f.duration_min ? ` · ${f.duration_min} Min.` : ''}{f.players ? ` · ${f.players}` : ''}</div>
          <ConceptChips ids={f.concepts ?? []} />
        </div>
        <div style={{ display: 'flex', gap: 6, flexDirection: 'column', alignItems: 'stretch', minWidth: 150 }}>
          <button onClick={() => router.push(`/sessions/${f.session_id}/forms/${f.id}`)} style={chip(false)}>{mode === 'review' ? 'Prüfen' : 'Ansehen'}</button>
          {mode === 'library' && (
            <button data-testid="adopt" disabled={!target || busy === f.id} onClick={() => adopt(f)} style={{ ...chip(true), opacity: target ? 1 : 0.45 }} title={target ? 'Kopiert die Form in die gewählte Einheit' : 'Wähle oben zuerst eine Einheit'}>{busy === f.id ? 'Übernimmt …' : 'In Einheit übernehmen'}</button>
          )}
          {mode === 'review' && f.library_status === 'archived' && <span style={{ fontSize: 11, color: MUTED, textAlign: 'center' }}>Archiviert</span>}
        </div>
      </div>
    )
  }

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={role} email={user?.email} onSignOut={handleSignOut} backHref="/sessions" />
      <div style={{ padding: 16, maxWidth: 1000, margin: '0 auto' }}>
        <h1 style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 30, color: NAVY, margin: '4px 0 2px', letterSpacing: '0.02em' }}>Bibliothek</h1>
        <div style={{ fontSize: 13, color: '#4A4F5C', marginBottom: 12 }}>Geprüfte Übungen und Spielformen, geordnet nach unserem Spielmodell.</div>

        {!isCoachOrAdmin(role) ? (
          <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '40px 20px', textAlign: 'center', color: MUTED, fontSize: 14 }}>Die Bibliothek ist für Trainerinnen und Trainer.</div>
        ) : (
          <>
            {admin && (
              <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                <button data-testid="tab-library" onClick={() => setTab('library')} style={chip(tab === 'library')}>Bibliothek ({inLibrary.length})</button>
                <button data-testid="tab-review" onClick={() => setTab('review')} style={chip(tab === 'review', '#92400e')}>Zur Prüfung ({proposed.length})</button>
              </div>
            )}
            {error && <div style={{ fontSize: 13, color: '#991b1b', marginBottom: 8 }}>{error}</div>}
            {done && (
              <div data-testid="adopted" style={{ background: '#dcfce7', border: '1px solid #86efac', borderRadius: 10, padding: '8px 12px', fontSize: 13, color: '#166534', marginBottom: 10 }}>
                „{done.title}“ ist jetzt in deiner Einheit. <button onClick={() => router.push(`/sessions/${done.sessionId}`)} style={{ background: 'none', border: 'none', color: '#166534', textDecoration: 'underline', cursor: 'pointer', fontWeight: 700, fontFamily: 'DM Sans, sans-serif' }}>Zur Einheit</button>
              </div>
            )}

            {tab === 'review' && admin ? (
              <div data-testid="review-list" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {proposed.length === 0 && <div style={{ fontSize: 13, color: MUTED, padding: 20, textAlign: 'center' }}>Nichts zu prüfen.</div>}
                {proposed.map(f => card(f, 'review'))}
                {archived.length > 0 && <div style={{ fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.06em', textTransform: 'uppercase', marginTop: 10 }}>Archiviert</div>}
                {archived.map(f => card(f, 'review'))}
              </div>
            ) : (
              <>
                <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 12, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    <button data-testid="filter-all" onClick={() => { setPhase(null); setConcept(null) }} style={chip(!phase && !concept)}>Alle</button>
                    {PHASES.map(p => <button key={p.id} data-testid={`filter-phase-${p.id}`} onClick={() => { setPhase(p.id); setConcept(null) }} style={chip(phase === p.id, phaseColor(p.id))}>{p.name}</button>)}
                  </div>
                  {phase && groupsOfPhase(phase).map(g => (
                    <div key={g.group} style={{ marginBottom: 6 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 3 }}>{g.group}</div>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {g.items.map(c => {
                          const n = countOf(c.id)
                          return <button key={c.id} data-testid={`filter-${c.id}`} onClick={() => setConcept(concept === c.id ? null : c.id)} style={{ ...chip(concept === c.id, phaseColor(c.phase)), opacity: n === 0 && concept !== c.id ? 0.5 : 1 }}>{c.name} <span style={{ opacity: 0.75 }}>{n}</span></button>
                        })}
                      </div>
                    </div>
                  ))}
                  <input data-testid="library-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Suchen …" style={{ marginTop: 8, width: '100%', boxSizing: 'border-box', fontSize: 14, padding: '8px 12px', borderRadius: 8, border: '1px solid #E4E6EE', fontFamily: 'DM Sans, sans-serif', outline: 'none' }} />
                </div>

                <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: MUTED }}>Übernehmen in:</span>
                  <select data-testid="target-session" value={target} onChange={e => setTarget(e.target.value)} style={{ fontSize: 13, padding: '7px 10px', borderRadius: 8, border: '1px solid #E4E6EE', fontFamily: 'DM Sans, sans-serif', background: '#fff', maxWidth: '100%' }}>
                    <option value="">Einheit wählen …</option>
                    {mySessions.map(s => <option key={s.id} value={s.id}>{s.title}{s.session_date ? ` (${s.session_date})` : ''}</option>)}
                  </select>
                </div>

                <div data-testid="library-list" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {shown.length === 0 && (
                    <div style={{ background: '#fff', border: '1px dashed #cfd4e2', borderRadius: 12, padding: '30px 16px', textAlign: 'center', color: MUTED, fontSize: 13 }}>
                      {inLibrary.length === 0 ? 'Die Bibliothek ist noch leer. Schlage eine fertige Übung auf ihrer Seite zur Aufnahme vor.' : 'Dazu gibt es noch keine Form in der Bibliothek.'}
                    </div>
                  )}
                  {shown.map(f => card(f, 'library'))}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
