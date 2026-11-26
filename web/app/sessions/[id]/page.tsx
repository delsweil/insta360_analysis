'use client'

export const dynamic = 'force-dynamic'

import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import SceneCanvas from '@/components/SceneCanvas'
import { defaultBackground } from '@/lib/scene'
import { parseTags, moveItem, reindex, totalMinutes, sceneOf, FORM_KIND_LABELS, type TrainingSession, type SessionForm, type FormSceneRow, type FormKind } from '@/lib/sessions'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

type SaveState = 'idle' | 'saving' | 'saved' | 'error'

export default function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = React.use(params)
  const router = useRouter()
  const [user, setUser] = useState<{ id: string; email?: string } | null>(null)
  const [role, setRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [session, setSession] = useState<TrainingSession | null>(null)
  const [notFound, setNotFound] = useState(false)
  const [loading, setLoading] = useState(true)
  const [save, setSave] = useState<SaveState>('idle')
  const [saveMsg, setSaveMsg] = useState('')
  const [tagText, setTagText] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [forms, setForms] = useState<SessionForm[]>([])
  const [firstScene, setFirstScene] = useState<Record<string, FormSceneRow>>({})
  const [confirmForm, setConfirmForm] = useState<string | null>(null)
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      setUser({ id: user.id, email: user.email })
      const r = await getCurrentUserRole()
      setRole(r)
      if (isCoachOrAdmin(r)) {
        const { data } = await supabase.from('training_sessions').select('*').eq('id', id).maybeSingle()
        if (data) {
          setSession(data as TrainingSession); setTagText((data as TrainingSession).tags.join(', '))
          const { data: fs } = await supabase.from('session_forms').select('*').eq('session_id', id)
          const list = ((fs ?? []) as SessionForm[]).sort((a, b) => a.position - b.position)
          setForms(list)
          if (list.length) {
            const { data: sc } = await supabase.from('form_scenes').select('*').in('form_id', list.map(f => f.id))
            const first: Record<string, FormSceneRow> = {}
            for (const r of ((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position)) if (!first[r.form_id]) first[r.form_id] = r
            setFirstScene(first)
          }
        } else setNotFound(true)
      } else setNotFound(true)
      setLoading(false)
    }
    load()
  }, [id, router])

  const mine = !!session && session.created_by === user?.id

  const patch = useCallback(async (p: Partial<TrainingSession>) => {
    if (!session) return
    setSession(s => (s ? { ...s, ...p } : s))
    setSave('saving')
    const { error } = await supabase.from('training_sessions').update(p).eq('id', session.id)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    setSave('saved')
    if (savedTimer.current) clearTimeout(savedTimer.current)
    savedTimer.current = setTimeout(() => setSave('idle'), 1800)
  }, [session])

  async function writeFormPositions(list: SessionForm[]) {
    for (const f of list) {
      const { error } = await supabase.from('session_forms').update({ position: f.position }).eq('id', f.id)
      if (error) { setSave('error'); setSaveMsg(error.message); return }
    }
  }

  async function addForm(kind: FormKind) {
    if (!session) return
    const { data, error } = await supabase.from('session_forms').insert({ session_id: session.id, position: forms.length, kind, title: kind === 'spielform' ? 'Neue Spielform' : 'Neue Übung' }).select().single()
    if (error || !data) { setSave('error'); setSaveMsg(error?.message ?? 'Konnte nicht angelegt werden.'); return }
    const f = data as SessionForm
    // a first, empty sketch so drawing can start right away
    await supabase.from('form_scenes').insert({ form_id: f.id, position: 0, background: defaultBackground('free'), shapes: [], caption: '' })
    router.push(`/sessions/${session.id}/forms/${f.id}`)
  }

  async function moveForm(i: number, to: number) {
    if (to < 0 || to >= forms.length) return
    const next = reindex(moveItem(forms, i, to))
    setForms(next)
    await writeFormPositions(next)
  }

  async function duplicateForm(f: SessionForm) {
    if (!session) return
    const i = forms.findIndex(x => x.id === f.id)
    const { data, error } = await supabase.from('session_forms').insert({ session_id: session.id, position: i + 1, kind: f.kind, title: `${f.title} (Kopie)`, duration_min: f.duration_min, players: f.players, sections: f.sections }).select().single()
    if (error || !data) { setSave('error'); setSaveMsg(error?.message ?? 'Konnte nicht kopiert werden.'); return }
    const copy = data as SessionForm
    const { data: sc } = await supabase.from('form_scenes').select('*').eq('form_id', f.id)
    const rows = ((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position)
    for (const r of rows) await supabase.from('form_scenes').insert({ form_id: copy.id, position: r.position, background: r.background, shapes: r.shapes, caption: r.caption })
    const next = reindex([...forms.slice(0, i + 1), copy, ...forms.slice(i + 1)])
    setForms(next)
    if (rows[0]) setFirstScene(m => ({ ...m, [copy.id]: { ...rows[0], id: 'copy-' + copy.id, form_id: copy.id } }))
    await writeFormPositions(next.filter((_, k) => k > i + 1))
  }

  async function deleteForm(f: SessionForm) {
    const { error } = await supabase.from('session_forms').delete().eq('id', f.id)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    const next = reindex(forms.filter(x => x.id !== f.id))
    setForms(next); setConfirmForm(null)
    await writeFormPositions(next.filter((_, k) => k >= forms.findIndex(x => x.id === f.id)))
  }

  async function remove() {
    if (!session) return
    const { error } = await supabase.from('training_sessions').delete().eq('id', session.id)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    router.push('/sessions')
  }

  const handleSignOut = async () => { await supabase.auth.signOut(); router.push('/login') }

  const field: React.CSSProperties = {
    width: '100%', fontSize: 14, padding: '8px 12px', borderRadius: 8, border: '1px solid #E4E6EE',
    fontFamily: 'DM Sans, sans-serif', outline: 'none', background: mine ? '#fff' : '#f7f7f5', color: '#1c2333', boxSizing: 'border-box',
  }
  const label: React.CSSProperties = { fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 4 }

  if (loading) return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C' }}>Lädt …</div>
  )

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={role} email={user?.email} onSignOut={handleSignOut} backHref="/sessions" />
      <div style={{ padding: 20, maxWidth: 900, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12, gap: 8 }}>
          <button onClick={() => router.push('/sessions')} style={{ fontSize: 12, fontWeight: 600, color: NAVY, background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontFamily: 'DM Sans, sans-serif' }}>
            ← Trainingseinheiten
          </button>
          {session && (
            <button data-testid="pdf-session" onClick={() => window.open(`/sessions/${session.id}/print`, '_blank')} style={{ fontSize: 12, fontWeight: 600, color: NAVY, background: '#fff', border: '1.5px solid #E4E6EE', borderRadius: 8, cursor: 'pointer', padding: '7px 12px', fontFamily: 'DM Sans, sans-serif' }}>
              PDF der Einheit
            </button>
          )}
        </div>

        {notFound || !session ? (
          <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '40px 20px', textAlign: 'center', color: MUTED, fontSize: 14 }}>
            Diese Einheit gibt es nicht, oder sie ist noch nicht für dich geteilt.
          </div>
        ) : (
          <>
            <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={label}>Titel</div>
                <input
                  data-testid="title"
                  style={{ ...field, fontSize: 20, fontFamily: 'Bebas Neue, sans-serif', letterSpacing: '0.02em', color: NAVY }}
                  value={session.title}
                  readOnly={!mine}
                  onChange={e => setSession({ ...session, title: e.target.value })}
                  onBlur={e => { if (mine) patch({ title: e.target.value.trim() || 'Neue Trainingseinheit' }) }}
                />
              </div>

              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ flex: '0 0 170px' }}>
                  <div style={label}>Datum</div>
                  <input
                    data-testid="date" type="date" style={field} readOnly={!mine} disabled={!mine}
                    value={session.session_date ?? ''}
                    onChange={e => patch({ session_date: e.target.value || null })}
                  />
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={label}>Schwerpunkt</div>
                  <input
                    data-testid="focus" style={field} readOnly={!mine} placeholder="z. B. Freilaufen und Passwege"
                    value={session.focus}
                    onChange={e => setSession({ ...session, focus: e.target.value })}
                    onBlur={e => { if (mine) patch({ focus: e.target.value.trim() }) }}
                  />
                </div>
              </div>

              <div>
                <div style={label}>Tags</div>
                <input
                  data-testid="tags" style={field} readOnly={!mine} placeholder="z. B. U13, Pressing, Spielform (mit Komma trennen)"
                  value={tagText}
                  onChange={e => setTagText(e.target.value)}
                  onBlur={() => { if (!mine) return; const t = parseTags(tagText); setTagText(t.join(', ')); patch({ tags: t }) }}
                />
                {session.tags.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 7 }}>
                    {session.tags.map(t => (
                      <span key={t} style={{ fontSize: 11, fontWeight: 600, color: NAVY, background: '#e8edf8', padding: '3px 10px', borderRadius: 99 }}>{t}</span>
                    ))}
                  </div>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', borderTop: '1px solid #eceef3', paddingTop: 14 }}>
                <div>
                  <div style={label}>Sichtbarkeit</div>
                  <div data-testid="visibility-text" style={{ fontSize: 13, color: '#4A4F5C' }}>
                    {session.visibility === 'shared'
                      ? 'Geteilt: alle Trainerinnen und Trainer des Vereins können diese Einheit sehen.'
                      : mine ? 'Privat: nur du siehst diese Einheit.' : 'Privat'}
                  </div>
                </div>
                {mine ? (
                  <button
                    data-testid="toggle-visibility"
                    onClick={() => patch({ visibility: session.visibility === 'shared' ? 'private' : 'shared' })}
                    style={{ fontSize: 12, fontWeight: 600, padding: '8px 16px', borderRadius: 8, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', border: session.visibility === 'shared' ? '1px solid #E4E6EE' : 'none', background: session.visibility === 'shared' ? '#fff' : NAVY, color: session.visibility === 'shared' ? '#4A4F5C' : '#fff' }}
                  >
                    {session.visibility === 'shared' ? 'Wieder privat machen' : 'Für Trainer teilen'}
                  </button>
                ) : (
                  <span style={{ fontSize: 12, color: MUTED }}>Nur der Autor kann bearbeiten.</span>
                )}
              </div>

              <div data-testid="save-state" style={{ fontSize: 12, minHeight: 16, color: save === 'error' ? '#991b1b' : MUTED }}>
                {save === 'saving' && 'Speichert …'}
                {save === 'saved' && 'Gespeichert'}
                {save === 'error' && `Nicht gespeichert: ${saveMsg}`}
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                <div style={label}>
                  Formen{forms.length ? ` · ${forms.length}` : ''}{totalMinutes(forms) ? ` · ${totalMinutes(forms)} Min.` : ''}
                </div>
                {mine && (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button data-testid="add-exercise" onClick={() => addForm('exercise')} style={{ fontSize: 12, fontWeight: 600, padding: '7px 14px', borderRadius: 8, border: 'none', background: NAVY, color: '#fff', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>+ Übung</button>
                    <button data-testid="add-spielform" onClick={() => addForm('spielform')} style={{ fontSize: 12, fontWeight: 600, padding: '7px 14px', borderRadius: 8, border: 'none', background: NAVY, color: '#fff', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>+ Spielform</button>
                  </div>
                )}
              </div>
              {forms.length === 0 ? (
                <div style={{ border: '1.5px dashed #cfd4e2', borderRadius: 12, padding: '28px 20px', textAlign: 'center', color: MUTED, fontSize: 13 }}>
                  {mine ? 'Noch keine Formen. Füge eine Übung oder Spielform hinzu.' : 'Diese Einheit hat noch keine Formen.'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {forms.map((f, i) => {
                    const fs = firstScene[f.id]
                    return (
                      <div key={f.id}>
                        <div data-testid="form-card" style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: confirmForm === f.id ? '12px 12px 0 0' : 12, padding: 10, display: 'flex', alignItems: 'center', gap: 12 }}>
                          <div onClick={() => router.push(`/sessions/${session.id}/forms/${f.id}`)} style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 0, cursor: 'pointer' }}>
                            <div style={{ width: 92, flex: '0 0 92px', borderRadius: 8, overflow: 'hidden', border: '1px solid #eceef3', background: '#fbfbf8' }}>
                              {fs ? <SceneCanvas scene={sceneOf(fs)} look="clean" testId="card-thumb" /> : <div style={{ height: 62 }} />}
                            </div>
                            <div style={{ minWidth: 0 }}>
                              <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 17, color: NAVY, letterSpacing: '0.02em', lineHeight: 1.1 }}>{i + 1}. {f.title}</div>
                              <div style={{ fontSize: 12, color: MUTED, marginTop: 3 }}>
                                {FORM_KIND_LABELS[f.kind]}{f.duration_min ? ` · ${f.duration_min} Min.` : ''}{f.players ? ` · ${f.players}` : ''}
                              </div>
                            </div>
                          </div>
                          {mine && (
                            <div style={{ display: 'flex', gap: 2, flexShrink: 0 }}>
                              <button aria-label="Nach oben" disabled={i === 0} onClick={() => moveForm(i, i - 1)} style={{ fontSize: 14, color: MUTED, background: 'none', border: 'none', cursor: 'pointer', padding: '6px', opacity: i === 0 ? 0.35 : 1 }}>↑</button>
                              <button aria-label="Nach unten" disabled={i === forms.length - 1} onClick={() => moveForm(i, i + 1)} style={{ fontSize: 14, color: MUTED, background: 'none', border: 'none', cursor: 'pointer', padding: '6px', opacity: i === forms.length - 1 ? 0.35 : 1 }}>↓</button>
                              <button aria-label="PDF dieser Form" title="PDF dieser Form" data-testid="form-pdf" onClick={() => window.open(`/sessions/${id}/print?form=${f.id}`, '_blank')} style={{ fontSize: 11, fontWeight: 700, color: MUTED, background: 'none', border: 'none', cursor: 'pointer', padding: '6px' }}>PDF</button>
                              <button aria-label="Duplizieren" data-testid="form-duplicate" onClick={() => duplicateForm(f)} style={{ fontSize: 13, color: MUTED, background: 'none', border: 'none', cursor: 'pointer', padding: '6px' }}>⧉</button>
                              <button aria-label="Löschen" data-testid="form-delete" onClick={() => setConfirmForm(confirmForm === f.id ? null : f.id)} style={{ fontSize: 14, color: MUTED, background: 'none', border: 'none', cursor: 'pointer', padding: '6px' }}>✕</button>
                            </div>
                          )}
                        </div>
                        {confirmForm === f.id && (
                          <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderTop: 'none', borderRadius: '0 0 12px 12px', padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                            <span style={{ fontSize: 13, color: '#991b1b' }}>Form mit allen Skizzen löschen?</span>
                            <div style={{ display: 'flex', gap: 8 }}>
                              <button onClick={() => setConfirmForm(null)} style={{ fontSize: 12, fontWeight: 600, padding: '5px 14px', borderRadius: 6, border: '1px solid #fca5a5', background: '#fff', color: '#991b1b', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>Abbrechen</button>
                              <button data-testid="form-delete-confirm" onClick={() => deleteForm(f)} style={{ fontSize: 12, fontWeight: 600, padding: '5px 14px', borderRadius: 6, border: 'none', background: '#ef4444', color: '#fff', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>Löschen</button>
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>

            {mine && (
              <div style={{ marginTop: 18 }}>
                {!confirmDelete ? (
                  <button onClick={() => setConfirmDelete(true)} style={{ fontSize: 12, color: '#991b1b', background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontFamily: 'DM Sans, sans-serif' }}>
                    Einheit löschen
                  </button>
                ) : (
                  <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                    <span style={{ fontSize: 13, color: '#991b1b' }}>Einheit mit allen Formen und Skizzen löschen?</span>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button onClick={() => setConfirmDelete(false)} style={{ fontSize: 12, fontWeight: 600, padding: '5px 14px', borderRadius: 6, border: '1px solid #fca5a5', background: '#fff', color: '#991b1b', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>Abbrechen</button>
                      <button data-testid="confirm-delete" onClick={remove} style={{ fontSize: 12, fontWeight: 600, padding: '5px 14px', borderRadius: 6, border: 'none', background: '#ef4444', color: '#fff', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>Löschen</button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
