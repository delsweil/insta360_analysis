'use client'

export const dynamic = 'force-dynamic'

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin, isAdmin } from '@/lib/supabase'
import Topbar from '@/components/Topbar'
import SceneEditor, { SizeInput } from '@/components/SceneEditor'
import SceneAnimator from '@/components/SceneAnimator'
import { DEFAULT_HOLD_S, DEFAULT_MOVE_S, MIN_S, MAX_S, holdOf, moveOf, type Step } from '@/lib/anim'
import SceneCanvas from '@/components/SceneCanvas'
import ConceptPicker from '@/components/ConceptPicker'
import { STATUS_LABELS, completeness, copyOfForm, copyOfScene, missing } from '@/lib/library'
import { copyScene, defaultBackground, type Scene } from '@/lib/scene'
import {
  FORM_KIND_LABELS, moveItem, newSection, parseMinutes, reindex, sceneOf, unusedStarters,
  type FormKind, type FormSceneRow, type LibraryStatus, type Section, type SessionForm, type TrainingSession,
} from '@/lib/sessions'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'
type SaveState = 'idle' | 'saving' | 'saved' | 'error'

export default function FormPage({ params }: { params: Promise<{ id: string; fid: string }> }) {
  const { id: sessionId, fid } = React.use(params)
  const router = useRouter()
  const [user, setUser] = useState<{ id: string; email?: string } | null>(null)
  const [role, setRole] = useState<'admin' | 'coach' | 'player' | null>(null)
  const [session, setSession] = useState<TrainingSession | null>(null)
  const [form, setForm] = useState<SessionForm | null>(null)
  const [scenes, setScenes] = useState<FormSceneRow[]>([])
  const [active, setActive] = useState(0)
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [save, setSave] = useState<SaveState>('idle')
  const [saveMsg, setSaveMsg] = useState('')
  const [duration, setDuration] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [confirmScene, setConfirmScene] = useState(false)
  const steps = useMemo<Step[]>(() => scenes.map(r => ({ scene: sceneOf(r), timing: r.timing })), [scenes])

  const pending = useRef<{ form: Partial<SessionForm>; scenes: Map<string, Partial<FormSceneRow>> }>({ form: {}, scenes: new Map() })
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const owner = !!session && session.created_by === user?.id
  const status: LibraryStatus = form?.library_status ?? 'private'
  const admin = isAdmin(role)
  // the author edits while the form is a draft; admins edit what is in the review/library process
  const mine = owner ? status === 'private' : admin && status !== 'private'
  const [returnNote, setReturnNote] = useState('')

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      setUser({ id: user.id, email: user.email })
      const r = await getCurrentUserRole()
      setRole(r)
      if (!isCoachOrAdmin(r)) { setNotFound(true); setLoading(false); return }
      const { data: f } = await supabase.from('session_forms').select('*').eq('id', fid).maybeSingle()
      const { data: s } = await supabase.from('training_sessions').select('*').eq('id', sessionId).maybeSingle()
      if (!f || !s || (f as SessionForm).session_id !== sessionId) { setNotFound(true); setLoading(false); return }
      setForm(f as SessionForm); setSession(s as TrainingSession)
      setDuration((f as SessionForm).duration_min ? String((f as SessionForm).duration_min) : '')
      const { data: sc } = await supabase.from('form_scenes').select('*').eq('form_id', fid)
      setScenes(((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position))
      setLoading(false)
    }
    load()
  }, [fid, sessionId, router])

  const flush = useCallback(async () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null }
    const p = pending.current
    const formPatch = p.form
    const scenePatches = [...p.scenes.entries()]
    if (!Object.keys(formPatch).length && !scenePatches.length) return
    pending.current = { form: {}, scenes: new Map() }
    setSave('saving')
    let failed: string | null = null
    if (Object.keys(formPatch).length) {
      const { error } = await supabase.from('session_forms').update(formPatch).eq('id', fid)
      if (error) failed = error.message
    }
    for (const [sid, patch] of scenePatches) {
      const { error } = await supabase.from('form_scenes').update(patch).eq('id', sid)
      if (error) failed = error.message
    }
    if (failed) { setSave('error'); setSaveMsg(failed); return }
    setSave('saved')
    if (savedTimer.current) clearTimeout(savedTimer.current)
    savedTimer.current = setTimeout(() => setSave('idle'), 1800)
  }, [fid])

  const schedule = useCallback(() => {
    setSave('saving')
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => { flush() }, 700)
  }, [flush])

  // save what is still waiting when the page is left
  useEffect(() => {
    const onHide = () => { if (document.visibilityState === 'hidden') flush() }
    document.addEventListener('visibilitychange', onHide)
    return () => { document.removeEventListener('visibilitychange', onHide); flush() }
  }, [flush])

  function patchForm(p: Partial<SessionForm>) {
    setForm(f => (f ? { ...f, ...p } : f))
    pending.current.form = { ...pending.current.form, ...p }
    schedule()
  }

  function patchScene(sceneId: string, p: Partial<FormSceneRow>) {
    setScenes(list => list.map(s => (s.id === sceneId ? { ...s, ...p } : s)))
    pending.current.scenes.set(sceneId, { ...(pending.current.scenes.get(sceneId) ?? {}), ...p })
    schedule()
  }

  const onSceneChange = (sceneId: string) => (sc: Scene) => patchScene(sceneId, { background: sc.background, shapes: sc.shapes })

  // ── scenes ──
  async function insertScene(sc: Scene, position: number, timing?: FormSceneRow['timing']): Promise<FormSceneRow | null> {
    // `timing` is only sent when there is some, so this keeps working before the timing column exists
    const { data, error } = await supabase.from('form_scenes').insert({ form_id: fid, position, background: sc.background, shapes: sc.shapes, caption: sc.caption, ...(timing ? { timing } : {}) }).select().single()
    if (error || !data) { setSave('error'); setSaveMsg(error?.message ?? 'Konnte nicht angelegt werden.'); return null }
    return data as FormSceneRow
  }
  async function writePositions(list: FormSceneRow[]) {
    for (const s of list) {
      const { error } = await supabase.from('form_scenes').update({ position: s.position }).eq('id', s.id)
      if (error) { setSave('error'); setSaveMsg(error.message); return }
    }
  }

  async function addScene() {
    await flush()
    const bg = scenes[active]?.background ?? defaultBackground('free')
    const row = await insertScene({ background: { ...bg }, shapes: [], caption: '' }, scenes.length)
    if (row) { setScenes(l => [...l, row]); setActive(scenes.length) }
  }
  async function copyAsNext() {
    const cur = scenes[active]
    if (!cur) return
    await flush()
    const copy = copyScene(sceneOf(cur))
    const row = await insertScene(copy, active + 1, cur.timing)
    if (!row) return
    const next = reindex([...scenes.slice(0, active + 1), row, ...scenes.slice(active + 1)])
    setScenes(next); setActive(active + 1)
    await writePositions(next.filter((s, i) => i > active + 1))
  }
  async function moveScene(dir: -1 | 1) {
    const to = active + dir
    if (to < 0 || to >= scenes.length) return
    await flush()
    const next = reindex(moveItem(scenes, active, to))
    setScenes(next); setActive(to)
    await writePositions(next)
  }
  async function deleteScene() {
    const cur = scenes[active]
    if (!cur) return
    await flush()
    const { error } = await supabase.from('form_scenes').delete().eq('id', cur.id)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    const next = reindex(scenes.filter(s => s.id !== cur.id))
    setScenes(next); setActive(Math.max(0, Math.min(active, next.length - 1))); setConfirmScene(false)
    await writePositions(next.filter((s, i) => i >= active))
  }

  // ── sections ──
  const sections = form?.sections ?? []
  const setSections = (next: Section[]) => patchForm({ sections: next })
  const editSection = (sid: string, p: Partial<Section>) => setSections(sections.map(s => (s.id === sid ? { ...s, ...p } : s)))

  async function deleteForm() {
    await flush()
    const { error } = await supabase.from('session_forms').delete().eq('id', fid)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    router.push(`/sessions/${sessionId}`)
  }

  async function setLibraryStatus(next: LibraryStatus, note?: string) {
    await flush()
    const patch: Partial<SessionForm> = { library_status: next }
    if (note !== undefined) patch.library_note = note
    const { error } = await supabase.from('session_forms').update(patch).eq('id', fid)
    if (error) { setSave('error'); setSaveMsg(error.message); return }
    setForm(f => (f ? { ...f, ...patch } : f)); setReturnNote('')
  }

  /** A library form is frozen: the author works on a copy in the same session. */
  async function newVersion() {
    if (!form) return
    await flush()
    const { data: sib } = await supabase.from('session_forms').select('position').eq('session_id', sessionId)
    const pos = ((sib ?? []) as { position: number }[]).reduce((m, r) => Math.max(m, r.position), -1) + 1
    const { data: nf, error } = await supabase.from('session_forms').insert({ ...copyOfForm(form, sessionId, pos), title: `${form.title} (neue Version)` }).select().single()
    if (error || !nf) { setSave('error'); setSaveMsg(error?.message ?? 'Konnte nicht angelegt werden.'); return }
    for (const sc of scenes) await supabase.from('form_scenes').insert(copyOfScene(sc, (nf as SessionForm).id))
    router.push(`/sessions/${sessionId}/forms/${(nf as SessionForm).id}`)
  }

  const handleSignOut = async () => { await supabase.auth.signOut(); router.push('/login') }

  const field: React.CSSProperties = { width: '100%', fontSize: 14, padding: '8px 12px', borderRadius: 8, border: '1px solid #E4E6EE', fontFamily: 'DM Sans, sans-serif', outline: 'none', background: mine ? '#fff' : '#f7f7f5', color: '#1c2333', boxSizing: 'border-box' }
  const label: React.CSSProperties = { fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 4 }
  const btn = (active = false, extra: React.CSSProperties = {}): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '7px 12px', minHeight: 36, borderRadius: 9, cursor: 'pointer', whiteSpace: 'nowrap',
    border: `1.5px solid ${active ? NAVY : '#E4E6EE'}`, background: active ? '#e8edf8' : '#fff', color: active ? NAVY : '#4A4F5C', fontFamily: 'DM Sans, sans-serif', ...extra,
  })

  if (loading) return <div style={{ minHeight: '100vh', background: '#F8F8F6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'DM Sans, sans-serif', color: '#4A4F5C' }}>Lädt …</div>

  const scene = scenes[active]

  return (
    <div style={{ minHeight: '100vh', background: '#F8F8F6', fontFamily: 'DM Sans, sans-serif' }}>
      <Topbar role={role} email={user?.email} onSignOut={handleSignOut} backHref="/sessions" />
      <div style={{ padding: 16, maxWidth: 1000, margin: '0 auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, gap: 8 }}>
          <button onClick={() => router.push(`/sessions/${sessionId}`)} style={{ fontSize: 12, fontWeight: 600, color: NAVY, background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontFamily: 'DM Sans, sans-serif' }}>
            ← {session?.title ?? 'Einheit'}
          </button>
          {form && (
            <button data-testid="pdf-form" onClick={() => window.open(`/sessions/${sessionId}/print?form=${form.id}`, '_blank')} style={{ fontSize: 12, fontWeight: 600, color: NAVY, background: '#fff', border: '1.5px solid #E4E6EE', borderRadius: 8, cursor: 'pointer', padding: '7px 12px', fontFamily: 'DM Sans, sans-serif' }}>
              PDF dieser Form
            </button>
          )}
        </div>

        {notFound || !form ? (
          <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: '40px 20px', textAlign: 'center', color: MUTED, fontSize: 14 }}>Diese Form gibt es nicht, oder du darfst sie nicht sehen.</div>
        ) : (
          <>
            <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div>
                <div style={label}>Titel</div>
                <input data-testid="form-title" style={{ ...field, fontSize: 20, fontFamily: 'Bebas Neue, sans-serif', letterSpacing: '0.02em', color: NAVY }} readOnly={!mine} value={form.title} onChange={e => patchForm({ title: e.target.value })} onBlur={e => { if (mine && !e.target.value.trim()) patchForm({ title: 'Neue Form' }) }} />
              </div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div>
                  <div style={label}>Art</div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {(['exercise', 'spielform'] as FormKind[]).map(k => <button key={k} data-testid={`kind-${k}`} disabled={!mine} onClick={() => patchForm({ kind: k })} style={btn(form.kind === k)}>{FORM_KIND_LABELS[k]}</button>)}
                  </div>
                </div>
                <div style={{ width: 110 }}>
                  <div style={label}>Dauer (Min.)</div>
                  <input data-testid="form-duration" style={field} readOnly={!mine} inputMode="numeric" placeholder="15" value={duration} onChange={e => { setDuration(e.target.value); patchForm({ duration_min: parseMinutes(e.target.value) }) }} />
                </div>
                <div style={{ flex: 1, minWidth: 200 }}>
                  <div style={label}>Spieler und Gruppen</div>
                  <input data-testid="form-players" style={field} readOnly={!mine} placeholder="z. B. 4 v 2, dazu 4 Wartende" value={form.players} onChange={e => patchForm({ players: e.target.value })} />
                </div>
              </div>
              <div data-testid="form-save" style={{ fontSize: 12, minHeight: 16, color: save === 'error' ? '#991b1b' : MUTED }}>
                {save === 'saving' && 'Speichert …'}{save === 'saved' && 'Gespeichert'}{save === 'error' && `Nicht gespeichert: ${saveMsg}`}
              </div>
            </div>

            {/* game model and library */}
            <div data-testid="library-card" style={{ marginTop: 14, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                <div style={label}>Spielmodell</div>
                <span data-testid="library-status" style={{ fontSize: 11, fontWeight: 700, padding: '3px 10px', borderRadius: 99, background: status === 'library' ? '#dcfce7' : status === 'proposed' ? '#fef3c7' : status === 'archived' ? '#e5e7eb' : '#eef0f5', color: status === 'library' ? '#166534' : status === 'proposed' ? '#92400e' : '#4A4F5C' }}>{STATUS_LABELS[status]}</span>
              </div>
              <ConceptPicker value={form.concepts ?? []} onChange={ids => patchForm({ concepts: ids })} readOnly={!mine} />
              {owner && status === 'private' && form.library_note && (
                <div data-testid="library-note" style={{ fontSize: 13, background: '#fef3c7', border: '1px solid #fcd34d', borderRadius: 10, padding: '8px 12px', color: '#78350f' }}><b>Rückmeldung zur Bibliothek:</b> {form.library_note}</div>
              )}
              {owner && status === 'private' && (() => {
                const checks = completeness(form, scenes)
                const gaps = missing(checks)
                return (
                  <div data-testid="propose-box" style={{ borderTop: '1px solid #eceef3', paddingTop: 10 }}>
                    <div style={{ fontSize: 12, color: '#4A4F5C', marginBottom: 6 }}>Für die Bibliothek vorschlagen. Nur vollständige Formen kommen in die Prüfung:</div>
                    <ul style={{ margin: '0 0 8px', padding: 0, listStyle: 'none', fontSize: 12 }}>
                      {checks.map(c => <li key={c.key} data-testid={`check-${c.key}`} data-ok={c.ok} style={{ color: c.ok ? '#166534' : '#991b1b' }}>{c.ok ? '✓' : '○'} {c.label}</li>)}
                    </ul>
                    <button data-testid="propose" disabled={gaps.length > 0} onClick={() => setLibraryStatus('proposed')} style={btn(true, { opacity: gaps.length ? 0.45 : 1, cursor: gaps.length ? 'not-allowed' : 'pointer' })}>Zur Aufnahme vorschlagen</button>
                    <div style={{ fontSize: 11, color: MUTED, marginTop: 6 }}>Nach dem Einreichen ist die Form gesperrt, bis sie geprüft ist. Du kannst den Vorschlag zurückziehen.</div>
                  </div>
                )
              })()}
              {owner && status === 'proposed' && (
                <div style={{ borderTop: '1px solid #eceef3', paddingTop: 10, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 13, color: '#4A4F5C' }}>Wartet auf die Prüfung durch einen Admin. Die Form ist gesperrt.</span>
                  <button data-testid="withdraw" onClick={() => setLibraryStatus('private')} style={btn()}>Vorschlag zurückziehen</button>
                </div>
              )}
              {owner && status === 'library' && (
                <div style={{ borderTop: '1px solid #eceef3', paddingTop: 10, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 13, color: '#4A4F5C' }}>In der Bibliothek, fixiert. Änderungen machst du an einer neuen Version.</span>
                  <button data-testid="new-version" onClick={newVersion} style={btn(true)}>Neue Version erstellen</button>
                </div>
              )}
              {admin && status !== 'private' && (
                <div data-testid="admin-review" style={{ borderTop: '1px solid #eceef3', paddingTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: NAVY }}>Admin</div>
                  {status === 'proposed' && (
                    <>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <button data-testid="approve" onClick={() => setLibraryStatus('library', '')} style={btn(true)}>In die Bibliothek aufnehmen</button>
                      </div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                        <input data-testid="return-note" style={{ ...field, flex: 1, minWidth: 200, background: '#fff' }} placeholder="Was fehlt oder soll geändert werden?" value={returnNote} onChange={e => setReturnNote(e.target.value)} />
                        <button data-testid="return" disabled={!returnNote.trim()} onClick={() => setLibraryStatus('private', returnNote.trim())} style={btn(false, { opacity: returnNote.trim() ? 1 : 0.45 })}>Zurückgeben</button>
                      </div>
                    </>
                  )}
                  {status === 'library' && <div><button data-testid="archive" onClick={() => setLibraryStatus('archived')} style={btn(false, { color: '#991b1b', borderColor: '#fca5a5' })}>Archivieren</button></div>}
                  {status === 'archived' && <div><button data-testid="restore" onClick={() => setLibraryStatus('library')} style={btn(true)}>Wieder aufnehmen</button></div>}
                </div>
              )}
            </div>

            {/* sketches */}
            <div style={{ marginTop: 14, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, marginBottom: 10 }}>
                <div style={label}>Skizzen</div>
                {mine && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <button data-testid="add-scene" onClick={addScene} style={btn(true)}>+ Neue Skizze</button>
                    {scene && <button data-testid="copy-scene" onClick={copyAsNext} style={btn()} title="Kopiert diese Skizze als nächste, zum Weiterzeichnen">Als nächste kopieren</button>}
                    {scene && scenes.length > 1 && <button data-testid="scene-left" onClick={() => moveScene(-1)} disabled={active === 0} style={btn(false, { opacity: active === 0 ? 0.4 : 1 })}>←</button>}
                    {scene && scenes.length > 1 && <button data-testid="scene-right" onClick={() => moveScene(1)} disabled={active === scenes.length - 1} style={btn(false, { opacity: active === scenes.length - 1 ? 0.4 : 1 })}>→</button>}
                    {scene && !confirmScene && <button data-testid="delete-scene" onClick={() => setConfirmScene(true)} style={btn(false, { color: '#991b1b', borderColor: '#fca5a5' })}>Skizze löschen</button>}
                    {scene && confirmScene && <button data-testid="confirm-delete-scene" onClick={deleteScene} style={btn(false, { background: '#ef4444', color: '#fff', borderColor: '#ef4444' })}>Wirklich löschen</button>}
                  </div>
                )}
              </div>

              {scenes.length === 0 ? (
                <div style={{ border: '1.5px dashed #cfd4e2', borderRadius: 12, padding: '28px 16px', textAlign: 'center', color: MUTED, fontSize: 13 }}>
                  {mine ? 'Noch keine Skizze. Lege die erste an.' : 'Keine Skizze.'}
                </div>
              ) : (
                <>
                  {scenes.length > 1 && (
                    <div data-testid="scene-strip" style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 8, marginBottom: 8 }}>
                      {scenes.map((s, i) => (
                        <button key={s.id} data-testid="scene-thumb" onClick={() => { setActive(i); setConfirmScene(false) }} style={{ flex: '0 0 auto', width: 116, padding: 4, borderRadius: 10, cursor: 'pointer', border: `2px solid ${i === active ? NAVY : '#E4E6EE'}`, background: '#fff' }}>
                          <SceneCanvas scene={sceneOf(s)} look="clean" testId="thumb-svg" />
                          <div style={{ fontSize: 11, fontWeight: 600, color: i === active ? NAVY : MUTED, marginTop: 2 }}>{i + 1}</div>
                        </button>
                      ))}
                    </div>
                  )}
                  {scene && (
                    <>
                      <SceneEditor key={scene.id} scene={sceneOf(scene)} onChange={onSceneChange(scene.id)} readOnly={!mine} />
                      <div style={{ marginTop: 8 }}>
                        <div style={label}>Beschriftung der Skizze</div>
                        <input data-testid="scene-caption" style={field} readOnly={!mine} placeholder="z. B. Aufbau, 4 v 2" value={scene.caption} onChange={e => patchScene(scene.id, { caption: e.target.value })} />
                      </div>
                      {mine && scenes.length > 1 && (
                        <div data-testid="timing-row" style={{ marginTop: 8, display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap', fontSize: 12, color: '#4A4F5C' }}>
                          <span style={label}>Animation</span>
                          {active > 0 && (
                            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                              Bewegung hierher
                              <SizeInput testId="timing-move" label="Dauer der Bewegung in Sekunden" min={MIN_S} max={MAX_S} style={{ ...field, width: 64 }} value={moveOf(scene.timing)} onValue={v => patchScene(scene.id, { timing: { ...(scene.timing ?? {}), move_s: v } })} />
                              s
                            </span>
                          )}
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                            Pause danach
                            <SizeInput testId="timing-hold" label="Pause in Sekunden" min={0} max={MAX_S} style={{ ...field, width: 64 }} value={holdOf(scene.timing)} onValue={v => patchScene(scene.id, { timing: { ...(scene.timing ?? {}), hold_s: v } })} />
                            s
                          </span>
                          <button data-testid="timing-flow" type="button" onClick={() => scenes.forEach((r, i) => patchScene(r.id, { timing: { move_s: 1.2, hold_s: i === scenes.length - 1 ? 1.5 : 0 } }))} style={{ fontSize: 12, fontWeight: 600, padding: '5px 10px', borderRadius: 8, border: '1.5px solid #E4E6EE', background: '#fff', color: '#0f2972', cursor: 'pointer' }} title="Alle Skizzen ohne Pausen ineinander übergehen lassen">Alles fließend</button>
                          <span style={{ color: MUTED }}>Standard: {String(DEFAULT_MOVE_S).replace('.', ',')} s Bewegung, {String(DEFAULT_HOLD_S).replace('.', ',')} s Pause</span>
                        </div>
                      )}
                    </>
                  )}
                </>
              )}
            </div>

            {/* animation */}
            <div data-testid="animation-card" style={{ marginTop: 14, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 14 }}>
              <div style={{ ...label, marginBottom: 4 }}>Animation</div>
              {scenes.length < 2 ? (
                <div style={{ fontSize: 13, color: MUTED }}>
                  {mine ? 'Zeichne mehrere Skizzen nacheinander („Als nächste kopieren“ und weiterzeichnen). Daraus entsteht die Animation: Alles, was in beiden Skizzen vorkommt, bewegt sich von der einen zur nächsten Position. Laufwege und Pässe, die du eingezeichnet hast, bestimmen den Weg.' : 'Diese Form hat nur eine Skizze.'}
                </div>
              ) : (
                <>
                  <div style={{ fontSize: 12, color: MUTED, marginBottom: 8 }}>Spielt die Skizzen der Reihe nach ab. Gleiche Figuren wandern von Skizze zu Skizze, eingezeichnete Läufe und Pässe geben den Weg vor.</div>
                  <SceneAnimator steps={steps} />
                </>
              )}
            </div>

            {/* sections */}
            <div style={{ marginTop: 14, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 14 }}>
              <div style={{ ...label, marginBottom: 10 }}>Beschreibung</div>
              {sections.length === 0 && <div style={{ fontSize: 13, color: MUTED, marginBottom: 10 }}>{mine ? 'Noch kein Abschnitt. Füge hinzu, was du brauchst, in beliebiger Reihenfolge.' : 'Keine Beschreibung.'}</div>}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {sections.map((s, i) => (
                  <div key={s.id} data-testid="section" style={{ border: '1px solid #eceef3', borderRadius: 10, padding: 10 }}>
                    <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
                      <input data-testid="section-title" style={{ ...field, fontWeight: 700, color: NAVY, flex: 1 }} readOnly={!mine} placeholder="Titel" value={s.title} onChange={e => editSection(s.id, { title: e.target.value })} />
                      {mine && (
                        <>
                          <button aria-label="Nach oben" disabled={i === 0} onClick={() => setSections(moveItem(sections, i, i - 1))} style={btn(false, { opacity: i === 0 ? 0.4 : 1 })}>↑</button>
                          <button aria-label="Nach unten" disabled={i === sections.length - 1} onClick={() => setSections(moveItem(sections, i, i + 1))} style={btn(false, { opacity: i === sections.length - 1 ? 0.4 : 1 })}>↓</button>
                          <button aria-label="Abschnitt löschen" data-testid="section-delete" onClick={() => setSections(sections.filter(x => x.id !== s.id))} style={btn(false, { color: '#991b1b' })}>✕</button>
                        </>
                      )}
                    </div>
                    <textarea data-testid="section-text" style={{ ...field, resize: 'vertical', lineHeight: 1.45 }} readOnly={!mine} rows={Math.max(3, Math.min(14, s.text.split('\n').length + 1))} value={s.text} onChange={e => editSection(s.id, { text: e.target.value })} />
                  </div>
                ))}
              </div>
              {mine && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12, alignItems: 'center' }}>
                  <span style={{ fontSize: 12, color: MUTED }}>Hinzufügen:</span>
                  {unusedStarters(sections).map(t => <button key={t} data-testid="starter" onClick={() => setSections([...sections, newSection(t)])} style={btn()}>{t}</button>)}
                  <button data-testid="add-section" onClick={() => setSections([...sections, newSection('')])} style={btn(true)}>+ Eigener Abschnitt</button>
                </div>
              )}
            </div>

            {mine && status === 'private' && (
              <div style={{ marginTop: 18 }}>
                {!confirmDelete ? (
                  <button onClick={() => setConfirmDelete(true)} style={{ fontSize: 12, color: '#991b1b', background: 'none', border: 'none', cursor: 'pointer', padding: 0, fontFamily: 'DM Sans, sans-serif' }}>Form löschen</button>
                ) : (
                  <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13, color: '#991b1b' }}>Form mit allen Skizzen löschen?</span>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button onClick={() => setConfirmDelete(false)} style={btn()}>Abbrechen</button>
                      <button data-testid="confirm-delete-form" onClick={deleteForm} style={btn(false, { background: '#ef4444', color: '#fff', borderColor: '#ef4444' })}>Löschen</button>
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
