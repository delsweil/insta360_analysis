'use client'

// Print view of a training session, or of a single form (?form=<id>). "Drucken / Als PDF sichern" opens the browser's print dialog,
// where "Als PDF sichern" makes the PDF. The sketches are vector graphics, so they stay sharp at any size.

export const dynamic = 'force-dynamic'

import React, { Suspense, useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { supabase, getCurrentUserRole, isCoachOrAdmin } from '@/lib/supabase'
import SceneCanvas from '@/components/SceneCanvas'
import { FORM_KIND_LABELS, sceneOf, totalMinutes, type FormSceneRow, type SessionForm, type TrainingSession } from '@/lib/sessions'

const NAVY = '#0f2972'
const MUTED = '#6b7080'
const BEBAS = "var(--font-bebas), 'Bebas Neue', Impact, sans-serif"
const SANS = "var(--font-dm-sans), 'DM Sans', system-ui, sans-serif"

function dateDe(d: string | null): string {
  if (!d) return ''
  const t = new Date(`${d}T12:00:00`)
  return Number.isNaN(t.getTime()) ? d : t.toLocaleDateString('de-DE', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })
}

function FormBlock({ f, scenes, index, total, sessionTitle }: { f: SessionForm; scenes: FormSceneRow[]; index?: number; total?: number; sessionTitle?: string }) {
  const meta = [f.duration_min ? `${f.duration_min} Min.` : '', f.players?.trim()].filter(Boolean).join('  ·  ')
  const sections = (f.sections ?? []).filter(s => s.title.trim() || s.text.trim())
  const single = scenes.length === 1
  return (
    <section className="form-block">
      <header className="form-head">
        <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: MUTED }}>
          {index !== undefined ? `${index} von ${total} · ` : ''}{FORM_KIND_LABELS[f.kind]}{sessionTitle ? ` · ${sessionTitle}` : ''}
        </div>
        <h2 style={{ fontFamily: BEBAS, fontSize: 30, letterSpacing: '0.02em', color: NAVY, margin: '2px 0 2px', fontWeight: 400 }}>{f.title || 'Ohne Titel'}</h2>
        {meta && <div style={{ fontSize: 13, color: MUTED, fontWeight: 500 }}>{meta}</div>}
      </header>

      {scenes.length > 0 && (
        <div className="scenes" style={{ gridTemplateColumns: single ? '1fr' : '1fr 1fr' }}>
          {scenes.map((r, i) => (
            <figure key={r.id} className="scene">
              <SceneCanvas scene={sceneOf(r)} plain style={{ maxHeight: single ? '150mm' : '88mm', border: '1px solid #d9dce6', borderRadius: 6 }} testId={`print-scene-${r.id}`} />
              {(r.caption?.trim() || !single) && (
                <figcaption style={{ fontSize: 12, color: '#33384a', marginTop: 4 }}>
                  {!single && <b style={{ color: NAVY }}>{i + 1}. </b>}{r.caption}
                </figcaption>
              )}
            </figure>
          ))}
        </div>
      )}

      {sections.length > 0 && (
        <div className="sections">
          {sections.map(s => (
            <div key={s.id} className="section">
              {s.title.trim() && <h3 style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', color: NAVY, margin: '0 0 2px' }}>{s.title}</h3>}
              <div style={{ fontSize: 13, lineHeight: 1.5, color: '#1c2033', whiteSpace: 'pre-wrap' }}>{s.text}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

function PrintInner({ id }: { id: string }) {
  const router = useRouter()
  const only = useSearchParams().get('form')
  const [state, setState] = useState<'loading' | 'missing' | 'ready'>('loading')
  const [session, setSession] = useState<TrainingSession | null>(null)
  const [forms, setForms] = useState<SessionForm[]>([])
  const [scenes, setScenes] = useState<Record<string, FormSceneRow[]>>({})

  useEffect(() => {
    async function load() {
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) { router.push('/login'); return }
      const role = await getCurrentUserRole()
      if (!isCoachOrAdmin(role)) { setState('missing'); return }
      const { data: s } = await supabase.from('training_sessions').select('*').eq('id', id).maybeSingle()
      if (!s) { setState('missing'); return }
      const { data: fs } = await supabase.from('session_forms').select('*').eq('session_id', id)
      let list = ((fs ?? []) as SessionForm[]).sort((a, b) => a.position - b.position)
      if (only) list = list.filter(f => f.id === only)
      const by: Record<string, FormSceneRow[]> = {}
      if (list.length) {
        const { data: sc } = await supabase.from('form_scenes').select('*').in('form_id', list.map(f => f.id))
        for (const r of ((sc ?? []) as FormSceneRow[]).sort((a, b) => a.position - b.position)) (by[r.form_id] ??= []).push(r)
      }
      if (only && !list.length) { setState('missing'); return }
      setSession(s as TrainingSession); setForms(list); setScenes(by); setState('ready')
    }
    load()
  }, [id, only, router])

  if (state === 'loading') return <div style={{ padding: 40, fontFamily: SANS, color: MUTED }}>Lädt …</div>
  if (state === 'missing' || !session) return <div style={{ padding: 40, fontFamily: SANS, color: MUTED }}>Diese Einheit gibt es nicht, oder sie ist nicht für dich sichtbar.</div>

  const wholeSession = !only
  const fileTitle = wholeSession ? session.title : `${session.title} – ${forms[0]?.title ?? 'Form'}`
  const mins = totalMinutes(forms)

  return (
    <div style={{ fontFamily: SANS, background: '#eceef3', minHeight: '100vh' }}>
      <style>{`
        @page { size: A4; margin: 14mm 13mm 16mm; }
        .toolbar { position: sticky; top: 0; z-index: 10; background: #fff; border-bottom: 1px solid #E4E6EE; padding: 10px 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .paper { width: 210mm; max-width: 100%; margin: 14px auto; background: #fff; padding: 14mm 13mm 16mm; box-sizing: border-box; box-shadow: 0 2px 14px rgba(15,41,114,0.12); }
        .scenes { display: grid; gap: 8mm 6mm; margin: 5mm 0 4mm; }
        .scene { margin: 0; break-inside: avoid; }
        .scene svg { display: block; margin: 0 auto; }
        .sections { display: flex; flex-direction: column; gap: 3.5mm; margin-top: 4mm; }
        .section { break-inside: avoid; }
        .form-head { break-after: avoid; }
        .form-block + .form-block { break-before: page; margin-top: 10mm; }
        .overview { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6mm; }
        .overview td, .overview th { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e4e6ee; }
        .overview th { font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; color: ${MUTED}; }
        .cover + .form-block { break-before: page; margin-top: 10mm; }
        @media print {
          html, body { background: #fff !important; }
          .toolbar { display: none !important; }
          .paper { margin: 0; padding: 0; box-shadow: none; width: auto; }
          * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
          .form-block + .form-block, .cover + .form-block { margin-top: 0; }
        }
      `}</style>

      <div className="toolbar">
        <button data-testid="print-btn" onClick={() => window.print()} style={{ background: NAVY, color: '#fff', border: 'none', borderRadius: 9, padding: '9px 16px', fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: SANS }}>Drucken / Als PDF sichern</button>
        <button onClick={() => router.back()} style={{ background: '#fff', color: NAVY, border: '1.5px solid #E4E6EE', borderRadius: 9, padding: '8px 14px', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: SANS }}>← Zurück</button>
        <span style={{ fontSize: 12, color: MUTED, flex: 1, minWidth: 220 }}>Im Druckdialog „Als PDF sichern“ wählen. Tipp: „Kopf- und Fußzeilen“ ausschalten, dann sieht es am saubersten aus.</span>
      </div>

      <div className="paper" data-testid="print-paper">
        <title>{fileTitle}</title>
        {wholeSession && (
          <div className="cover">
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: MUTED }}>Trainingseinheit{session.session_date ? ` · ${dateDe(session.session_date)}` : ''}</div>
            <h1 style={{ fontFamily: BEBAS, fontSize: 44, letterSpacing: '0.02em', color: NAVY, margin: '2px 0 4px', fontWeight: 400, lineHeight: 1.05 }}>{session.title}</h1>
            {session.focus?.trim() && <div style={{ fontSize: 14, color: '#1c2033', whiteSpace: 'pre-wrap', marginBottom: 6 }}>{session.focus}</div>}
            {session.tags.length > 0 && <div style={{ fontSize: 12, color: MUTED }}>{session.tags.map(t => `#${t}`).join('   ')}</div>}
            <div style={{ fontSize: 13, color: MUTED, marginTop: 6, fontWeight: 500 }}>{forms.length} {forms.length === 1 ? 'Form' : 'Formen'}{mins > 0 ? `  ·  ${mins} Min. gesamt` : ''}</div>
            {forms.length > 0 && (
              <table className="overview">
                <thead><tr><th>Nr.</th><th>Form</th><th>Art</th><th>Dauer</th><th>Spieler</th></tr></thead>
                <tbody>
                  {forms.map((f, i) => (
                    <tr key={f.id}><td>{i + 1}</td><td style={{ fontWeight: 600, color: NAVY }}>{f.title || 'Ohne Titel'}</td><td>{FORM_KIND_LABELS[f.kind]}</td><td>{f.duration_min ? `${f.duration_min} Min.` : ''}</td><td>{f.players}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )}
        {forms.map((f, i) => (
          <FormBlock key={f.id} f={f} scenes={scenes[f.id] ?? []} index={wholeSession ? i + 1 : undefined} total={wholeSession ? forms.length : undefined} sessionTitle={wholeSession ? undefined : session.title} />
        ))}
      </div>
    </div>
  )
}

export default function PrintPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = React.use(params)
  return <Suspense fallback={null}><PrintInner id={id} /></Suspense>
}
