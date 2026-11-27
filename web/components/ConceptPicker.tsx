'use client'

// Chooses concepts and principles of the game model (lib/gameModel.ts). The first one is the main concept.
import { useState } from 'react'
import { PHASES, conceptById, groupsOfPhase, makeMain, normalizeConceptIds, toggleConcept, MAX_CONCEPTS, type PhaseId } from '@/lib/gameModel'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

export const phaseColor = (p: PhaseId) => ({ ballbesitz: '#2c5fb3', ohneball: '#5b6277', 'umschalten-verteidigung': '#8a5a14', 'umschalten-angriff': '#1f7a4d' }[p])

export function ConceptChips({ ids, testId }: { ids: readonly string[]; testId?: string }) {
  const list = normalizeConceptIds(ids)
  if (!list.length) return null
  return (
    <span data-testid={testId} style={{ display: 'inline-flex', gap: 5, flexWrap: 'wrap' }}>
      {list.map((id, i) => {
        const c = conceptById(id)!
        return <span key={id} title={c.hint} style={{ fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 99, color: '#fff', background: phaseColor(c.phase), opacity: i === 0 ? 1 : 0.82 }}>{i === 0 && list.length > 1 ? '★ ' : ''}{c.name}</span>
      })}
    </span>
  )
}

export default function ConceptPicker({ value, onChange, readOnly, testId = 'concepts', max = MAX_CONCEPTS }: { value: readonly string[]; onChange: (ids: string[]) => void; readOnly?: boolean; testId?: string; max?: number }) {
  const ids = normalizeConceptIds(value)
  const [open, setOpen] = useState(false)
  const [phase, setPhase] = useState<PhaseId>(ids.length ? conceptById(ids[0])!.phase : 'ballbesitz')
  const chip = (active: boolean, color: string): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '6px 10px', minHeight: 32, borderRadius: 99, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', textAlign: 'left',
    border: `1.5px solid ${active ? color : '#E4E6EE'}`, background: active ? color : '#fff', color: active ? '#fff' : '#33384a',
  })

  return (
    <div data-testid={testId}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        {ids.length === 0 && <span style={{ fontSize: 12, color: MUTED }}>{readOnly ? 'Nicht zugeordnet.' : 'Noch nichts gewählt.'}</span>}
        {ids.map((id, i) => {
          const c = conceptById(id)!
          return (
            <span key={id} data-testid="concept-chip" title={c.hint} style={{ display: 'inline-flex', alignItems: 'center', gap: 2, borderRadius: 99, background: phaseColor(c.phase), color: '#fff', fontSize: 12, fontWeight: 600, padding: '3px 4px 3px 10px' }}>
              {i === 0 && ids.length > 1 ? '★ ' : ''}{c.name}
              {!readOnly && i > 0 && <button data-testid="concept-main" aria-label="Zum Hauptkonzept machen" title="Zum Hauptkonzept machen" onClick={() => onChange(makeMain(ids, id))} style={{ border: 'none', background: 'none', color: '#fff', cursor: 'pointer', padding: '0 4px', fontSize: 12 }}>↑</button>}
              {!readOnly && <button data-testid="concept-remove" aria-label={`${c.name} entfernen`} onClick={() => onChange(toggleConcept(ids, id))} style={{ border: 'none', background: 'none', color: '#fff', cursor: 'pointer', padding: '0 6px', fontSize: 14, lineHeight: 1 }}>×</button>}
            </span>
          )
        })}
        {!readOnly && <button data-testid="concepts-toggle" onClick={() => setOpen(o => !o)} style={{ fontSize: 12, fontWeight: 600, padding: '6px 12px', minHeight: 32, borderRadius: 99, border: `1.5px solid ${NAVY}`, background: open ? '#e8edf8' : '#fff', color: NAVY, cursor: 'pointer', fontFamily: 'DM Sans, sans-serif' }}>{open ? 'Fertig' : '+ Spielmodell'}</button>}
      </div>
      {open && !readOnly && (
        <div data-testid="concepts-panel" style={{ marginTop: 8, border: '1px solid #E4E6EE', borderRadius: 12, padding: 10, background: '#fcfcfb' }}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
            {PHASES.map(p => <button key={p.id} data-testid={`phase-${p.id}`} onClick={() => setPhase(p.id)} style={chip(phase === p.id, phaseColor(p.id))}>{p.name}</button>)}
          </div>
          {groupsOfPhase(phase).map(g => (
            <div key={g.group} style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 4 }}>{g.group}</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {g.items.map(c => {
                  const on = ids.includes(c.id)
                  return <button key={c.id} data-testid={`concept-${c.id}`} title={c.hint} disabled={!on && ids.length >= max} onClick={() => onChange(toggleConcept(ids, c.id))} style={{ ...chip(on, phaseColor(c.phase)), opacity: !on && ids.length >= max ? 0.45 : 1 }}>{c.name}</button>
                })}
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11, color: MUTED }}>Das erste gewählte ist das Hauptkonzept (★). Bis zu {max} möglich.</div>
        </div>
      )}
    </div>
  )
}
