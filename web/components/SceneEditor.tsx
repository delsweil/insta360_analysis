'use client'

// The sketch editor: tool bar, selection panel, undo/redo, background and size, and the canvas.
// It owns the editing history; the parent gets every committed state through onChange and saves it. Give it a `key` per scene.

import { useCallback, useEffect, useRef, useState } from 'react'
import SceneCanvas, { equipGlyph, INK, POLE_COLORS, type Look } from './SceneCanvas'
import {
  TEAMS, TEAM_COLORS, TEAM_NAMES, EQUIP_KINDS, EQUIP_LABELS, LINE_LABELS, BG_LABELS, defaultBackground, clampFrameSize,
  duplicateShape, removeShape, updateShape, equipSize, symbolUnit, newId, hist, hPush, hUndo, hRedo,
  type Scene, type SceneShape, type Tool, type Team, type EquipKind, type BackgroundType, type Pt, type Hist, type LineKind,
} from '@/lib/scene'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: 'select', label: 'Auswahl', hint: 'Tippe eine Figur an, um sie zu verschieben. Runde Griffe ändern Länge, Bogen und Drehung.' },
  { id: 'player', label: 'Spieler', hint: 'Tippen setzt einen Spieler, gedrückt halten und ziehen positioniert ihn.' },
  { id: 'ball', label: 'Ball', hint: 'Tippen setzt den Ball.' },
  { id: 'equip', label: 'Material', hint: 'Material unten wählen, dann tippen.' },
  { id: 'pass', label: 'Pass', hint: 'Vom Start zum Ziel ziehen. Die Enden rasten an Spielern ein.' },
  { id: 'return', label: 'Rückpass', hint: 'Vom Start zum Ziel ziehen. Die Enden rasten an Spielern ein.' },
  { id: 'move', label: 'Laufweg', hint: 'Vom Start zum Ziel ziehen. Die Enden rasten an Spielern ein.' },
  { id: 'dim', label: 'Maß', hint: 'Zwischen zwei Punkten ziehen. Die Länge in Metern steht automatisch dabei.' },
  { id: 'grid', label: 'Feld', hint: 'Ein Feld aufziehen. Die Größe in Metern steht automatisch dabei.' },
  { id: 'pen', label: 'Stift', hint: 'Frei zeichnen.' },
  { id: 'text', label: 'Text', hint: 'Dorthin tippen, wo der Text stehen soll.' },
]

function ToolIcon({ id }: { id: Tool }) {
  const c = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }
  const w = 30, h = 20
  const svg = (children: React.ReactNode) => <svg width={w} height={h} viewBox="0 0 30 20" aria-hidden>{children}</svg>
  switch (id) {
    case 'select': return svg(<path d="M10 3L10 15L13.5 12L16 17.5L18.5 16.4L16 11L20.5 11Z" {...c} />)
    case 'player': return svg(<circle cx="15" cy="10" r="6" fill="#e94fae" stroke="currentColor" strokeWidth="1.8" />)
    case 'ball': return svg(<g><circle cx="15" cy="10" r="6" fill="#fff" stroke="currentColor" strokeWidth="1.8" /><path d="M15 7L17.4 8.8L16.5 11.6L13.5 11.6L12.6 8.8Z" fill="currentColor" /></g>)
    case 'equip': return svg(<path d="M15 3L21 16H9Z" fill="#f59e0b" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />)
    case 'pass': return svg(<path d="M3 10H25M19 5L25 10L19 15" {...c} />)
    case 'return': return svg(<path d="M3 10H25M19 5L25 10L19 15" {...c} strokeDasharray="4 3" />)
    case 'move': return svg(<path d="M3 10q3 -5 6 0t6 0t6 0M21 5L27 10L21 15" {...c} />)
    case 'dim': return svg(<path d="M4 10H26M4 5V15M26 5V15M8 7L4 10L8 13M22 7L26 10L22 13" {...c} strokeWidth="1.5" />)
    case 'grid': return svg(<rect x="4" y="3" width="22" height="14" {...c} strokeDasharray="5 3" />)
    case 'pen': return svg(<path d="M4 15C8 4 11 4 14 10S20 16 26 5" {...c} />)
    case 'text': return svg(<path d="M8 4H22M15 4V17" {...c} strokeWidth="2.2" />)
  }
}

function EquipIcon({ kind, color }: { kind: EquipKind; color?: string }) {
  const z = equipSize(kind, 1)
  const half = Math.max(z.w, z.d) / 2 * 1.25
  return (
    <svg width={34} height={26} viewBox={`${-half} ${-half * 0.76} ${half * 2} ${half * 1.52}`} aria-hidden>
      {equipGlyph(kind, 1, half / 20, { color, look: 'clean' })}
    </svg>
  )
}

interface Props {
  scene: Scene
  onChange: (s: Scene) => void
  readOnly?: boolean
}

export default function SceneEditor({ scene, onChange, readOnly = false }: Props) {
  const [h, setH] = useState<Hist<Scene>>(() => hist(scene))
  const [tool, setTool] = useState<Tool>('select')
  const [team, setTeam] = useState<Team>('a')
  const [equip, setEquip] = useState<EquipKind>('cone')
  const [poleColor, setPoleColor] = useState(POLE_COLORS[0])
  const [look, setLook] = useState<Look>('sketch')
  const [snap, setSnap] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [pendingText, setPendingText] = useState<Pt | null>(null)
  const [textValue, setTextValue] = useState('')
  const [full, setFull] = useState(false)
  const lastKey = useRef<{ key: string; at: number } | null>(null)
  const cur = h.present

  const emit = useCallback((next: Scene) => onChange(next), [onChange])

  /** Add a history step. Edits with the same key within 1.2 s (typing in a field) are merged into one step. */
  const apply = useCallback((next: Scene, key?: string) => {
    const now = Date.now()
    const merge = !!key && lastKey.current?.key === key && now - lastKey.current.at < 1200
    lastKey.current = key ? { key, at: now } : null
    setH(prev => (merge ? { ...prev, present: next } : hPush(prev, next)))
    emit(next)
  }, [emit])

  const setShapes = useCallback((shapes: SceneShape[], key?: string) => apply({ ...cur, shapes }, key), [apply, cur])

  const undo = useCallback(() => { const n = hUndo(h); if (n !== h) { setH(n); emit(n.present) } setSelectedId(null) }, [h, emit])
  const redo = useCallback(() => { const n = hRedo(h); if (n !== h) { setH(n); emit(n.present) } setSelectedId(null) }, [h, emit])

  const selected = cur.shapes.find(s => s.id === selectedId)

  const patchSel = useCallback((patch: Partial<SceneShape>, key?: string) => {
    if (!selectedId) return
    setShapes(updateShape(cur.shapes, selectedId, patch), key)
  }, [cur.shapes, selectedId, setShapes])

  const deleteSel = useCallback(() => { if (selectedId) { setShapes(removeShape(cur.shapes, selectedId)); setSelectedId(null) } }, [cur.shapes, selectedId, setShapes])
  const duplicateSel = useCallback(() => {
    if (!selected) return
    const o = symbolUnit(cur.background) * 1.6
    const d = duplicateShape(selected, o, o)
    setShapes([...cur.shapes, d]); setSelectedId(d.id)
  }, [selected, cur, setShapes])

  useEffect(() => {
    if (readOnly) return
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
      else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo() }
      else if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSel() }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) { e.preventDefault(); deleteSel() }
      else if (e.key === 'Escape') { setSelectedId(null); setPendingText(null); setTool('select'); setFull(false) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [readOnly, undo, redo, duplicateSel, deleteSel, selectedId])

  function addText() {
    const t = textValue.trim()
    if (pendingText && t) {
      const s: SceneShape = { id: newId(), type: 'text', pos: pendingText, text: t, size: 1 }
      setShapes([...cur.shapes, s]); setSelectedId(s.id)
    }
    setPendingText(null); setTextValue('')
  }

  const setBg = (type: BackgroundType) => apply({ ...cur, background: defaultBackground(type) })
  const setSize = (which: 'width_m' | 'height_m', v: number) => apply({ ...cur, background: { ...cur.background, [which]: clampFrameSize(v) } }, 'size')

  const btn = (active: boolean, extra: React.CSSProperties = {}): React.CSSProperties => ({
    fontSize: 12, fontWeight: 600, padding: '7px 11px', minHeight: 38, borderRadius: 9, cursor: 'pointer', whiteSpace: 'nowrap',
    border: `1.5px solid ${active ? NAVY : '#E4E6EE'}`, background: active ? '#e8edf8' : '#fff', color: active ? NAVY : '#4A4F5C',
    fontFamily: 'DM Sans, sans-serif', ...extra,
  })
  const small: React.CSSProperties = { fontSize: 11, fontWeight: 600, color: MUTED, letterSpacing: '0.05em', textTransform: 'uppercase' }
  const input: React.CSSProperties = { fontSize: 13, padding: '7px 10px', borderRadius: 8, border: '1px solid #E4E6EE', fontFamily: 'DM Sans, sans-serif', outline: 'none', minHeight: 38, boxSizing: 'border-box' }

  const hint = TOOLS.find(t => t.id === tool)?.hint ?? ''
  const bgType = cur.background.type

  const wrap: React.CSSProperties = full
    ? { position: 'fixed', inset: 0, zIndex: 60, background: '#F8F8F6', display: 'flex', flexDirection: 'column', padding: 10, overflow: 'auto' }
    : { display: 'flex', flexDirection: 'column' }

  if (readOnly) {
    return <SceneCanvas scene={cur} look={look} testId="scene-readonly" />
  }

  return (
    <div data-testid="scene-editor" style={wrap}>
      {/* top bar: background, size, look, snap, history */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center', marginBottom: 8 }}>
        {(['pitch', 'halfpitch', 'free', 'blank'] as BackgroundType[]).map(t => (
          <button key={t} data-testid={`bg-${t}`} onClick={() => setBg(t)} style={btn(bgType === t)}>{BG_LABELS[t]}</button>
        ))}
        {(bgType === 'free' || bgType === 'blank') && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <input data-testid="bg-w" aria-label="Breite in Metern" type="number" step="0.5" min={3} max={120} style={{ ...input, width: 64 }} value={cur.background.width_m} onChange={e => setSize('width_m', parseFloat(e.target.value))} />
            <span style={{ fontSize: 12, color: MUTED }}>×</span>
            <input data-testid="bg-h" aria-label="Höhe in Metern" type="number" step="0.5" min={3} max={120} style={{ ...input, width: 64 }} value={cur.background.height_m} onChange={e => setSize('height_m', parseFloat(e.target.value))} />
            <span style={{ fontSize: 12, color: MUTED }}>m</span>
          </span>
        )}
        <span style={{ flex: 1 }} />
        <button data-testid="look" onClick={() => setLook(look === 'sketch' ? 'clean' : 'sketch')} style={btn(false)} title="Aussehen">{look === 'sketch' ? 'Skizze' : 'Sauber'}</button>
        <button data-testid="snap" onClick={() => setSnap(!snap)} style={btn(snap)} title="Auf ganze Meter einrasten">Raster</button>
        <button data-testid="undo" onClick={undo} disabled={!h.past.length} style={btn(false, { opacity: h.past.length ? 1 : 0.4 })} title="Rückgängig (Strg+Z)">↶</button>
        <button data-testid="redo" onClick={redo} disabled={!h.future.length} style={btn(false, { opacity: h.future.length ? 1 : 0.4 })} title="Wiederholen">↷</button>
        <button data-testid="fullscreen" onClick={() => setFull(!full)} style={btn(full)} title="Vollbild">{full ? 'Schließen' : 'Vollbild'}</button>
      </div>

      {/* tools */}
      <div data-testid="toolbar" style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4, marginBottom: 6 }}>
        {TOOLS.map(t => (
          <button key={t.id} data-testid={`tool-${t.id}`} onClick={() => { setTool(t.id); if (t.id !== 'select') setSelectedId(null) }} style={btn(tool === t.id, { display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 1, padding: '5px 9px', minWidth: 56 })}>
            <ToolIcon id={t.id} />
            <span style={{ fontSize: 10.5 }}>{t.label}</span>
          </button>
        ))}
      </div>

      {/* options of the active tool */}
      {tool === 'player' && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center', marginBottom: 6 }}>
          {TEAMS.map(t => (
            <button key={t} data-testid={`team-${t}`} onClick={() => setTeam(t)} style={btn(team === t, { display: 'inline-flex', alignItems: 'center', gap: 6 })}>
              <span style={{ width: 14, height: 14, borderRadius: 99, background: TEAM_COLORS[t], border: `2px solid ${INK}`, display: 'inline-block' }} />{TEAM_NAMES[t]}
            </button>
          ))}
        </div>
      )}
      {tool === 'equip' && (
        <div style={{ marginBottom: 6 }}>
          <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
            {EQUIP_KINDS.map(k => (
              <button key={k} data-testid={`equip-${k}`} onClick={() => setEquip(k)} style={btn(equip === k, { display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 1, padding: '4px 8px', minWidth: 64 })}>
                <EquipIcon kind={k} color={poleColor} /><span style={{ fontSize: 10 }}>{EQUIP_LABELS[k]}</span>
              </button>
            ))}
          </div>
          {equip === 'colorpole' && (
            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
              {POLE_COLORS.map(c => <button key={c} aria-label={`Farbe ${c}`} onClick={() => setPoleColor(c)} style={{ width: 30, height: 30, borderRadius: 99, background: c, border: `2.5px solid ${poleColor === c ? NAVY : '#cfd4e2'}`, cursor: 'pointer' }} />)}
            </div>
          )}
        </div>
      )}
      <div style={{ ...small, textTransform: 'none', letterSpacing: 0, fontWeight: 500, minHeight: 18, marginBottom: 6 }}>{hint}</div>

      {pendingText && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
          <input data-testid="text-input" autoFocus value={textValue} onChange={e => setTextValue(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') addText(); if (e.key === 'Escape') { setPendingText(null); setTextValue('') } }} placeholder="Text eingeben …" style={{ ...input, flex: 1 }} />
          <button data-testid="text-ok" onClick={addText} style={btn(true)}>OK</button>
          <button onClick={() => { setPendingText(null); setTextValue('') }} style={btn(false)}>Abbrechen</button>
        </div>
      )}

      {/* canvas */}
      <div style={{ border: '1px solid #E4E6EE', borderRadius: 12, overflow: 'hidden', background: '#fbfbf8', flex: full ? 1 : undefined, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 0 }}>
        <SceneCanvas
          scene={cur}
          look={look}
          editable
          tool={tool}
          team={team}
          equip={equip}
          poleColor={poleColor}
          snap={snap}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onCommit={shapes => setShapes(shapes)}
          onRequestText={pos => { setPendingText(pos); setTextValue('') }}
          style={full ? { maxHeight: 'calc(100dvh - 250px)' } : { maxHeight: 'max(340px, calc(100dvh - 360px))' }}
        />
      </div>

      {/* selection panel */}
      {selected && tool === 'select' && (
        <div data-testid="selection-panel" style={{ marginTop: 8, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 10, display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          {selected.type === 'player' && (
            <>
              {TEAMS.map(t => <button key={t} aria-label={TEAM_NAMES[t]} onClick={() => patchSel({ team: t } as Partial<SceneShape>)} style={{ width: 32, height: 32, borderRadius: 99, background: TEAM_COLORS[t], border: `2.5px solid ${selected.team === t ? NAVY : '#cfd4e2'}`, cursor: 'pointer' }} />)}
              <input data-testid="num" placeholder="Nr." maxLength={3} value={selected.num ?? ''} onChange={e => patchSel({ num: e.target.value || undefined } as Partial<SceneShape>, 'num')} style={{ ...input, width: 64 }} />
            </>
          )}
          {selected.type === 'equip' && (
            <>
              <span style={small}>{EQUIP_LABELS[selected.kind]}</span>
              <button onClick={() => patchSel({ angle: (selected.angle + 345) % 360 } as Partial<SceneShape>)} style={btn(false)}>⟲ 15°</button>
              <button onClick={() => patchSel({ angle: (selected.angle + 15) % 360 } as Partial<SceneShape>)} style={btn(false)}>⟳ 15°</button>
              {selected.kind === 'colorpole' && (
                <>
                  {POLE_COLORS.map(c => <button key={c} aria-label={`Farbe ${c}`} onClick={() => patchSel({ color: c } as Partial<SceneShape>)} style={{ width: 28, height: 28, borderRadius: 99, background: c, border: `2.5px solid ${(selected.color ?? POLE_COLORS[0]) === c ? NAVY : '#cfd4e2'}`, cursor: 'pointer' }} />)}
                  <input placeholder="Nr." maxLength={2} value={selected.num ?? ''} onChange={e => patchSel({ num: e.target.value || undefined } as Partial<SceneShape>, 'num')} style={{ ...input, width: 60 }} />
                </>
              )}
            </>
          )}
          {selected.type === 'line' && (
            <>
              {(['pass', 'return', 'move'] as LineKind[]).map(k => <button key={k} data-testid={`linekind-${k}`} onClick={() => patchSel({ kind: k } as Partial<SceneShape>)} style={btn(selected.kind === k)}>{LINE_LABELS[k]}</button>)}
              <button onClick={() => patchSel({ from: selected.to, to: selected.from, fromId: selected.toId, toId: selected.fromId, bend: -selected.bend } as Partial<SceneShape>)} style={btn(false)}>Richtung tauschen</button>
              {selected.bend !== 0 && <button onClick={() => patchSel({ bend: 0 } as Partial<SceneShape>)} style={btn(false)}>Gerade</button>}
            </>
          )}
          {selected.type === 'dim' && (
            <input data-testid="dim-text" placeholder="Text (leer = gemessene Länge)" value={selected.text ?? ''} onChange={e => patchSel({ text: e.target.value || undefined } as Partial<SceneShape>, 'dim')} style={{ ...input, flex: 1, minWidth: 180 }} />
          )}
          {selected.type === 'grid' && (
            <>
              <input data-testid="grid-label" placeholder="Beschriftung, z. B. Feld A" value={selected.label ?? ''} onChange={e => patchSel({ label: e.target.value || undefined } as Partial<SceneShape>, 'grid')} style={{ ...input, flex: 1, minWidth: 160 }} />
              <button onClick={() => patchSel({ showSize: !selected.showSize } as Partial<SceneShape>)} style={btn(selected.showSize)}>Größe anzeigen</button>
            </>
          )}
          {selected.type === 'text' && (
            <>
              <input data-testid="text-edit" value={selected.text} onChange={e => patchSel({ text: e.target.value } as Partial<SceneShape>, 'text')} style={{ ...input, flex: 1, minWidth: 160 }} />
              <button onClick={() => patchSel({ size: Math.max(0.6, +(selected.size - 0.2).toFixed(1)) } as Partial<SceneShape>)} style={btn(false)}>A−</button>
              <button onClick={() => patchSel({ size: Math.min(3, +(selected.size + 0.2).toFixed(1)) } as Partial<SceneShape>)} style={btn(false)}>A+</button>
            </>
          )}
          <span style={{ flex: 1 }} />
          {selected.type !== 'pen' && <button data-testid="duplicate" onClick={duplicateSel} style={btn(false)}>Duplizieren</button>}
          <button data-testid="delete" onClick={deleteSel} style={btn(false, { color: '#991b1b', borderColor: '#fca5a5' })}>Löschen</button>
        </div>
      )}
    </div>
  )
}
