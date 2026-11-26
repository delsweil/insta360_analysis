'use client'

// The sketch editor: tool bar, selection panel, undo/redo, background and size, and the canvas.
// It owns the editing history; the parent gets every committed state through onChange and saves it. Give it a `key` per scene.

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import SceneCanvas, { equipGlyph, INK, POLE_COLORS, type Look } from './SceneCanvas'
import {
  TEAMS, TEAM_COLORS, TEAM_NAMES, EQUIP_KINDS, EQUIP_LABELS, LINE_LABELS, BG_LABELS, defaultBackground, clampFrameSize,
  duplicateShapes, removeShapes, updateShape, colorShapes, scaleShapes, equipSize, symbolUnit, newId, hist, hPush, hUndo, hRedo,
  COLOR_CHOICES, fieldShape, canColor, canScale, scaleOf, rotateShapes, shapesInGrid, shapeBox, unionBox, gridSize, gridResized, gridSplits, addFieldCones, addConesForGrid, frameGrid, frameSplits, cloneShapes, MAX_SPLIT,
  type Scene, type SceneShape, type Tool, type Team, type EquipKind, type BackgroundType, type Pt, type Hist, type LineKind,
} from '@/lib/scene'

const NAVY = '#0f2972'
const MUTED = '#8A8F9E'

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: 'select', label: 'Auswahl', hint: 'Figur antippen und ziehen. Auf leeren Platz tippen und einen Rahmen aufziehen wählt mehrere aus (Umschalt/Strg + Klick fügt hinzu). Runde Griffe ändern Länge, Bogen, Drehung und Größe. Löschen: Papierkorb oder Entf.' },
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

// The clipboard for copying shapes between sketches lives in localStorage so it works across forms and sessions.
const CLIP = 'sceneClipboard'
const CLIP_EVENT = 'sceneclip'
function readClip(): string { try { return localStorage.getItem(CLIP) ?? '' } catch { return '' } }
function subscribeClip(cb: () => void) {
  window.addEventListener('storage', cb); window.addEventListener(CLIP_EVENT, cb)
  return () => { window.removeEventListener('storage', cb); window.removeEventListener(CLIP_EVENT, cb) }
}

/** A number field that lets you clear and retype freely: it only applies a value once it is a valid size, and tidies the text on blur. */
function SizeInput({ value, onValue, label, testId, style, min = 3, max = Infinity }: { value: number; onValue: (v: number) => void; label: string; testId: string; style: React.CSSProperties; min?: number; max?: number }) {
  const shown = String(value).replace('.', ',')
  const [text, setText] = useState(shown)
  const [focus, setFocus] = useState(false)
  return (
    <input
      data-testid={testId}
      aria-label={label}
      inputMode="decimal"
      value={focus ? text : shown}
      style={style}
      onFocus={e => { setText(shown); setFocus(true); e.currentTarget.select() }}
      onBlur={() => setFocus(false)}
      onKeyDown={e => { if (e.key === 'Enter') (e.currentTarget as HTMLInputElement).blur() }}
      onChange={e => {
        const t = e.target.value.replace(/[^0-9.,]/g, '')
        setText(t)
        const n = parseFloat(t.replace(',', '.'))
        if (Number.isFinite(n) && n >= min && n <= max) onValue(n)
      }}
    />
  )
}

const TrashIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></svg>
const CopyIcon = () => <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></svg>

interface Props {
  scene: Scene
  onChange: (s: Scene) => void
  readOnly?: boolean
  /** Feldaufbau: forms that can be laid onto the pitch as labelled fields of their own size. */
  placeables?: { key: string; label: string; w: number; h: number; color?: string }[]
}

export default function SceneEditor({ scene, onChange, readOnly = false, placeables }: Props) {
  const [h, setH] = useState<Hist<Scene>>(() => hist(scene))
  const [tool, setTool] = useState<Tool>('select')
  const [team, setTeam] = useState<Team>('a')
  const [equip, setEquip] = useState<EquipKind>('cone')
  const [poleColor, setPoleColor] = useState(POLE_COLORS[0])
  const look: Look = 'sketch' // steady lines; 'sketch' only selects the handwriting font for labels
  const [snap, setSnap] = useState(false)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
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

  const undo = useCallback(() => { const n = hUndo(h); if (n !== h) { setH(n); emit(n.present) } setSelectedIds([]) }, [h, emit])
  const redo = useCallback(() => { const n = hRedo(h); if (n !== h) { setH(n); emit(n.present) } setSelectedIds([]) }, [h, emit])

  const ids = useMemo(() => selectedIds.filter(id => cur.shapes.some(s => s.id === id)), [selectedIds, cur.shapes])
  const picked = useMemo(() => cur.shapes.filter(s => ids.includes(s.id)), [cur.shapes, ids])
  const selected = picked.length === 1 ? picked[0] : undefined
  const many = picked.length > 1

  const patchSel = useCallback((patch: Partial<SceneShape>, key?: string) => {
    if (!selected) return
    setShapes(updateShape(cur.shapes, selected.id, patch), key)
  }, [cur.shapes, selected, setShapes])

  const deleteSel = useCallback(() => { if (ids.length) { setShapes(removeShapes(cur.shapes, ids)); setSelectedIds([]) } }, [cur.shapes, ids, setShapes])
  const duplicateSel = useCallback(() => {
    if (!ids.length) return
    const o = symbolUnit(cur.background) * 1.6
    const r = duplicateShapes(cur.shapes, ids, o, o)
    setShapes(r.shapes); setSelectedIds(r.newIds)
  }, [ids, cur, setShapes])
  const pasteN = useRef(0)
  const clipRaw = useSyncExternalStore(subscribeClip, readClip, () => '')
  const clipCount = useMemo(() => { try { const c = JSON.parse(clipRaw || 'null'); return Array.isArray(c?.shapes) ? c.shapes.length : 0 } catch { return 0 } }, [clipRaw])
  const copySel = useCallback(() => {
    if (!picked.length) return
    try { localStorage.setItem(CLIP, JSON.stringify({ shapes: picked })); pasteN.current = 0; window.dispatchEvent(new Event(CLIP_EVENT)) } catch { /* storage full or blocked */ }
  }, [picked])
  const paste = useCallback(() => {
    try {
      const c = JSON.parse(readClip() || 'null')
      if (!Array.isArray(c?.shapes) || !c.shapes.length) return
      pasteN.current += 1
      const o = symbolUnit(cur.background) * 1.6 * pasteN.current
      const copies = cloneShapes(c.shapes as SceneShape[], o, o)
      setShapes([...cur.shapes, ...copies]); setSelectedIds(copies.map(x => x.id)); setTool('select')
    } catch { /* nothing usable on the clipboard */ }
  }, [cur, setShapes])
  const turn = (deg: number) => {
    const box = unionBox(picked.map(x => shapeBox(x, cur.background)))
    if (box) setShapes(rotateShapes(cur.shapes, ids, deg, [(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]))
  }
  const recolor = (color: string | undefined) => setShapes(colorShapes(cur.shapes, ids, color))
  const resize = (f: number) => setShapes(scaleShapes(cur.shapes, ids, f))
  const colorTargets = picked.filter(canColor)
  const scaleTargets = picked.filter(canScale)
  const commonColor = colorTargets.length && colorTargets.every(s => ('color' in s ? s.color : undefined) === ('color' in colorTargets[0] ? colorTargets[0].color : undefined)) ? ('color' in colorTargets[0] ? colorTargets[0].color : undefined) : undefined

  useEffect(() => {
    if (readOnly) return
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo() }
      else if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); redo() }
      else if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicateSel() }
      else if (mod && e.key.toLowerCase() === 'c' && ids.length) { e.preventDefault(); copySel() }
      else if (mod && e.key.toLowerCase() === 'v') { e.preventDefault(); paste() }
      else if (mod && e.key.toLowerCase() === 'a') { e.preventDefault(); setTool('select'); setSelectedIds(cur.shapes.map(s => s.id)) }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && ids.length) { e.preventDefault(); deleteSel() }
      else if (e.key === 'Escape') { setSelectedIds([]); setPendingText(null); setTool('select'); setFull(false) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [readOnly, undo, redo, duplicateSel, deleteSel, copySel, paste, ids.length, cur.shapes])

  function addText() {
    const t = textValue.trim()
    if (pendingText && t) {
      const s: SceneShape = { id: newId(), type: 'text', pos: pendingText, text: t, size: 1 }
      setShapes([...cur.shapes, s]); setSelectedIds([s.id])
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
            <SizeInput testId="bg-w" label="Breite in Metern" style={{ ...input, width: 64 }} value={cur.background.width_m} onValue={v => setSize('width_m', v)} />
            <span style={{ fontSize: 12, color: MUTED }}>×</span>
            <SizeInput testId="bg-h" label="Höhe in Metern" style={{ ...input, width: 64 }} value={cur.background.height_m} onValue={v => setSize('height_m', v)} />
            <span style={{ fontSize: 12, color: MUTED }}>m</span>
          </span>
        )}
        <span style={{ flex: 1 }} />
        <button data-testid="snap" onClick={() => setSnap(!snap)} style={btn(snap)} title="Auf ganze Meter einrasten">Raster</button>
        <button data-testid="undo" onClick={undo} disabled={!h.past.length} style={btn(false, { opacity: h.past.length ? 1 : 0.4 })} title="Rückgängig (Strg+Z)">↶</button>
        <button data-testid="redo" onClick={redo} disabled={!h.future.length} style={btn(false, { opacity: h.future.length ? 1 : 0.4 })} title="Wiederholen">↷</button>
        {clipCount > 0 && <button data-testid="paste" onClick={paste} style={btn(false)} title="Aus der Zwischenablage einfügen (Strg+V), auch in anderen Skizzen">Einfügen · {clipCount}</button>}
        <button data-testid="delete-top" onClick={deleteSel} disabled={!ids.length} aria-label="Auswahl löschen" title="Auswahl löschen (Entf)" style={btn(false, { display: 'inline-flex', alignItems: 'center', opacity: ids.length ? 1 : 0.4, color: ids.length ? '#991b1b' : '#4A4F5C' })}><TrashIcon /></button>
        <button data-testid="fullscreen" onClick={() => setFull(!full)} style={btn(full)} title="Vollbild">{full ? 'Schließen' : 'Vollbild'}</button>
      </div>

      {bgType === 'free' && (
        <div data-testid="frame-tools" style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 8 }}>
          <span style={small}>Feld teilen</span>
          {(['cols', 'rows'] as const).map(key => {
            const n = frameSplits(cur.background)[key]
            return (
              <span key={key} style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                <span style={{ fontSize: 12, color: '#4A4F5C' }}>{key === 'cols' ? 'Spalten' : 'Reihen'}</span>
                <button data-testid={`fsplit-${key}-minus`} onClick={() => apply({ ...cur, background: { ...cur.background, [key]: Math.max(1, n - 1) } })} disabled={n <= 1} style={btn(false, { padding: '5px 10px', opacity: n <= 1 ? 0.4 : 1 })} aria-label={`Weniger ${key === 'cols' ? 'Spalten' : 'Reihen'}`}>−</button>
                <span data-testid={`fsplit-${key}`} style={{ minWidth: 14, textAlign: 'center', fontSize: 13, fontWeight: 700, color: NAVY }}>{n}</span>
                <button data-testid={`fsplit-${key}-plus`} onClick={() => apply({ ...cur, background: { ...cur.background, [key]: Math.min(MAX_SPLIT, n + 1) } })} disabled={n >= MAX_SPLIT} style={btn(false, { padding: '5px 10px', opacity: n >= MAX_SPLIT ? 0.4 : 1 })} aria-label={`Mehr ${key === 'cols' ? 'Spalten' : 'Reihen'}`}>+</button>
              </span>
            )
          })}
          <button data-testid="fcones-corners" onClick={() => { const r = addConesForGrid(cur.shapes, frameGrid(cur.background), cur.background, false); if (r.added.length) setShapes(r.shapes) }} style={btn(false)} title="Auf jede Ecke des Feldes ein Hütchen setzen">Hütchen an den Ecken</button>
          {(frameSplits(cur.background).cols > 1 || frameSplits(cur.background).rows > 1) && (
            <button data-testid="fcones-splits" onClick={() => { const r = addConesForGrid(cur.shapes, frameGrid(cur.background), cur.background, true); if (r.added.length) setShapes(r.shapes) }} style={btn(false)} title="Hütchen auch auf die Teilungslinien setzen">Hütchen an Ecken und Teilungen</button>
          )}
        </div>
      )}

      {placeables && (
        <div data-testid="placeables" style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center', marginBottom: 8 }}>
          <span style={small}>Aufs Feld legen</span>
          {placeables.map(pb => (
            <button key={pb.key} data-testid="place-form" onClick={() => { const f = fieldShape(cur.background, pb.label, pb.w, pb.h, cur.shapes.filter(x => x.type === 'grid').length, pb.color); setShapes([...cur.shapes, f]); setSelectedIds([f.id]); setTool('select') }} style={btn(false, { display: 'inline-flex', alignItems: 'center', gap: 6 })}>
              {pb.color && <span style={{ width: 10, height: 10, borderRadius: 3, background: pb.color, display: 'inline-block' }} />}{pb.label}<span style={{ color: MUTED, fontWeight: 500 }}>{String(pb.w).replace('.', ',')}×{String(pb.h).replace('.', ',')} m</span>
            </button>
          ))}
          <button data-testid="place-other" onClick={() => { const f = fieldShape(cur.background, 'Anderes Team', 30, 20, cur.shapes.filter(x => x.type === 'grid').length, '#8a8f9e'); setShapes([...cur.shapes, f]); setSelectedIds([f.id]); setTool('select') }} style={btn(false)}>+ Anderes Team</button>
        </div>
      )}

      {/* tools */}
      <div data-testid="toolbar" style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4, marginBottom: 6 }}>
        {TOOLS.map(t => (
          <button key={t.id} data-testid={`tool-${t.id}`} onClick={() => { setTool(t.id); if (t.id !== 'select') setSelectedIds([]) }} style={btn(tool === t.id, { display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap: 1, padding: '5px 9px', minWidth: 56 })}>
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
      <div style={{ position: 'relative', border: '1px solid #E4E6EE', borderRadius: 12, overflow: 'hidden', background: '#fbfbf8', flex: full ? 1 : undefined, display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 0 }}>
        {ids.length > 0 && tool === 'select' && (
          <div data-testid="quick-actions" style={{ position: 'absolute', top: 8, right: 8, zIndex: 2, display: 'flex', gap: 6, alignItems: 'center', background: 'rgba(255,255,255,0.94)', border: '1px solid #E4E6EE', borderRadius: 10, padding: 4, boxShadow: '0 2px 8px rgba(15,41,114,0.12)' }}>
            {many && <span style={{ fontSize: 12, fontWeight: 600, color: NAVY, padding: '0 6px' }}>{picked.length} gewählt</span>}
            <button data-testid="quick-duplicate" onClick={duplicateSel} aria-label="Duplizieren" title="Duplizieren (Strg+D)" style={btn(false, { padding: 0, width: 40, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' })}><CopyIcon /></button>
            <button data-testid="quick-delete" onClick={deleteSel} aria-label="Löschen" title="Löschen (Entf)" style={btn(false, { padding: 0, width: 40, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', color: '#991b1b', borderColor: '#fca5a5' })}><TrashIcon /></button>
          </div>
        )}
        <SceneCanvas
          scene={cur}
          look={look}
          editable
          tool={tool}
          team={team}
          equip={equip}
          poleColor={poleColor}
          snap={snap}
          selectedIds={ids}
          onSelect={setSelectedIds}
          onCommit={shapes => setShapes(shapes)}
          onRequestText={pos => { setPendingText(pos); setTextValue('') }}
          style={full ? { maxHeight: 'calc(100dvh - 250px)' } : { maxHeight: 'max(340px, calc(100dvh - 360px))' }}
        />
      </div>

      {/* selection panel */}
      {picked.length > 0 && tool === 'select' && (
        <div data-testid="selection-panel" style={{ marginTop: 8, background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, padding: 10, display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
          {selected?.type === 'player' && (
            <>
              {TEAMS.map(t => <button key={t} aria-label={TEAM_NAMES[t]} onClick={() => patchSel({ team: t, color: undefined } as Partial<SceneShape>)} style={{ width: 32, height: 32, borderRadius: 99, background: TEAM_COLORS[t], border: `2.5px solid ${!selected.color && selected.team === t ? NAVY : '#cfd4e2'}`, cursor: 'pointer' }} />)}
              <input data-testid="num" placeholder="Nr." maxLength={3} value={selected.num ?? ''} onChange={e => patchSel({ num: e.target.value || undefined } as Partial<SceneShape>, 'num')} style={{ ...input, width: 64 }} />
            </>
          )}
          {selected?.type === 'equip' && (
            <>
              <span style={small}>{EQUIP_LABELS[selected.kind]}</span>
              <button onClick={() => patchSel({ angle: (selected.angle + 345) % 360 } as Partial<SceneShape>)} style={btn(false)}>⟲ 15°</button>
              <button onClick={() => patchSel({ angle: (selected.angle + 15) % 360 } as Partial<SceneShape>)} style={btn(false)}>⟳ 15°</button>
              {selected.kind === 'colorpole' && (
                <input placeholder="Nr." maxLength={2} value={selected.num ?? ''} onChange={e => patchSel({ num: e.target.value || undefined } as Partial<SceneShape>, 'num')} style={{ ...input, width: 60 }} />
              )}
            </>
          )}
          {selected?.type === 'line' && (
            <>
              {(['pass', 'return', 'move'] as LineKind[]).map(k => <button key={k} data-testid={`linekind-${k}`} onClick={() => patchSel({ kind: k } as Partial<SceneShape>)} style={btn(selected.kind === k)}>{LINE_LABELS[k]}</button>)}
              <button onClick={() => patchSel({ from: selected.to, to: selected.from, fromId: selected.toId, toId: selected.fromId, bend: -selected.bend } as Partial<SceneShape>)} style={btn(false)}>Richtung tauschen</button>
              {selected.bend !== 0 && <button onClick={() => patchSel({ bend: 0 } as Partial<SceneShape>)} style={btn(false)}>Gerade</button>}
            </>
          )}
          {selected?.type === 'dim' && (
            <input data-testid="dim-text" placeholder="Text (leer = gemessene Länge)" value={selected.text ?? ''} onChange={e => patchSel({ text: e.target.value || undefined } as Partial<SceneShape>, 'dim')} style={{ ...input, flex: 1, minWidth: 180 }} />
          )}
          {selected?.type === 'grid' && (
            <>
              <input data-testid="grid-label" placeholder="Beschriftung, z. B. Feld A" value={selected.label ?? ''} onChange={e => patchSel({ label: e.target.value || undefined } as Partial<SceneShape>, 'grid')} style={{ ...input, flex: 1, minWidth: 160 }} />
              <button onClick={() => patchSel({ showSize: !selected.showSize } as Partial<SceneShape>)} style={btn(selected.showSize)}>Größe anzeigen</button>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <SizeInput testId="grid-w" label="Breite des Feldes in Metern" min={1} max={200} style={{ ...input, width: 64 }} value={Math.round(gridSize(selected).w * 10) / 10} onValue={v => patchSel(gridResized(selected, v, gridSize(selected).h), 'gridsize')} />
                <span style={{ fontSize: 12, color: MUTED }}>×</span>
                <SizeInput testId="grid-h" label="Höhe des Feldes in Metern" min={1} max={200} style={{ ...input, width: 64 }} value={Math.round(gridSize(selected).h * 10) / 10} onValue={v => patchSel(gridResized(selected, gridSize(selected).w, v), 'gridsize')} />
                <span style={{ fontSize: 12, color: MUTED }}>m</span>
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <SizeInput testId="grid-angle" label="Drehung in Grad" min={0} max={360} style={{ ...input, width: 56 }} value={Math.round(selected.angle ?? 0)} onValue={v => patchSel({ angle: v % 360 || undefined } as Partial<SceneShape>, 'gridangle')} />
                <span style={{ fontSize: 12, color: MUTED }}>°</span>
              </span>
              <span data-testid="grid-split" style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <span style={small}>Teilen</span>
                {(['cols', 'rows'] as const).map(key => {
                  const n = gridSplits(selected)[key]
                  return (
                    <span key={key} style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                      <span style={{ fontSize: 12, color: '#4A4F5C' }}>{key === 'cols' ? 'Spalten' : 'Reihen'}</span>
                      <button data-testid={`split-${key}-minus`} onClick={() => patchSel({ [key]: Math.max(1, n - 1) || undefined } as Partial<SceneShape>)} disabled={n <= 1} style={btn(false, { padding: '5px 10px', opacity: n <= 1 ? 0.4 : 1 })} aria-label={`Weniger ${key === 'cols' ? 'Spalten' : 'Reihen'}`}>−</button>
                      <span data-testid={`split-${key}`} style={{ minWidth: 14, textAlign: 'center', fontSize: 13, fontWeight: 700, color: NAVY }}>{n}</span>
                      <button data-testid={`split-${key}-plus`} onClick={() => patchSel({ [key]: Math.min(MAX_SPLIT, n + 1) } as Partial<SceneShape>)} disabled={n >= MAX_SPLIT} style={btn(false, { padding: '5px 10px', opacity: n >= MAX_SPLIT ? 0.4 : 1 })} aria-label={`Mehr ${key === 'cols' ? 'Spalten' : 'Reihen'}`}>+</button>
                    </span>
                  )
                })}
              </span>
              <button data-testid="cones-corners" onClick={() => { const r = addFieldCones(cur.shapes, selected.id, cur.background, false); if (r.added.length) setShapes(r.shapes) }} style={btn(false)} title="Auf jede Ecke des Feldes ein Hütchen setzen">Hütchen an den Ecken</button>
              {(gridSplits(selected).cols > 1 || gridSplits(selected).rows > 1) && (
                <button data-testid="cones-splits" onClick={() => { const r = addFieldCones(cur.shapes, selected.id, cur.background, true); if (r.added.length) setShapes(r.shapes) }} style={btn(false)} title="Hütchen auch auf die Teilungslinien setzen">Hütchen an Ecken und Teilungen</button>
              )}
              <button data-testid="grid-select-content" onClick={() => setSelectedIds(shapesInGrid(cur.shapes, selected.id, cur.background))} style={btn(false)} title="Das Feld und alles darin auswählen, um es gemeinsam zu drehen, zu verschieben oder zu kopieren">Inhalt wählen</button>
            </>
          )}
          {selected?.type === 'text' && (
            <>
              <input data-testid="text-edit" value={selected.text} onChange={e => patchSel({ text: e.target.value } as Partial<SceneShape>, 'text')} style={{ ...input, flex: 1, minWidth: 160 }} />
              <button onClick={() => patchSel({ size: Math.max(0.6, +(selected.size - 0.2).toFixed(1)) } as Partial<SceneShape>)} style={btn(false)}>A−</button>
              <button onClick={() => patchSel({ size: Math.min(3, +(selected.size + 0.2).toFixed(1)) } as Partial<SceneShape>)} style={btn(false)}>A+</button>
            </>
          )}
          {many && <span style={{ ...small, color: NAVY }}>{picked.length} Objekte gewählt</span>}
          {colorTargets.length > 0 && (
            <span data-testid="color-row" style={{ display: 'inline-flex', gap: 5, alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={small}>Farbe</span>
              {COLOR_CHOICES.map(c => (
                <button key={c.color} data-testid={`color-${c.color.slice(1)}`} aria-label={c.name} title={c.name} onClick={() => recolor(c.color)}
                  style={{ width: 26, height: 26, borderRadius: 99, background: c.color, border: `2.5px solid ${commonColor === c.color ? '#e8780a' : '#cfd4e2'}`, cursor: 'pointer', padding: 0 }} />
              ))}
              <button data-testid="color-reset" onClick={() => recolor(undefined)} style={btn(false, { minHeight: 30, padding: '3px 8px', fontSize: 11 })} title="Standardfarbe">Standard</button>
            </span>
          )}
          {scaleTargets.length > 0 && selected?.type !== 'text' && (
            <span style={{ display: 'inline-flex', gap: 5, alignItems: 'center' }}>
              <span style={small}>Größe</span>
              <button data-testid="size-down" onClick={() => resize(1 / 1.25)} style={btn(false)} aria-label="Kleiner">−</button>
              <button data-testid="size-up" onClick={() => resize(1.25)} style={btn(false)} aria-label="Größer">+</button>
              {selected && scaleOf(selected) !== 1 && <button data-testid="size-reset" onClick={() => setShapes(scaleShapes(cur.shapes, ids, 1 / scaleOf(selected)))} style={btn(false, { fontSize: 11 })}>{Math.round(scaleOf(selected) * 100)} % · zurücksetzen</button>}
            </span>
          )}
          <span style={{ flex: 1 }} />
          {(many || selected?.type === 'grid') && (
            <span style={{ display: 'inline-flex', gap: 5, alignItems: 'center' }}>
              <span style={small}>Drehen</span>
              <button data-testid="turn-left" onClick={() => turn(-15)} style={btn(false)} aria-label="15° gegen den Uhrzeigersinn">⟲ 15°</button>
              <button data-testid="turn-right" onClick={() => turn(15)} style={btn(false)} aria-label="15° im Uhrzeigersinn">⟳ 15°</button>
              <button data-testid="turn-90" onClick={() => turn(90)} style={btn(false)} aria-label="90° im Uhrzeigersinn">90°</button>
            </span>
          )}
          {many && <button onClick={() => setSelectedIds([])} style={btn(false)}>Auswahl aufheben</button>}
          <button data-testid="copy" onClick={copySel} style={btn(false)} title="Kopieren (Strg+C). Einfügen geht auch in einer anderen Skizze.">Kopieren</button>
          <button data-testid="duplicate" onClick={duplicateSel} style={btn(false)}>Duplizieren</button>
          <button data-testid="delete" onClick={deleteSel} style={btn(false, { color: '#991b1b', borderColor: '#fca5a5' })}>Löschen</button>
        </div>
      )}
    </div>
  )
}
