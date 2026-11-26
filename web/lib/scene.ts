// Scene model for training sketches. All coordinates are metres in the scene's own frame (x to the right, y down),
// so a distance on the drawing is a real distance. The pure logic lives here (and is tested); the editor only draws and listens.
//
// Shapes carry optional t0/t1/kf animation fields for step 2, like the annotation shapes do.
// Ids are kept when a scene is copied: the same id in the next scene means "the same player", which animation will use later.

import { curvePoints, dist, trimPolyline, type Pt } from './sketch.ts'

export type { Pt }

export type BackgroundType = 'pitch' | 'halfpitch' | 'free' | 'blank'
export interface SceneBackground { type: BackgroundType; width_m: number; height_m: number }

export type Team = 'a' | 'b' | 'c' | 'd'
export const TEAM_COLORS: Record<Team, string> = { a: '#e94fae', b: '#49b05c', c: '#3b82e8', d: '#f5c542' }
export const TEAM_NAMES: Record<Team, string> = { a: 'Ballbesitz', b: 'Wartend', c: 'Gegner', d: 'Torwart' }
export const TEAMS: Team[] = ['a', 'b', 'c', 'd']

export type EquipKind = 'cone' | 'colorpole' | 'pole' | 'hurdle' | 'ring' | 'rebounder' | 'minigoal' | 'kleinfeldgoal' | 'goal'
export const EQUIP_KINDS: EquipKind[] = ['cone', 'colorpole', 'pole', 'hurdle', 'ring', 'rebounder', 'minigoal', 'kleinfeldgoal', 'goal']
export const EQUIP_LABELS: Record<EquipKind, string> = {
  cone: 'Hütchen', colorpole: 'Farbstange', pole: 'Slalomstange', hurdle: 'Hürde', ring: 'Koordinationsring',
  rebounder: 'Rebounder', minigoal: 'Minitor', kleinfeldgoal: 'Kleinfeldtor', goal: 'Großes Tor',
}
export const GOAL_KINDS: EquipKind[] = ['minigoal', 'kleinfeldgoal', 'goal']
/** Real widths in metres. The goals are drawn at these sizes; a minigoal never gets smaller than a readable symbol. */
export const GOAL_WIDTH_M = { minigoal: 1.2, kleinfeldgoal: 5, goal: 7.32 } as const
export const GOAL_DEPTH_M = { minigoal: 0.8, kleinfeldgoal: 1.5, goal: 2 } as const

export type LineKind = 'pass' | 'return' | 'move'
export const LINE_LABELS: Record<LineKind, string> = { pass: 'Pass', return: 'Rückpass', move: 'Laufweg' }

export type Animated = { t0?: number; t1?: number; kf?: { t: number; v: unknown }[] }

export type SceneShape = { id: string } & Animated & (
  | { type: 'player'; pos: Pt; team: Team; num?: string }
  | { type: 'ball'; pos: Pt }
  | { type: 'equip'; kind: EquipKind; pos: Pt; angle: number; color?: string; num?: string }
  | { type: 'line'; kind: LineKind; from: Pt; to: Pt; bend: number; fromId?: string; toId?: string }
  | { type: 'dim'; a: Pt; b: Pt; text?: string }
  | { type: 'grid'; a: Pt; b: Pt; label?: string; showSize: boolean }
  | { type: 'pen'; points: Pt[]; color: string }
  | { type: 'text'; pos: Pt; text: string; size: number }
)
export type ShapeType = SceneShape['type']

export interface Scene { background: SceneBackground; shapes: SceneShape[]; caption: string }

export type Tool = 'select' | 'player' | 'ball' | 'equip' | 'pass' | 'return' | 'move' | 'dim' | 'grid' | 'pen' | 'text'

// ── Frame ──────────────────────────────────────────────────────

export const BG_LABELS: Record<BackgroundType, string> = { pitch: 'Vollfeld', halfpitch: 'Halbfeld', free: 'Feld (frei)', blank: 'Leer' }

export function defaultBackground(type: BackgroundType): SceneBackground {
  if (type === 'pitch') return { type, width_m: 105, height_m: 68 }
  if (type === 'halfpitch') return { type, width_m: 68, height_m: 52.5 }
  if (type === 'free') return { type, width_m: 20, height_m: 30 }
  return { type, width_m: 40, height_m: 30 }
}

export const MIN_FRAME_M = 3
export const MAX_FRAME_M = 120

export function clampFrameSize(v: number): number {
  if (!Number.isFinite(v)) return MIN_FRAME_M
  return Math.round(Math.min(MAX_FRAME_M, Math.max(MIN_FRAME_M, v)) * 10) / 10
}

export function emptyScene(type: BackgroundType = 'free'): Scene {
  return { background: defaultBackground(type), shapes: [], caption: '' }
}

export interface ViewBox { x: number; y: number; w: number; h: number }
/** The visible area: the frame plus a margin for goals, distance labels and anything placed just outside. */
export function viewBoxOf(bg: SceneBackground): ViewBox {
  const m = Math.max(bg.width_m, bg.height_m) * 0.07
  return { x: -m, y: -m, w: bg.width_m + 2 * m, h: bg.height_m + 2 * m }
}

/** Size of the schematic symbols (a player's radius) in metres: about 3 % of the longest side, so a drawing reads at any frame size. */
export function symbolUnit(bg: SceneBackground): number {
  return Math.max(0.6, Math.max(bg.width_m, bg.height_m) * 0.03)
}

export function clampPt(p: Pt, bg: SceneBackground): Pt {
  const v = viewBoxOf(bg)
  return [Math.min(v.x + v.w, Math.max(v.x, p[0])), Math.min(v.y + v.h, Math.max(v.y, p[1]))]
}

export function snapStep(bg: SceneBackground): number { return Math.max(bg.width_m, bg.height_m) <= 60 ? 1 : 5 }
export function snapPt(p: Pt, step: number): Pt { return [Math.round(p[0] / step) * step, Math.round(p[1] / step) * step] }

// ── Numbers as text ────────────────────────────────────────────

const de = (n: number) => String(n).replace('.', ',')
export function formatMetres(m: number): string {
  const r = Math.round(m * 10) / 10
  return `${de(r)} m`
}
export function dimText(s: Extract<SceneShape, { type: 'dim' }>): string {
  const t = s.text?.trim()
  return t ? t : formatMetres(dist(s.a, s.b))
}
export function gridSizeText(s: Extract<SceneShape, { type: 'grid' }>): string {
  const w = Math.round(Math.abs(s.b[0] - s.a[0]) * 10) / 10, h = Math.round(Math.abs(s.b[1] - s.a[1]) * 10) / 10
  return `${de(w)} × ${de(h)} m`
}

// ── Sizes of things ────────────────────────────────────────────

/** Width and depth (metres) of an equipment symbol before rotation. */
export function equipSize(kind: EquipKind, u: number): { w: number; d: number } {
  switch (kind) {
    case 'cone': return { w: u * 0.9, d: u * 0.9 }
    case 'colorpole': return { w: u * 0.8, d: u * 0.8 }
    case 'pole': return { w: u * 0.6, d: u * 0.6 }
    case 'hurdle': return { w: u * 2.2, d: u * 0.5 }
    case 'ring': return { w: u * 1.7, d: u * 1.7 }
    case 'rebounder': return { w: u * 2.2, d: u * 0.7 }
    case 'minigoal': return { w: Math.max(GOAL_WIDTH_M.minigoal, u * 1.8), d: Math.max(GOAL_DEPTH_M.minigoal, u * 1.0) }
    case 'kleinfeldgoal': return { w: GOAL_WIDTH_M.kleinfeldgoal, d: GOAL_DEPTH_M.kleinfeldgoal }
    case 'goal': return { w: GOAL_WIDTH_M.goal, d: GOAL_DEPTH_M.goal }
  }
}
export function ballRadius(u: number): number { return u * 0.38 }
export function textHeight(size: number, u: number): number { return u * 0.95 * size }

// ── Making shapes ──────────────────────────────────────────────

export const newId = () => Math.random().toString(36).slice(2, 10)

export function makeShape(tool: Tool, p: Pt, opts: { team?: Team; equip?: EquipKind; color?: string } = {}): SceneShape | null {
  const id = newId()
  switch (tool) {
    case 'player': return { id, type: 'player', pos: p, team: opts.team ?? 'a' }
    case 'ball': return { id, type: 'ball', pos: p }
    case 'equip': return { id, type: 'equip', kind: opts.equip ?? 'cone', pos: p, angle: 0, ...(opts.color ? { color: opts.color } : {}) }
    case 'pass': case 'return': case 'move': return { id, type: 'line', kind: tool, from: p, to: p, bend: 0 }
    case 'dim': return { id, type: 'dim', a: p, b: p }
    case 'grid': return { id, type: 'grid', a: p, b: p, showSize: true }
    case 'pen': return { id, type: 'pen', points: [p], color: '#1b2a6b' }
    case 'text': return { id, type: 'text', pos: p, text: '', size: 1 }
    default: return null
  }
}

/** The shape is big enough to keep after a drag-to-create gesture (a plain tap on a line tool makes nothing). */
export function isMeaningful(s: SceneShape, minLen: number): boolean {
  switch (s.type) {
    case 'line': return dist(s.from, s.to) >= minLen
    case 'dim': return dist(s.a, s.b) >= minLen
    case 'grid': return Math.abs(s.b[0] - s.a[0]) >= minLen && Math.abs(s.b[1] - s.a[1]) >= minLen
    case 'pen': return s.points.length >= 2
    case 'text': return s.text.trim().length > 0
    default: return true
  }
}

export function duplicateShape(s: SceneShape, dx: number, dy: number): SceneShape {
  const moved = movePatch(s, dx, dy)
  const c = { ...s, ...moved, id: newId() } as SceneShape
  if (c.type === 'line') { delete c.fromId; delete c.toId }
  return c
}

export function copyScene(s: Scene): Scene { return JSON.parse(JSON.stringify(s)) as Scene }

// ── Moving and editing ─────────────────────────────────────────

const sh = (p: Pt, dx: number, dy: number): Pt => [p[0] + dx, p[1] + dy]

export function movePatch(s: SceneShape, dx: number, dy: number): Partial<SceneShape> {
  switch (s.type) {
    case 'player': case 'ball': case 'equip': case 'text': return { pos: sh(s.pos, dx, dy) } as Partial<SceneShape>
    case 'line': return { from: sh(s.from, dx, dy), to: sh(s.to, dx, dy), fromId: undefined, toId: undefined } as Partial<SceneShape>
    case 'dim': case 'grid': return { a: sh(s.a, dx, dy), b: sh(s.b, dx, dy) } as Partial<SceneShape>
    case 'pen': return { points: s.points.map(p => sh(p, dx, dy)) } as Partial<SceneShape>
  }
}

function withPatch(s: SceneShape, patch: Partial<SceneShape>): SceneShape {
  const out = { ...s, ...patch } as Record<string, unknown>
  for (const k of Object.keys(out)) if (out[k] === undefined) delete out[k]
  return out as unknown as SceneShape
}

export function updateShape(shapes: SceneShape[], id: string, patch: Partial<SceneShape>): SceneShape[] {
  return shapes.map(s => (s.id === id ? withPatch(s, patch) : s))
}

/** Move a shape. Moving a player also carries the ends of every pass or run line attached to that player. */
export function moveShape(shapes: SceneShape[], id: string, dx: number, dy: number): SceneShape[] {
  const target = shapes.find(s => s.id === id)
  if (!target) return shapes
  return shapes.map(s => {
    if (s.id === id) return withPatch(s, movePatch(s, dx, dy))
    if (target.type === 'player' && s.type === 'line') {
      const patch: Record<string, Pt> = {}
      if (s.fromId === id) patch.from = sh(s.from, dx, dy)
      if (s.toId === id) patch.to = sh(s.to, dx, dy)
      if (Object.keys(patch).length) return withPatch(s, patch as unknown as Partial<SceneShape>)
    }
    return s
  })
}

/** Remove a shape. Lines that were attached to a deleted player keep their points but lose the attachment. */
export function removeShape(shapes: SceneShape[], id: string): SceneShape[] {
  return shapes.filter(s => s.id !== id).map(s => {
    if (s.type === 'line' && (s.fromId === id || s.toId === id)) {
      return withPatch(s, { fromId: s.fromId === id ? undefined : s.fromId, toId: s.toId === id ? undefined : s.toId } as Partial<SceneShape>)
    }
    return s
  })
}

/** The nearest player within `tol` of p, for attaching a line end to it. */
export function playerNear(shapes: SceneShape[], p: Pt, tol: number, exceptId?: string): { id: string; pos: Pt } | null {
  let best: { id: string; pos: Pt } | null = null, bd = tol
  for (const s of shapes) {
    if (s.type !== 'player' || s.id === exceptId) continue
    const d = dist(s.pos, p)
    if (d <= bd) { best = { id: s.id, pos: s.pos }; bd = d }
  }
  return best
}

// ── Layers, hit-testing, handles ───────────────────────────────

const LAYER: Record<ShapeType, number> = { grid: 0, pen: 1, equip: 2, dim: 3, line: 4, player: 5, ball: 6, text: 7 }
export const layerOf = (t: ShapeType) => LAYER[t]
/** Drawing order: grids at the back, text on top; within a layer, later shapes are on top. */
export function drawOrder(shapes: SceneShape[]): SceneShape[] {
  return shapes.map((s, i) => ({ s, i })).sort((a, b) => LAYER[a.s.type] - LAYER[b.s.type] || a.i - b.i).map(x => x.s)
}

function inRotRect(p: Pt, c: Pt, w: number, d: number, angleDeg: number, pad: number): boolean {
  const a = (angleDeg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a)
  const dx = p[0] - c[0], dy = p[1] - c[1]
  const lx = dx * cos + dy * sin, ly = -dx * sin + dy * cos
  return Math.abs(lx) <= w / 2 + pad && Math.abs(ly) <= d / 2 + pad
}
function distToPolyline(p: Pt, pts: Pt[]): number {
  let best = Infinity
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i]
    const l2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    let t = l2 === 0 ? 0 : ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / l2
    t = Math.max(0, Math.min(1, t))
    best = Math.min(best, Math.hypot(p[0] - (a[0] + t * (b[0] - a[0])), p[1] - (a[1] + t * (b[1] - a[1]))))
  }
  return best
}

/** Which shape is under p? `tol` is the touch slop in metres. Topmost first. Grids only react near their edges. */
export function hitTest(shapes: SceneShape[], p: Pt, tol: number, bg: SceneBackground): string | null {
  const u = symbolUnit(bg)
  const order = drawOrder(shapes)
  for (let i = order.length - 1; i >= 0; i--) {
    const s = order[i]
    let hit = false
    switch (s.type) {
      case 'player': hit = dist(s.pos, p) <= u + tol * 0.9; break
      case 'ball': hit = dist(s.pos, p) <= ballRadius(u) + tol; break
      case 'equip': { const z = equipSize(s.kind, u); hit = inRotRect(p, s.pos, z.w, z.d, s.angle, tol * 0.6); break }
      case 'line': hit = distToPolyline(p, curvePoints(s.from, s.to, s.bend)) <= tol; break
      case 'dim': hit = distToPolyline(p, [s.a, s.b]) <= tol; break
      case 'pen': hit = distToPolyline(p, s.points.length > 1 ? s.points : [s.points[0], s.points[0]]) <= tol; break
      case 'text': {
        const h = textHeight(s.size, u), w = Math.max(1, s.text.length) * h * 0.55
        hit = Math.abs(p[0] - s.pos[0]) <= w / 2 + tol * 0.5 && Math.abs(p[1] - s.pos[1]) <= h / 2 + tol * 0.5
        break
      }
      case 'grid': {
        const x0 = Math.min(s.a[0], s.b[0]), x1 = Math.max(s.a[0], s.b[0]), y0 = Math.min(s.a[1], s.b[1]), y1 = Math.max(s.a[1], s.b[1])
        const nearX = (Math.abs(p[0] - x0) <= tol || Math.abs(p[0] - x1) <= tol) && p[1] >= y0 - tol && p[1] <= y1 + tol
        const nearY = (Math.abs(p[1] - y0) <= tol || Math.abs(p[1] - y1) <= tol) && p[0] >= x0 - tol && p[0] <= x1 + tol
        hit = nearX || nearY
        break
      }
    }
    if (hit) return s.id
  }
  return null
}

export interface Handle { key: string; pos: Pt }

/** Where the grab points of the selected shape are. */
export function handlesOf(s: SceneShape, bg: SceneBackground, shapes?: SceneShape[]): Handle[] {
  const u = symbolUnit(bg)
  switch (s.type) {
    case 'line': {
      const L = dist(s.from, s.to) || 1
      const nx = -(s.to[1] - s.from[1]) / L, ny = (s.to[0] - s.from[0]) / L
      const m: Pt = [(s.from[0] + s.to[0]) / 2 + nx * s.bend, (s.from[1] + s.to[1]) / 2 + ny * s.bend]
      // an end that is attached to a player is drawn trimmed at the player's edge; its handle sits there, so the player itself stays grabbable
      const attached = (id?: string) => !!id && !!shapes?.some(x => x.id === id && x.type === 'player')
      const t = trimPolyline(curvePoints(s.from, s.to, s.bend, 28), attached(s.fromId) ? u * 1.25 : 0, attached(s.toId) ? u * 1.25 : 0)
      return [{ key: 'from', pos: t[0] }, { key: 'to', pos: t[t.length - 1] }, { key: 'bend', pos: m }]
    }
    case 'dim': return [{ key: 'a', pos: s.a }, { key: 'b', pos: s.b }]
    case 'grid': return [{ key: 'a', pos: s.a }, { key: 'b', pos: s.b }, { key: 'ab', pos: [s.a[0], s.b[1]] }, { key: 'ba', pos: [s.b[0], s.a[1]] }]
    case 'equip': return [{ key: 'rot', pos: rotHandle(s, bg).pos }]
    default: return []
  }
}

/** Where the rotation handle of a piece of equipment sits. It normally hangs off the front (the top at angle 0);
 *  near the edge of the picture it swings to the other side so it can always be grabbed. */
export function rotHandle(s: Extract<SceneShape, { type: 'equip' }>, bg: SceneBackground): { pos: Pt; flipped: boolean } {
  const u = symbolUnit(bg), z = equipSize(s.kind, u), r = Math.max(z.w, z.d) / 2 + u * 1.3
  const v = viewBoxOf(bg), pad = u * 0.4
  const at = (deg: number): Pt => { const a = (deg * Math.PI) / 180; return [s.pos[0] + Math.sin(a) * r, s.pos[1] - Math.cos(a) * r] }
  const inside = (p: Pt) => p[0] >= v.x + pad && p[0] <= v.x + v.w - pad && p[1] >= v.y + pad && p[1] <= v.y + v.h - pad
  const front = at(s.angle)
  if (inside(front)) return { pos: front, flipped: false }
  const back = at(s.angle + 180)
  return inside(back) ? { pos: back, flipped: true } : { pos: front, flipped: false }
}

export function handleAt(s: SceneShape, bg: SceneBackground, p: Pt, tol: number, shapes?: SceneShape[]): string | null {
  let best: string | null = null, bd = tol
  for (const h of handlesOf(s, bg, shapes)) { const d = dist(h.pos, p); if (d <= bd) { best = h.key; bd = d } }
  return best
}

/** The change to a shape when one of its handles is dragged to p. */
export function handlePatch(s: SceneShape, key: string, p: Pt, shapes: SceneShape[], bg: SceneBackground, snap?: number): Partial<SceneShape> | null {
  const u = symbolUnit(bg), tolAttach = u * 1.7
  const q = clampPt(snap ? snapPt(p, snap) : p, bg)
  if (s.type === 'line') {
    if (key === 'bend') {
      const L = dist(s.from, s.to) || 1
      const nx = -(s.to[1] - s.from[1]) / L, ny = (s.to[0] - s.from[0]) / L
      const m: Pt = [(s.from[0] + s.to[0]) / 2, (s.from[1] + s.to[1]) / 2]
      let b = (p[0] - m[0]) * nx + (p[1] - m[1]) * ny
      if (Math.abs(b) < L * 0.03) b = 0
      return { bend: Math.max(-L, Math.min(L, b)) } as Partial<SceneShape>
    }
    const near = playerNear(shapes, p, tolAttach)
    const pos = near ? near.pos : q
    return key === 'from' ? ({ from: pos, fromId: near?.id } as Partial<SceneShape>) : ({ to: pos, toId: near?.id } as Partial<SceneShape>)
  }
  if (s.type === 'dim') return (key === 'a' ? { a: q } : { b: q }) as Partial<SceneShape>
  if (s.type === 'grid') {
    if (key === 'a') return { a: q } as Partial<SceneShape>
    if (key === 'b') return { b: q } as Partial<SceneShape>
    if (key === 'ab') return { a: [q[0], s.a[1]], b: [s.b[0], q[1]] } as Partial<SceneShape>
    if (key === 'ba') return { a: [s.a[0], q[1]], b: [q[0], s.b[1]] } as Partial<SceneShape>
  }
  if (s.type === 'equip' && key === 'rot') {
    let deg = (Math.atan2(p[0] - s.pos[0], -(p[1] - s.pos[1])) * 180) / Math.PI
    if (rotHandle(s, bg).flipped) deg -= 180
    deg = Math.round(deg / 5) * 5
    return { angle: ((deg % 360) + 360) % 360 } as Partial<SceneShape>
  }
  return null
}

/** While creating a line, pull the dragged end onto a nearby player. Returns the end point and the player it attached to. */
export function lineEndTarget(shapes: SceneShape[], p: Pt, bg: SceneBackground, exceptId?: string): { pos: Pt; id?: string } {
  const near = playerNear(shapes, p, symbolUnit(bg) * 1.7, exceptId)
  return near ? { pos: near.pos, id: near.id } : { pos: p }
}

// ── Undo and redo ──────────────────────────────────────────────

export interface Hist<T> { past: T[]; present: T; future: T[] }
export const HIST_LIMIT = 100
export const hist = <T,>(present: T): Hist<T> => ({ past: [], present, future: [] })
export function hPush<T>(h: Hist<T>, next: T): Hist<T> { return { past: [...h.past, h.present].slice(-HIST_LIMIT), present: next, future: [] } }
export function hUndo<T>(h: Hist<T>): Hist<T> { return h.past.length ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] } : h }
export function hRedo<T>(h: Hist<T>): Hist<T> { return h.future.length ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) } : h }
