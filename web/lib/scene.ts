// Scene model for training sketches. All coordinates are metres in the scene's own frame (x to the right, y down),
// so a distance on the drawing is a real distance. The pure logic lives here (and is tested); the editor only draws and listens.
//
// Shapes carry optional t0/t1/kf animation fields for step 2, like the annotation shapes do.
// Ids are kept when a scene is copied: the same id in the next scene means "the same player", which animation will use later.

import { curvePoints, dist, trimPolyline, type Pt } from './sketch.ts'

export type { Pt }

export type BackgroundType = 'pitch' | 'halfpitch' | 'free' | 'blank'
export interface SceneBackground { type: BackgroundType; width_m: number; height_m: number; /** A free field can be divided by cross lines. */ cols?: number; rows?: number }

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

/** `alpha` is only set while an animation fades a shape in or out; it is never saved. */
export type SceneShape = { id: string; alpha?: number } & Animated & (
  | { type: 'player'; pos: Pt; team: Team; num?: string; color?: string; scale?: number; /** A ring around the player to point him out. */ ring?: boolean }
  | { type: 'ball'; pos: Pt; scale?: number; /** The player who has the ball at his feet: it travels with him. */ carriedBy?: string }
  | { type: 'equip'; kind: EquipKind; pos: Pt; angle: number; color?: string; num?: string; scale?: number }
  | { type: 'line'; kind: LineKind; from: Pt; to: Pt; bend: number; fromId?: string; toId?: string; color?: string }
  | { type: 'dim'; a: Pt; b: Pt; text?: string; color?: string }
  | { type: 'grid'; a: Pt; b: Pt; label?: string; showSize: boolean; color?: string; angle?: number; cols?: number; rows?: number }
  | { type: 'pen'; points: Pt[]; color: string }
  | { type: 'text'; pos: Pt; text: string; size: number; color?: string }
)
export type ShapeType = SceneShape['type']

export interface Scene { background: SceneBackground; shapes: SceneShape[]; caption: string }

export type Tool = 'select' | 'player' | 'ball' | 'equip' | 'pass' | 'return' | 'move' | 'dim' | 'grid' | 'pen' | 'text'

// ── Frame ──────────────────────────────────────────────────────

export const BG_LABELS: Record<BackgroundType, string> = { pitch: 'Vollfeld', halfpitch: 'Halbfeld', free: 'Feld (frei)', blank: 'Leer' }

/** Columns and rows a free field is divided into (1 = not divided). */
export function frameSplits(bg: SceneBackground): { cols: number; rows: number } { return gridSplits(frameGrid(bg)) }

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
/** How far an arrow key moves the selection, in metres: 1 m on a small field, 2 m on a big one; the option key makes it fine (0.2 m / 0.5 m). */
export function nudgeStep(bg: SceneBackground, fine: boolean): number {
  const small = Math.max(bg.width_m, bg.height_m) <= 60
  return fine ? (small ? 0.2 : 0.5) : (small ? 1 : 2)
}
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
export function gridCellText(s: Extract<SceneShape, { type: 'grid' }>): string {
  const { cols, rows } = gridSplits(s)
  if (cols === 1 && rows === 1) return ''
  const w = Math.round((Math.abs(s.b[0] - s.a[0]) / cols) * 10) / 10, h = Math.round((Math.abs(s.b[1] - s.a[1]) / rows) * 10) / 10
  return `je ${de(w)} × ${de(h)} m`
}
export function gridSizeText(s: Extract<SceneShape, { type: 'grid' }>): string {
  const w = Math.round(Math.abs(s.b[0] - s.a[0]) * 10) / 10, h = Math.round(Math.abs(s.b[1] - s.a[1]) * 10) / 10
  return `${de(w)} × ${de(h)} m`
}

/** A labelled field of w × h metres for the Feldaufbau, dropped near the middle of the frame and nudged by n so new ones don't sit exactly on top of each other. */
export function fieldShape(bg: SceneBackground, label: string, w: number, h: number, n: number, color?: string): SceneShape {
  const u = symbolUnit(bg), cx = bg.width_m / 2 + (n % 5) * u * 1.5 - u * 3, cy = bg.height_m / 2 + (n % 5) * u * 1.5 - u * 3
  const r = (v: number) => Math.round(v * 100) / 100
  return { id: newId(), type: 'grid', a: [r(cx - w / 2), r(cy - h / 2)], b: [r(cx + w / 2), r(cy + h / 2)], label, showSize: true, ...(color ? { color } : {}) }
}

// ── Rotated fields and group rotation ──────────────────────────

export const rotPt = (p: Pt, c: Pt, deg: number): Pt => {
  const a = (deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a), dx = p[0] - c[0], dy = p[1] - c[1]
  return [c[0] + dx * cos - dy * sin, c[1] + dx * sin + dy * cos]
}
type Grid = Extract<SceneShape, { type: 'grid' }>
export const gridCentre = (g: Grid): Pt => [(g.a[0] + g.b[0]) / 2, (g.a[1] + g.b[1]) / 2]
export const gridSize = (g: Grid): { w: number; h: number } => ({ w: Math.abs(g.b[0] - g.a[0]), h: Math.abs(g.b[1] - g.a[1]) })
/** The four corners of a field in the picture: a, (b.x,a.y), b, (a.x,b.y), turned by its angle around its centre. */
export function gridCorners(g: Grid): Pt[] {
  const c = gridCentre(g), ang = g.angle ?? 0
  return ([g.a, [g.b[0], g.a[1]], g.b, [g.a[0], g.b[1]]] as Pt[]).map(q => (ang ? rotPt(q, c, ang) : q))
}
/** Is p inside the (possibly turned) field, with a margin? */
export function inGrid(g: Grid, p: Pt, pad = 0): boolean {
  const q = g.angle ? rotPt(p, gridCentre(g), -g.angle) : p
  return q[0] >= Math.min(g.a[0], g.b[0]) - pad && q[0] <= Math.max(g.a[0], g.b[0]) + pad && q[1] >= Math.min(g.a[1], g.b[1]) - pad && q[1] <= Math.max(g.a[1], g.b[1]) + pad
}
export const MAX_SPLIT = 8
const clampSplit = (n: number | undefined) => Math.min(MAX_SPLIT, Math.max(1, Math.round(n ?? 1)))
/** How many columns and rows a field is divided into by its cross lines (1 = not divided). */
export function gridSplits(g: Grid): { cols: number; rows: number } { return { cols: clampSplit(g.cols), rows: clampSplit(g.rows) } }
/** Size of one part of a divided field, metres. */
export function gridCellSize(g: Grid): { w: number; h: number } {
  const { w, h } = gridSize(g), { cols, rows } = gridSplits(g)
  return { w: w / cols, h: h / rows }
}
/** The points where cones belong: the four corners, or with `splits` every crossing of the dividing lines and their ends on the edge. In the picture (turned with the field). */
export function gridLattice(g: Grid, splits: boolean): Pt[] {
  const c = gridCentre(g), ang = g.angle ?? 0
  const x0 = Math.min(g.a[0], g.b[0]), x1 = Math.max(g.a[0], g.b[0]), y0 = Math.min(g.a[1], g.b[1]), y1 = Math.max(g.a[1], g.b[1])
  const { cols, rows } = splits ? gridSplits(g) : { cols: 1, rows: 1 }
  const out: Pt[] = []
  for (let i = 0; i <= cols; i++) for (let j = 0; j <= rows; j++) {
    const q: Pt = [x0 + ((x1 - x0) * i) / cols, y0 + ((y1 - y0) * j) / rows]
    out.push(ang ? rotPt(q, c, ang) : q)
  }
  return out
}
/** Put a cone on each corner of a field (and on the dividing lines if asked). Spots that already have a cone are left alone. */
export function addFieldCones(shapes: SceneShape[], gridId: string, bg: SceneBackground, splits: boolean): { shapes: SceneShape[]; added: string[] } {
  const g = shapes.find(s => s.id === gridId)
  if (!g || g.type !== 'grid') return { shapes, added: [] }
  return addConesForGrid(shapes, g, bg, splits)
}

/** The frame of a free field as if it were a drawn field, so splitting and corner cones work the same for both. */
export function frameGrid(bg: SceneBackground): Extract<SceneShape, { type: 'grid' }> {
  return { id: 'frame', type: 'grid', a: [0, 0], b: [bg.width_m, bg.height_m], showSize: false, cols: bg.cols, rows: bg.rows }
}

export function addConesForGrid(shapes: SceneShape[], g: Extract<SceneShape, { type: 'grid' }>, bg: SceneBackground, splits: boolean): { shapes: SceneShape[]; added: string[] } {
  const u = symbolUnit(bg), have = shapes.filter(s => s.type === 'equip' && s.kind === 'cone') as Extract<SceneShape, { type: 'equip' }>[]
  const made: SceneShape[] = []
  for (const q of gridLattice(g, splits)) {
    if (have.some(c => dist(c.pos, q) < u * 0.5) || made.some(c => (c as { pos: Pt }).pos && dist((c as { pos: Pt }).pos, q) < u * 0.5)) continue
    const r = (v: number) => Math.round(v * 100) / 100
    made.push({ id: newId(), type: 'equip', kind: 'cone', pos: [r(q[0]), r(q[1])], angle: 0 })
  }
  return { shapes: [...shapes, ...made], added: made.map(m => m.id) }
}

/** Same centre and angle, new size (metres). */
export function gridResized(g: Grid, w: number, h: number): Partial<SceneShape> {
  const c = gridCentre(g)
  return { a: [c[0] - w / 2, c[1] - h / 2], b: [c[0] + w / 2, c[1] + h / 2] } as Partial<SceneShape>
}
export function gridNew(g: Grid, c: Pt, w: number, h: number, angle = g.angle ?? 0): Partial<SceneShape> {
  return { a: [c[0] - w / 2, c[1] - h / 2], b: [c[0] + w / 2, c[1] + h / 2], angle: ((angle % 360) + 360) % 360 || undefined } as Partial<SceneShape>
}

/** Equipment that has a front or a long side turns with a group; round or pointed-in-all-directions pieces (cones, poles, rings) only change place. */
const TURNS_WITH_GROUP: EquipKind[] = ['hurdle', 'rebounder', 'minigoal', 'kleinfeldgoal', 'goal']

/** Turn shapes by `deg` (clockwise on screen) around `centre`. Equipment and fields turn too; players, the ball and text only change place. */
export function rotateShapes(shapes: SceneShape[], ids: string[], deg: number, centre: Pt): SceneShape[] {
  const set = new Set(ids), r = (q: Pt) => rotPt(q, centre, deg)
  return shapes.map(s => {
    if (!set.has(s.id)) return s
    switch (s.type) {
      case 'ball': return withPatch(s, { pos: r(s.pos), ...(s.carriedBy && !set.has(s.carriedBy) ? { carriedBy: undefined } : {}) } as Partial<SceneShape>)
      case 'player': case 'text': return withPatch(s, { pos: r(s.pos) } as Partial<SceneShape>)
      case 'equip': return withPatch(s, { pos: r(s.pos), ...(TURNS_WITH_GROUP.includes(s.kind) ? { angle: (((s.angle + deg) % 360) + 360) % 360 } : {}) } as Partial<SceneShape>)
      case 'line': return withPatch(s, { from: r(s.from), to: r(s.to) } as Partial<SceneShape>)
      case 'dim': return withPatch(s, { a: r(s.a), b: r(s.b) } as Partial<SceneShape>)
      case 'pen': return withPatch(s, { points: s.points.map(r) } as Partial<SceneShape>)
      case 'grid': {
        const c = gridCentre(s), nc = r(c), dx = nc[0] - c[0], dy = nc[1] - c[1]
        const ang = (((s.angle ?? 0) + deg) % 360 + 360) % 360
        return withPatch(s, { a: [s.a[0] + dx, s.a[1] + dy], b: [s.b[0] + dx, s.b[1] + dy], angle: Math.abs(ang) < 1e-9 || Math.abs(ang - 360) < 1e-9 ? undefined : ang } as Partial<SceneShape>)
      }
    }
  })
}

/** The field itself plus everything whose middle lies inside it. Used to turn or move a whole set-up as one. */
export function shapesInGrid(shapes: SceneShape[], gridId: string, bg: SceneBackground): string[] {
  const g = shapes.find(s => s.id === gridId)
  if (!g || g.type !== 'grid') return []
  const out = [gridId]
  for (const s of shapes) {
    if (s.id === gridId) continue
    const b = shapeBox(s, bg)
    if (inGrid(g, [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2], symbolUnit(bg) * 0.5)) out.push(s.id) // a little slack: cones on the corners belong to the field
  }
  return out
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

export const MIN_SCALE = 0.4
export const MAX_SCALE = 4
/** Symbols (players, ball, equipment) can be made bigger or smaller; 1 is the standard size. */
export function scaleOf(s: SceneShape): number {
  return s.type === 'player' || s.type === 'ball' || s.type === 'equip' ? s.scale ?? 1 : 1
}
export const clampScale = (v: number) => Math.round(Math.min(MAX_SCALE, Math.max(MIN_SCALE, v)) * 100) / 100
export const playerRadius = (u: number, scale = 1) => u * 0.9 * scale
/** Width and depth of a piece of equipment including its own scale. */
export function equipDims(s: Extract<SceneShape, { type: 'equip' }>, u: number): { w: number; d: number } {
  const z = equipSize(s.kind, u), sc = s.scale ?? 1
  return { w: z.w * sc, d: z.d * sc }
}

// ── Colours ────────────────────────────────────────────────────

export const COLOR_CHOICES: { color: string; name: string }[] = [
  { color: '#1b2a6b', name: 'Dunkelblau' }, { color: '#111827', name: 'Schwarz' }, { color: '#ef4444', name: 'Rot' },
  { color: '#f59e0b', name: 'Orange' }, { color: '#f5c542', name: 'Gelb' }, { color: '#49b05c', name: 'Grün' },
  { color: '#3b82e8', name: 'Blau' }, { color: '#e94fae', name: 'Pink' }, { color: '#8a8f9e', name: 'Grau' }, { color: '#ffffff', name: 'Weiß' },
]
export function isLight(hex: string): boolean {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex)
  if (!m) return false
  const n = parseInt(m[1], 16), r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  return 0.299 * r + 0.587 * g + 0.114 * b > 170
}
export const canColor = (s: SceneShape) => s.type !== 'ball'
export const canScale = (s: SceneShape) => s.type === 'player' || s.type === 'ball' || s.type === 'equip' || s.type === 'text'
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
  if (c.type === 'ball') delete c.carriedBy
  return c
}

/** Duplicate several shapes at once. Lines keep their attachment only to players that are duplicated with them. */
export function duplicateShapes(shapes: SceneShape[], ids: string[], dx: number, dy: number): { shapes: SceneShape[]; newIds: string[] } {
  const set = new Set(ids), map = new Map<string, string>()
  const copies: SceneShape[] = []
  for (const s of shapes) {
    if (!set.has(s.id)) continue
    const c = duplicateShape(s, dx, dy)
    map.set(s.id, c.id)
    copies.push(c)
  }
  const out = copies.map(c => {
    const orig = shapes.find(x => map.get(x.id) === c.id)
    if (orig && orig.type === 'ball' && c.type === 'ball' && orig.carriedBy && map.has(orig.carriedBy)) return { ...c, carriedBy: map.get(orig.carriedBy) }
    if (orig && orig.type === 'line' && c.type === 'line') {
      const r = { ...c } as typeof c
      if (orig.fromId && map.has(orig.fromId)) r.fromId = map.get(orig.fromId)
      if (orig.toId && map.has(orig.toId)) r.toId = map.get(orig.toId)
      return r
    }
    return c
  })
  return { shapes: [...shapes, ...out], newIds: out.map(c => c.id) }
}

/** Copies of loose shapes (from the clipboard) shifted by dx, dy, with fresh ids. Lines keep their attachment to players copied with them. */
export function cloneShapes(src: SceneShape[], dx: number, dy: number): SceneShape[] {
  return duplicateShapes(src, src.map(s => s.id), dx, dy).shapes.slice(src.length)
}

export function removeShapes(shapes: SceneShape[], ids: string[]): SceneShape[] {
  return ids.reduce((acc, id) => removeShape(acc, id), shapes)
}

/** Set (or with undefined, reset) the colour of the given shapes. The ball has no colour. */
export function colorShapes(shapes: SceneShape[], ids: string[], color: string | undefined): SceneShape[] {
  const set = new Set(ids)
  return shapes.map(s => {
    if (!set.has(s.id) || !canColor(s)) return s
    if (s.type === 'pen') return withPatch(s, { color: color ?? '#1b2a6b' } as Partial<SceneShape>)
    return withPatch(s, { color } as Partial<SceneShape>)
  })
}

/** Put a ring around the players among `ids` (or take it away). Other shapes are left alone. */
export function ringShapes(shapes: SceneShape[], ids: string[], on: boolean): SceneShape[] {
  const set = new Set(ids)
  return shapes.map(s => (s.type === 'player' && set.has(s.id) ? withPatch(s, { ring: on ? true : undefined } as Partial<SceneShape>) : s))
}

/** Are all players among `ids` ringed? (false when there are none) */
export function allRinged(shapes: SceneShape[], ids: string[]): boolean {
  const set = new Set(ids)
  const ps = shapes.filter(s => s.type === 'player' && set.has(s.id))
  return ps.length > 0 && ps.every(s => s.type === 'player' && !!s.ring)
}

/** Make symbols (and text) bigger or smaller by a factor. Lines, distances and fields are resized by their end handles instead. */
export function scaleShapes(shapes: SceneShape[], ids: string[], factor: number): SceneShape[] {
  const set = new Set(ids)
  return shapes.map(s => {
    if (!set.has(s.id)) return s
    if (s.type === 'player' || s.type === 'ball' || s.type === 'equip') return withPatch(s, { scale: clampScale((s.scale ?? 1) * factor) } as Partial<SceneShape>)
    if (s.type === 'text') return withPatch(s, { size: Math.min(3, Math.max(0.6, Math.round(s.size * factor * 10) / 10)) } as Partial<SceneShape>)
    return s
  })
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

/** Move several shapes together. Lines that are not selected but are attached to a moved player follow it; a selected line
 *  stays attached to the players that move with it and lets go of the others. */
export function moveShapes(shapes: SceneShape[], ids: string[], dx: number, dy: number): SceneShape[] {
  const set = new Set(ids)
  const movedPlayers = new Set(shapes.filter(s => s.type === 'player' && set.has(s.id)).map(s => s.id))
  return shapes.map(s => {
    if (set.has(s.id)) {
      const patch = movePatch(s, dx, dy) as Record<string, unknown>
      if (s.type === 'ball' && s.carriedBy && movedPlayers.has(s.carriedBy)) patch.carriedBy = s.carriedBy
      if (s.type === 'line') {
        if (s.fromId && movedPlayers.has(s.fromId)) patch.fromId = s.fromId
        if (s.toId && movedPlayers.has(s.toId)) patch.toId = s.toId
      }
      return withPatch(s, patch as Partial<SceneShape>)
    }
    if (s.type === 'ball' && s.carriedBy && movedPlayers.has(s.carriedBy)) return withPatch(s, { pos: sh(s.pos, dx, dy) } as Partial<SceneShape>)
    if (s.type === 'line') {
      const patch: Record<string, Pt> = {}
      if (s.fromId && movedPlayers.has(s.fromId)) patch.from = sh(s.from, dx, dy)
      if (s.toId && movedPlayers.has(s.toId)) patch.to = sh(s.to, dx, dy)
      if (Object.keys(patch).length) return withPatch(s, patch as unknown as Partial<SceneShape>)
    }
    return s
  })
}

/** Move a shape. Moving a player also carries the ends of every pass or run line attached to that player. */
export function moveShape(shapes: SceneShape[], id: string, dx: number, dy: number): SceneShape[] {
  return shapes.some(s => s.id === id) ? moveShapes(shapes, [id], dx, dy) : shapes
}

/** Remove a shape. Lines that were attached to a deleted player keep their points but lose the attachment. */
export function removeShape(shapes: SceneShape[], id: string): SceneShape[] {
  return shapes.filter(s => s.id !== id).map(s => {
    if (s.type === 'ball' && s.carriedBy === id) return withPatch(s, { carriedBy: undefined } as Partial<SceneShape>)
    if (s.type === 'line' && (s.fromId === id || s.toId === id)) {
      return withPatch(s, { fromId: s.fromId === id ? undefined : s.fromId, toId: s.toId === id ? undefined : s.toId } as Partial<SceneShape>)
    }
    return s
  })
}

/** After a ball was placed or dragged: it belongs to the player standing right at it, otherwise to nobody.
 *  A ball that moves together with its carrier keeps him. */
export function settleBalls(shapes: SceneShape[], ids: string[], bg: SceneBackground): SceneShape[] {
  const u = symbolUnit(bg), set = new Set(ids)
  return shapes.map(s => {
    if (s.type !== 'ball' || !set.has(s.id)) return s
    if (s.carriedBy && set.has(s.carriedBy)) return s
    let best: string | undefined, bd = u * 1.7 * 1 + 1e-9
    for (const q of shapes) {
      if (q.type !== 'player') continue
      const d = dist(q.pos, s.pos) - (q.scale ?? 1) * u * 0.3
      if (d <= bd) { best = q.id; bd = d }
    }
    return withPatch(s, { carriedBy: best } as Partial<SceneShape>)
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
      case 'player': hit = dist(s.pos, p) <= u * (s.scale ?? 1) + tol * 0.9; break
      case 'ball': hit = dist(s.pos, p) <= ballRadius(u) * (s.scale ?? 1) + tol; break
      case 'equip': { const z = equipDims(s, u); hit = inRotRect(p, s.pos, z.w, z.d, s.angle, tol * 0.6); break }
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
        const q = s.angle ? rotPt(p, gridCentre(s), -s.angle) : p
        const nearX = (Math.abs(q[0] - x0) <= tol || Math.abs(q[0] - x1) <= tol) && q[1] >= y0 - tol && q[1] <= y1 + tol
        const nearY = (Math.abs(q[1] - y0) <= tol || Math.abs(q[1] - y1) <= tol) && q[0] >= x0 - tol && q[0] <= x1 + tol
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
      const trim = (id?: string) => { const x = id ? shapes?.find(q => q.id === id && q.type === 'player') : undefined; return x ? u * 1.25 * scaleOf(x) : 0 }
      const t = trimPolyline(curvePoints(s.from, s.to, s.bend, 28), trim(s.fromId), trim(s.toId))
      return [{ key: 'from', pos: t[0] }, { key: 'to', pos: t[t.length - 1] }, { key: 'bend', pos: m }]
    }
    case 'dim': return [{ key: 'a', pos: s.a }, { key: 'b', pos: s.b }]
    case 'grid': {
      const [ca, cba, cb, cab] = gridCorners(s)
      return [{ key: 'a', pos: ca }, { key: 'b', pos: cb }, { key: 'ab', pos: cab }, { key: 'ba', pos: cba }, { key: 'rot', pos: gridRotHandle(s, bg).pos }]
    }
    case 'equip': return [{ key: 'rot', pos: rotHandle(s, bg).pos }, { key: 'size', pos: sizeHandle(s, bg) }]
    case 'player': case 'ball': return [{ key: 'size', pos: sizeHandle(s, bg) }]
    default: return []
  }
}

/** Rotation handle of a field: above the middle of its top edge, or below the bottom edge when that would leave the picture. */
export function gridRotHandle(g: Grid, bg: SceneBackground): { pos: Pt; flipped: boolean } {
  const u = symbolUnit(bg), c = gridCentre(g), { h } = gridSize(g), ang = g.angle ?? 0
  const v = viewBoxOf(bg), pad = u * 0.4
  const at = (dy: number): Pt => rotPt([c[0], c[1] + dy], c, ang)
  const inside = (q: Pt) => q[0] >= v.x + pad && q[0] <= v.x + v.w - pad && q[1] >= v.y + pad && q[1] <= v.y + v.h - pad
  const front = at(-(h / 2 + u * 1.3))
  if (inside(front)) return { pos: front, flipped: false }
  const back = at(h / 2 + u * 1.3)
  return inside(back) ? { pos: back, flipped: true } : { pos: front, flipped: false }
}

/** Extent (half-diagonal, metres) of a symbol at scale 1, used to place and read the size handle. */
function baseExtent(s: Extract<SceneShape, { type: 'player' | 'ball' | 'equip' }>, u: number): number {
  if (s.type === 'player') return playerRadius(u)
  if (s.type === 'ball') return ballRadius(u)
  const z = equipSize(s.kind, u)
  return Math.hypot(z.w, z.d) / 2
}
/** The size handle sits diagonally below-right of the symbol; dragging it further out makes the symbol bigger. */
export function sizeHandle(s: Extract<SceneShape, { type: 'player' | 'ball' | 'equip' }>, bg: SceneBackground): Pt {
  const u = symbolUnit(bg), r = baseExtent(s, u) * (s.scale ?? 1) + u * 0.5
  return [s.pos[0] + r * Math.SQRT1_2, s.pos[1] + r * Math.SQRT1_2]
}

/** Where the rotation handle of a piece of equipment sits. It normally hangs off the front (the top at angle 0);
 *  near the edge of the picture it swings to the other side so it can always be grabbed. */
export function rotHandle(s: Extract<SceneShape, { type: 'equip' }>, bg: SceneBackground): { pos: Pt; flipped: boolean } {
  const u = symbolUnit(bg), z = equipDims(s, u), r = Math.max(z.w, z.d) / 2 + u * 1.3
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
    if (key === 'rot') {
      const c = gridCentre(s)
      let deg = (Math.atan2(p[0] - c[0], -(p[1] - c[1])) * 180) / Math.PI
      if (gridRotHandle(s, bg).flipped) deg -= 180
      deg = ((Math.round(deg / 5) * 5) % 360 + 360) % 360
      return { angle: deg || undefined } as Partial<SceneShape>
    }
    // dragging a corner: the opposite corner stays where it is in the picture, whatever the angle
    const ang = s.angle ?? 0, c0 = gridCentre(s)
    const world: Record<string, Pt> = { a: s.a, ba: [s.b[0], s.a[1]], b: s.b, ab: [s.a[0], s.b[1]] }
    const opposite: Record<string, string> = { a: 'b', b: 'a', ab: 'ba', ba: 'ab' }
    if (!(key in world)) return null
    const F = ang ? rotPt(world[opposite[key]], c0, ang) : world[opposite[key]]
    const Q = q
    const c1: Pt = [(Q[0] + F[0]) / 2, (Q[1] + F[1]) / 2]
    const Ql = ang ? rotPt(Q, c1, -ang) : Q, Fl = ang ? rotPt(F, c1, -ang) : F
    if (key === 'a') return { a: Ql, b: Fl } as Partial<SceneShape>
    if (key === 'b') return { a: Fl, b: Ql } as Partial<SceneShape>
    if (key === 'ab') return { a: [Ql[0], Fl[1]], b: [Fl[0], Ql[1]] } as Partial<SceneShape>
    return { a: [Fl[0], Ql[1]], b: [Ql[0], Fl[1]] } as Partial<SceneShape>
  }
  if ((s.type === 'player' || s.type === 'ball' || s.type === 'equip') && key === 'size') {
    const base = baseExtent(s, u)
    const sc = clampScale(Math.round(((dist(s.pos, p) - u * 0.5) / base) * 20) / 20)
    return { scale: sc } as Partial<SceneShape>
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

// ── Boxes and rectangle selection ──────────────────────────────

export interface Box { x0: number; y0: number; x1: number; y1: number }

/** The smallest box around a shape (metres). */
export function shapeBox(s: SceneShape, bg: SceneBackground): Box {
  const u = symbolUnit(bg)
  const around = (c: Pt, r: number): Box => ({ x0: c[0] - r, y0: c[1] - r, x1: c[0] + r, y1: c[1] + r })
  const ofPts = (pts: Pt[]): Box => ({ x0: Math.min(...pts.map(q => q[0])), y0: Math.min(...pts.map(q => q[1])), x1: Math.max(...pts.map(q => q[0])), y1: Math.max(...pts.map(q => q[1])) })
  switch (s.type) {
    case 'player': return around(s.pos, playerRadius(u, s.scale))
    case 'ball': return around(s.pos, ballRadius(u) * (s.scale ?? 1))
    case 'equip': {
      const z = equipDims(s, u), a = (s.angle * Math.PI) / 180, c = Math.abs(Math.cos(a)), n = Math.abs(Math.sin(a))
      return { x0: s.pos[0] - (z.w * c + z.d * n) / 2, x1: s.pos[0] + (z.w * c + z.d * n) / 2, y0: s.pos[1] - (z.w * n + z.d * c) / 2, y1: s.pos[1] + (z.w * n + z.d * c) / 2 }
    }
    case 'line': return ofPts(curvePoints(s.from, s.to, s.bend, 16))
    case 'dim': return ofPts([s.a, s.b])
    case 'grid': return ofPts(gridCorners(s))
    case 'pen': return ofPts(s.points)
    case 'text': { const h = textHeight(s.size, u), w = Math.max(1, s.text.length) * h * 0.55; return { x0: s.pos[0] - w / 2, x1: s.pos[0] + w / 2, y0: s.pos[1] - h / 2, y1: s.pos[1] + h / 2 } }
  }
}

export function unionBox(boxes: Box[]): Box | null {
  if (!boxes.length) return null
  return { x0: Math.min(...boxes.map(b => b.x0)), y0: Math.min(...boxes.map(b => b.y0)), x1: Math.max(...boxes.map(b => b.x1)), y1: Math.max(...boxes.map(b => b.y1)) }
}

/** Ids of everything that lies completely inside the rectangle spanned by a and b. */
export function shapesInRect(shapes: SceneShape[], a: Pt, b: Pt, bg: SceneBackground): string[] {
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1])
  return shapes.filter(s => { const q = shapeBox(s, bg); return q.x0 >= x0 && q.x1 <= x1 && q.y0 >= y0 && q.y1 <= y1 }).map(s => s.id)
}

// ── Undo and redo ──────────────────────────────────────────────

export interface Hist<T> { past: T[]; present: T; future: T[] }
export const HIST_LIMIT = 100
export const hist = <T,>(present: T): Hist<T> => ({ past: [], present, future: [] })
export function hPush<T>(h: Hist<T>, next: T): Hist<T> { return { past: [...h.past, h.present].slice(-HIST_LIMIT), present: next, future: [] } }
export function hUndo<T>(h: Hist<T>): Hist<T> { return h.past.length ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] } : h }
export function hRedo<T>(h: Hist<T>): Hist<T> { return h.future.length ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) } : h }
