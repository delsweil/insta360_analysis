// Animation of a form's sketches. The coach draws the scenes; the animation plays them one after the other and moves everything in between.
// Shapes are matched by id (copying a scene keeps ids, so "the same player" is the same id). Nothing here guesses tactics:
// positions come from the sketches, and the way a player or the ball travels comes from the lines the coach drew
// (a run line bends the player's path, a pass line bends the ball's).

import { curvePoints, dist, type Pt } from './sketch.ts'
import { symbolUnit, type Scene, type SceneShape } from './scene.ts'

export interface Timing { move_s?: number; hold_s?: number }
export const DEFAULT_MOVE_S = 1.6
export const DEFAULT_HOLD_S = 1.2
export const MIN_S = 0.2
export const MAX_S = 30

export const clampSec = (v: number, fallback: number) => (Number.isFinite(v) ? Math.round(Math.min(MAX_S, Math.max(MIN_S, v)) * 10) / 10 : fallback)
export const moveOf = (t?: Timing | null) => clampSec(t?.move_s ?? DEFAULT_MOVE_S, DEFAULT_MOVE_S)
export const holdOf = (t?: Timing | null) => clampSec(t?.hold_s ?? DEFAULT_HOLD_S, DEFAULT_HOLD_S)

export interface Step { scene: Scene; timing?: Timing | null }
export interface Seg { index: number; kind: 'hold' | 'move'; t0: number; t1: number }

/** The time line: scene 0 is held; every further scene is moved into and then held. `starts[i]` is when scene i is fully reached. */
export function plan(steps: Step[]): { segs: Seg[]; total: number; starts: number[] } {
  const segs: Seg[] = [], starts: number[] = []
  let t = 0
  steps.forEach((st, i) => {
    if (i > 0) { const d = moveOf(st.timing); segs.push({ index: i, kind: 'move', t0: t, t1: t + d }); t += d }
    starts.push(t)
    const h = holdOf(st.timing); segs.push({ index: i, kind: 'hold', t0: t, t1: t + h }); t += h
  })
  return { segs, total: t, starts }
}

export const ease = (p: number) => { const x = Math.min(1, Math.max(0, p)); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2 }
const lerp = (a: number, b: number, p: number) => a + (b - a) * p
const lerpPt = (a: Pt, b: Pt, p: number): Pt => [lerp(a[0], b[0], p), lerp(a[1], b[1], p)]
/** Angles take the short way round. */
export function lerpAngle(a: number, b: number, p: number): number {
  const d = ((((b - a) % 360) + 540) % 360) - 180
  return ((a + d * p) % 360 + 360) % 360
}
const clamp01 = (x: number) => Math.min(1, Math.max(0, x))

/** The point at fraction f (0..1) of the way along a polyline, by length. */
export function along(pts: Pt[], f: number): Pt {
  if (pts.length === 1) return pts[0]
  const lens: number[] = [0]
  for (let i = 1; i < pts.length; i++) lens.push(lens[i - 1] + dist(pts[i - 1], pts[i]))
  const total = lens[lens.length - 1]
  if (total === 0) return pts[0]
  const want = clamp01(f) * total
  let i = 1
  while (i < lens.length - 1 && lens[i] < want) i++
  const seg = lens[i] - lens[i - 1] || 1, r = (want - lens[i - 1]) / seg
  return lerpPt(pts[i - 1], pts[i], r)
}

type PosShape = Extract<SceneShape, { type: 'player' | 'ball' | 'equip' | 'text' }>

/** The bend of the line the coach drew for this movement: a run line for a player, a pass or return line for the ball. 0 when there is none. */
export function pathBend(a: Scene, s: PosShape, from: Pt, to: Pt): number {
  const u = symbolUnit(a.background)
  const kinds = s.type === 'ball' ? ['pass', 'return'] : s.type === 'player' ? ['move'] : []
  if (!kinds.length) return 0
  let best = 0, bd = Infinity
  for (const l of a.shapes) {
    if (l.type !== 'line' || !kinds.includes(l.kind) || l.bend === 0) continue
    if (dist(l.from, from) > u * 2) continue
    const d = dist(l.to, to)
    if (d < bd) { bd = d; best = l.bend }
  }
  return best
}

const POS_TYPES = ['player', 'ball', 'equip', 'text']

function tweenShape(a: SceneShape, b: SceneShape, p: number, e: number, scene: Scene): SceneShape {
  const base = (p < 0.5 ? a : b) as Record<string, unknown>
  const out = { ...base } as Record<string, unknown>
  delete out.alpha
  if (POS_TYPES.includes(a.type) && b.type === a.type) {
    const sa = a as PosShape, sb = b as PosShape
    const bend = dist(sa.pos, sb.pos) > 1e-9 ? pathBend(scene, sa, sa.pos, sb.pos) : 0
    out.pos = bend ? along(curvePoints(sa.pos, sb.pos, bend, 48), e) : lerpPt(sa.pos, sb.pos, e)
    if (a.type === 'equip' && b.type === 'equip') { out.angle = lerpAngle(a.angle, b.angle, e); out.scale = lerp(a.scale ?? 1, b.scale ?? 1, e) }
    if ((a.type === 'player' && b.type === 'player') || (a.type === 'ball' && b.type === 'ball')) out.scale = lerp(a.scale ?? 1, b.scale ?? 1, e)
    if (a.type === 'text' && b.type === 'text') out.size = lerp(a.size, b.size, e)
  } else if (a.type === 'line' && b.type === 'line') {
    out.from = lerpPt(a.from, b.from, e); out.to = lerpPt(a.to, b.to, e); out.bend = lerp(a.bend, b.bend, e)
  } else if ((a.type === 'dim' && b.type === 'dim') || (a.type === 'grid' && b.type === 'grid')) {
    out.a = lerpPt(a.a, b.a, e); out.b = lerpPt(a.b, b.b, e)
    if (a.type === 'grid' && b.type === 'grid') out.angle = lerpAngle(a.angle ?? 0, b.angle ?? 0, e) || undefined
  }
  return out as unknown as SceneShape
}

/** What is on screen while moving from scene A to scene B, p = 0..1 (linear time; easing is applied here). */
export function tween(A: Scene, B: Scene, p: number): Scene {
  const e = ease(p)
  const inB = new Map(B.shapes.map(s => [s.id, s]))
  const inA = new Map(A.shapes.map(s => [s.id, s]))
  const out: SceneShape[] = []
  for (const a of A.shapes) {
    const b = inB.get(a.id)
    if (b && b.type === a.type) out.push(tweenShape(a, b, p, e, A))
    else out.push({ ...a, alpha: a.type === 'line' ? clamp01(1 - p / 0.4) : 1 - e })
  }
  for (const b of B.shapes) {
    const a = inA.get(b.id)
    if (a && a.type === b.type) continue
    out.push({ ...b, alpha: b.type === 'line' ? clamp01((p - 0.6) / 0.4) : e })
  }
  // a ball at a player's feet in both sketches stays with him on his way
  const posIn = (sc: Scene, id: string) => { const q = sc.shapes.find(x => x.id === id); return q && q.type === 'player' ? q.pos : null }
  const carried = out.map(s => {
    if (s.type !== 'ball') return s
    const a = inA.get(s.id), b = inB.get(s.id)
    if (!a || !b || a.type !== 'ball' || b.type !== 'ball' || !a.carriedBy || a.carriedBy !== b.carriedBy) return s
    const pa = posIn(A, a.carriedBy), pb = posIn(B, b.carriedBy)
    const now = out.find(x => x.id === a.carriedBy)
    if (!pa || !pb || !now || now.type !== 'player') return s
    const off = lerpPt([a.pos[0] - pa[0], a.pos[1] - pa[1]], [b.pos[0] - pb[0], b.pos[1] - pb[1]], e)
    return { ...s, pos: [now.pos[0] + off[0], now.pos[1] + off[1]] as Pt }
  })
  out.splice(0, out.length, ...carried)
  // lines attached to players follow them while they move
  const players = new Map(out.filter(s => s.type === 'player').map(s => [s.id, (s as Extract<SceneShape, { type: 'player' }>).pos]))
  const lines = out.map(s => {
    if (s.type !== 'line' || (!s.fromId && !s.toId)) return s
    const from = s.fromId && players.has(s.fromId) ? players.get(s.fromId)! : s.from
    const to = s.toId && players.has(s.toId) ? players.get(s.toId)! : s.to
    return { ...s, from, to }
  })
  return { background: p < 0.5 ? A.background : B.background, shapes: lines, caption: p < 0.5 ? A.caption : B.caption }
}

export interface Frame { scene: Scene; index: number; kind: 'hold' | 'move' }

/** The picture at time t (seconds). */
export function frameAt(steps: Step[], t: number): Frame {
  const pl = plan(steps)
  const tt = Math.min(Math.max(0, t), pl.total)
  const seg = pl.segs.find(s => tt >= s.t0 && tt < s.t1) ?? pl.segs[pl.segs.length - 1]
  if (!seg) return { scene: { background: { type: 'blank', width_m: 40, height_m: 30 }, shapes: [], caption: '' }, index: 0, kind: 'hold' }
  if (seg.kind === 'hold') return { scene: steps[seg.index].scene, index: seg.index, kind: 'hold' }
  const p = (tt - seg.t0) / (seg.t1 - seg.t0)
  return { scene: tween(steps[seg.index - 1].scene, steps[seg.index].scene, p), index: p < 0.5 ? seg.index - 1 : seg.index, kind: 'move' }
}

export function formatTime(s: number): string {
  const t = Math.max(0, Math.round(s)), m = Math.floor(t / 60)
  return `${m}:${String(t % 60).padStart(2, '0')}`
}
