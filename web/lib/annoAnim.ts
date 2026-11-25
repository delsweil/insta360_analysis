// lib/annoAnim.ts
// Pure animation logic for coach drawings — no React, no DOM, so it can be unit-tested (see annoAnim.test.mts).
//
// A drawing's shapes live on an *annotation clock* τ (seconds after the drawing's anchor moment).
// Every shape may carry optional animation fields; a shape with none of them is a static shape that
// is visible for the whole drawing, so everything saved before this existed renders exactly as before.
//
//   t0   τ at which the shape appears (default 0 = from the start; only a t0 > 0 fades in)
//   t1   τ at which the shape disappears (default: never)
//   kf   keyframes: { t, v } where v holds only the properties that changed at τ = t
//   draw curves only: the line draws itself on from its start to its arrowhead
//
// The shape's own geometry fields are its state at τ = t0 (the "base"); keyframes move it from there.
import type { Shape, NPoint } from '@/components/AnnotationCanvas'

export type Keyframe = { t: number; v: Record<string, unknown> }
export type Animated = { t0?: number; t1?: number; kf?: Keyframe[]; draw?: boolean }

export const FADE_S = 0.25      // fade in / out
export const DRAW_S = 0.6       // a curve drawing itself on
export const KEY_EPS = 0.05     // an edit within this many seconds of a keyframe edits that keyframe

/** Properties that can change over time, per shape type. Everything else (text, line style, …) is the same at every moment. */
export const ANIMATABLE: Record<Shape['type'], readonly string[]> = {
  zone: ['points', 'color', 'opacity'],
  curve: ['from', 'control', 'to', 'color', 'opacity'],
  label: ['pos', 'color'],
  highlight: ['pos', 'radiusX', 'radiusY', 'color', 'opacity'],
  number: ['pos', 'color'],
  cone: ['pos', 'angle', 'length', 'width', 'color', 'opacity'],
  connector: ['color', 'opacity'],
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x)
const round2 = (x: number) => Math.round(x * 100) / 100

// ---- value interpolation ------------------------------------------------------------------------

const HEX = /^#[0-9a-f]{6}$/i
function mixHex(a: string, b: string, u: number): string {
  const ch = (s: string, i: number) => parseInt(s.slice(1 + 2 * i, 3 + 2 * i), 16)
  let out = '#'
  for (let i = 0; i < 3; i++) out += Math.round(ch(a, i) + (ch(b, i) - ch(a, i)) * u).toString(16).padStart(2, '0')
  return out
}

function shortestAngle(a: number, b: number) {
  let d = b - a
  while (d > Math.PI) d -= 2 * Math.PI
  while (d < -Math.PI) d += 2 * Math.PI
  return d
}

type Val = unknown
function linear(a: Val, b: Val, u: number, key?: string): Val {
  if (typeof a === 'number' && typeof b === 'number') return key === 'angle' ? a + shortestAngle(a, b) * u : a + (b - a) * u
  if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) return a.map((x, i) => linear(x, b[i], u))
  if (typeof a === 'string' && typeof b === 'string' && HEX.test(a) && HEX.test(b)) return mixHex(a, b, u)
  return u < 1 ? a : b
}

/** Hermite step between keyframes i and i+1 with finite-difference tangents: smooth through ≥3 keyframes (a runner's path), plain chord otherwise. */
function hermite(pts: { t: number; val: Val }[], i: number, u: number, key: string): Val {
  const p0 = pts[i], p1 = pts[i + 1]
  const num = (v: Val): number | null => (typeof v === 'number' ? v : null)
  const a = num(p0.val), b = num(p1.val)
  if (a === null || b === null) {
    if (Array.isArray(p0.val) && Array.isArray(p1.val) && p0.val.length === p1.val.length) {
      return p0.val.map((_, k) => hermite(
        pts.map(p => ({ t: p.t, val: (p.val as Val[])[k] })), i, u, key))
    }
    return linear(p0.val, p1.val, u, key)
  }
  const h = p1.t - p0.t
  const slope = (j: number, k: number) => {
    const x = num(pts[j].val), y = num(pts[k].val)
    return x === null || y === null ? (b - a) / h : (y - x) / (pts[k].t - pts[j].t)
  }
  const m0 = i > 0 ? slope(i - 1, i + 1) : (b - a) / h
  const m1 = i + 2 < pts.length ? slope(i, i + 2) : (b - a) / h
  const u2 = u * u, u3 = u2 * u
  return (2 * u3 - 3 * u2 + 1) * a + (u3 - 2 * u2 + u) * h * m0 + (-2 * u3 + 3 * u2) * b + (u3 - u2) * h * m1
}

type Track = { t: number; val: Val }[]

/** The values one property takes over time: its base value at t0, then each keyframe that mentions it. */
function trackOf(s: Shape, prop: string): Track {
  const a = s as unknown as Animated & Record<string, Val>
  const tb = a.t0 ?? 0
  const pts: Track = [{ t: tb, val: a[prop] }]
  for (const k of a.kf ?? []) if (prop in k.v && k.t > tb + 1e-6) pts.push({ t: k.t, val: k.v[prop] })
  return pts
}

function sampleTrack(pts: Track, tau: number, key: string): Val {
  if (pts.length === 1 || tau <= pts[0].t) return pts[0].val
  for (let i = 0; i < pts.length - 1; i++) {
    if (tau <= pts[i + 1].t) {
      const u = (tau - pts[i].t) / (pts[i + 1].t - pts[i].t)
      if (pts.length === 2) return linear(pts[0].val, pts[1].val, u * u * (3 - 2 * u), key)   // one move: ease in and out
      return hermite(pts, i, u, key)
    }
  }
  return pts[pts.length - 1].val
}

// ---- reading a shape at a moment ---------------------------------------------------------------

export function isAnimatedShape(s: Shape): boolean {
  const a = s as unknown as Animated
  return !!(a.kf && a.kf.length) || (a.t0 ?? 0) > 0 || a.t1 != null || !!a.draw
}
export function isAnimated(shapes: Shape[]): boolean { return shapes.some(isAnimatedShape) }

/** 0 → invisible, 1 → fully shown. A shape with no t0 / t1 is simply always 1. */
export function alphaAt(s: Shape, tau: number): number {
  const a = s as unknown as Animated
  let v = 1
  const t0 = a.t0 ?? 0
  if (t0 > 0) v *= tau >= t0 + FADE_S ? 1 : clamp01((tau - t0) / FADE_S)
  if (a.t1 != null) v *= tau >= a.t1 ? 0 : tau <= a.t1 - FADE_S ? 1 : clamp01((a.t1 - tau) / FADE_S)
  return v
}

export type Resolved = { s: Shape; a: number; p: number }

/** The shape as it looks at τ (its interpolated geometry and colour), how visible it is, and how far a curve has drawn itself (0..1). */
export function resolveShape(s: Shape, tau: number): Resolved | null {
  const a = alphaAt(s, tau)
  if (a <= 0) return null
  const an = s as unknown as Animated
  let out: Shape = s
  if (an.kf && an.kf.length) {
    const patch: Record<string, Val> = {}
    for (const prop of ANIMATABLE[s.type]) {
      if (an.kf.some(k => prop in k.v)) patch[prop] = sampleTrack(trackOf(s, prop), tau, prop)
    }
    out = { ...s, ...patch } as Shape
  }
  let p = 1
  if (an.draw && s.type === 'curve') p = clamp01((tau - (an.t0 ?? 0)) / DRAW_S)
  return { s: out, a, p }
}

/** Resolve a whole drawing; connectors need their two highlights' *current* positions, so callers get the resolved list. */
export function resolveAll(shapes: Shape[], tau: number): Resolved[] {
  const out: Resolved[] = []
  for (const s of shapes) { const r = resolveShape(s, tau); if (r) out.push(r) }
  return out
}

// ---- editing: auto-keyframe --------------------------------------------------------------------

export function baseTime(s: Shape): number { return (s as unknown as Animated).t0 ?? 0 }

function sortedKf(kf: Keyframe[]) { return [...kf].sort((x, y) => x.t - y.t) }

/**
 * Apply an edit made while the playhead is at τ.
 *  - Non-animatable fields (text, dash, …) and the animation fields themselves are always written to the shape.
 *  - Animatable fields are written to the shape's base state when τ is at its start (or auto-key is off and there is no keyframe at τ);
 *    otherwise they become a keyframe at τ: a new one, or the existing one within KEY_EPS of τ.
 */
export function applyEdit(s: Shape, patch: Partial<Shape> & Record<string, unknown>, tau: number, autoKey: boolean): Shape {
  const anim = new Set(ANIMATABLE[s.type])
  const base: Record<string, Val> = {}
  const timed: Record<string, Val> = {}
  for (const [k, v] of Object.entries(patch)) (anim.has(k) ? timed : base)[k] = v
  const an = s as unknown as Animated
  const tb = baseTime(s)
  const existing = (an.kf ?? []).find(k => Math.abs(k.t - tau) <= KEY_EPS)
  const atBase = tau <= tb + KEY_EPS
  const asKeyframe = Object.keys(timed).length > 0 && !atBase && (autoKey || !!existing)
  if (!asKeyframe) return { ...s, ...base, ...timed } as Shape
  const kf = sortedKf(existing
    ? (an.kf ?? []).map(k => (k === existing ? { t: k.t, v: { ...k.v, ...timed } } : k))
    : [...(an.kf ?? []), { t: round2(tau), v: timed }])
  return { ...s, ...base, kf } as Shape
}

/** Keyframes that exist at (or near) τ, for showing a filled diamond on the timeline. */
export function keyframeAt(s: Shape, tau: number): Keyframe | undefined {
  return ((s as unknown as Animated).kf ?? []).find(k => Math.abs(k.t - tau) <= KEY_EPS)
}

export function removeKeyframe(s: Shape, t: number): Shape {
  const an = s as unknown as Animated
  const kf = (an.kf ?? []).filter(k => Math.abs(k.t - t) > 1e-6)
  const { kf: _drop, ...rest } = s as unknown as Record<string, unknown>
  void _drop
  return (kf.length ? { ...rest, kf } : rest) as unknown as Shape
}

export function moveKeyframe(s: Shape, from: number, to: number): Shape {
  const an = s as unknown as Animated
  const t = Math.max(baseTime(s) + 0.1, round2(to))
  return { ...s, kf: sortedKf((an.kf ?? []).map(k => (Math.abs(k.t - from) < 1e-6 ? { ...k, t } : k))) } as Shape
}

/** Appear / disappear times; t1 = null removes the limit. Keyframes that would end up before the new appear time move with it. */
export function setWindow(s: Shape, t0: number | undefined, t1: number | null | undefined): Shape {
  const an = s as unknown as Animated & Record<string, unknown>
  const out: Record<string, unknown> = { ...an }
  if (t0 !== undefined) { if (t0 <= 0) delete out.t0; else out.t0 = round2(t0) }
  if (t1 !== undefined) { if (t1 === null) delete out.t1; else out.t1 = round2(Math.max(t1, (out.t0 as number | undefined ?? 0) + 0.2)) }
  const tb = (out.t0 as number | undefined) ?? 0
  if (Array.isArray(out.kf)) out.kf = (out.kf as Keyframe[]).map(k => (k.t <= tb ? { ...k, t: round2(tb + 0.1) } : k))
  return out as unknown as Shape
}

/** Slide a shape's whole timing (appear, disappear, keyframes) later or earlier by `d` seconds. */
export function shiftShape(s: Shape, d: number): Shape {
  const an = s as unknown as Animated & Record<string, unknown>
  const out: Record<string, unknown> = { ...an }
  const t0 = Math.max(0, round2((an.t0 ?? 0) + d))
  const applied = t0 - (an.t0 ?? 0)
  if (t0 > 0) out.t0 = t0; else delete out.t0
  if (an.t1 != null) out.t1 = round2(an.t1 + applied)
  if (an.kf) out.kf = an.kf.map(k => ({ ...k, t: round2(k.t + applied) }))
  return out as unknown as Shape
}

/** Pin the shape's current look as a keyframe at τ (so the next edit moves it *from here*). Returns the shape unchanged at its start. */
export function addKeyframeAt(s: Shape, tau: number): Shape {
  if (tau <= baseTime(s) + KEY_EPS) return s
  const cur = resolveShape(s, tau)?.s as unknown as Record<string, Val> | undefined
  if (!cur) return s
  const geometry = ANIMATABLE[s.type].filter(k => k !== 'color' && k !== 'opacity' && k in cur)
  if (!geometry.length) return s
  const v: Record<string, Val> = {}
  for (const k of geometry) v[k] = JSON.parse(JSON.stringify(cur[k]))
  const an = s as unknown as Animated
  const existing = (an.kf ?? []).find(k => Math.abs(k.t - tau) <= KEY_EPS)
  if (existing) return s
  return { ...s, kf: sortedKf([...(an.kf ?? []), { t: round2(tau), v }]) } as Shape
}

// ---- whole-drawing helpers ---------------------------------------------------------------------

/** When everything has finished appearing and moving (seconds on the drawing's clock). */
export function spanEnd(shapes: Shape[]): number {
  let end = 0
  for (const s of shapes) {
    const an = s as unknown as Animated
    const t0 = an.t0 ?? 0
    end = Math.max(end, t0 > 0 ? t0 + FADE_S : 0, an.draw ? t0 + DRAW_S : 0)
    for (const k of an.kf ?? []) end = Math.max(end, k.t)
    if (an.t1 != null) end = Math.max(end, an.t1)
  }
  return round2(end)
}

/** One click: the shapes appear one after another in the order they were drawn (curves draw themselves on; connectors wait for both ends). */
export function staggerReveal(shapes: Shape[], step = 0.7, start = 0.3): Shape[] {
  const out: Shape[] = []
  let t = start
  const when = new Map<string, number>()
  for (const s of shapes) {
    let at = t
    if (s.type === 'connector') {
      at = Math.max(when.get(s.fromId) ?? 0, when.get(s.toId) ?? 0) + 0.15
    } else t += step
    when.set(s.id, at)
    const next = { ...(s as unknown as Record<string, unknown>), t0: round2(at) } as Record<string, unknown>
    if (s.type === 'curve') next.draw = true
    out.push(next as unknown as Shape)
  }
  return out
}

/** Back to a static drawing: everything visible from the start, no motion. Keyframes are kept unless `keepMotion` is false. */
export function clearReveal(shapes: Shape[]): Shape[] {
  return shapes.map(s => {
    const { t0: _a, t1: _b, draw: _c, ...rest } = s as unknown as Record<string, unknown>
    void _a; void _b; void _c
    return rest as unknown as Shape
  })
}

let dupSeq = 0
export function duplicateShape(s: Shape, dx = 0.02, dy = 0.03): Shape {
  const id = Math.random().toString(36).slice(2, 10) + (dupSeq++ % 10)
  const mv = (p: NPoint): NPoint => [p[0] + dx, p[1] + dy]
  const c = JSON.parse(JSON.stringify(s)) as Record<string, unknown>
  c.id = id
  if (c.pos) c.pos = mv(c.pos as NPoint)
  if (c.from) c.from = mv(c.from as NPoint)
  if (c.to) c.to = mv(c.to as NPoint)
  if (c.control) c.control = mv(c.control as NPoint)
  if (c.points) c.points = (c.points as NPoint[]).map(mv)
  delete c.kf                          // a copy starts as a still shape; keyframes belonged to the original's path
  return c as unknown as Shape
}

// ---- player marker size, learned from the coach's own corrections --------------------------------

/** A marker as the coach left it: centre height y (0 = top of the picture, 1 = bottom) and its radii (rx by width, ry by height). */
export type MarkerSample = { y: number; rx: number; ry: number }

const FAR_RX = 0.016, NEAR_RX = 0.042
const priorRx = (y: number) => FAR_RX + (NEAR_RX - FAR_RX) * clamp01(y)
/** A ground circle looks flatter the further away it is (the camera sees it at a shallower angle). */
const priorFlat = (y: number) => 0.5 + 0.4 * clamp01(y)

/** Radii for a new marker at height y, from the most recent corrections (oldest first). `aspect` = picture width / height. */
export function markerSize(samples: MarkerSample[], y: number, aspect: number): { radiusX: number; radiusY: number } {
  y = clamp01(y)
  const n = samples.length
  let rx = priorRx(y)
  let flat = priorFlat(y)
  if (n > 0) {
    const w = samples.map((_, i) => Math.pow(0.75, n - 1 - i))           // newer corrections count more
    const sw = w.reduce((a, b) => a + b, 0)
    const k = Math.exp(samples.reduce((a, s, i) => a + w[i] * Math.log(Math.max(1e-4, s.rx) / priorRx(s.y)), 0) / sw)
    rx = priorRx(y) * k
    const yMin = Math.min(...samples.map(s => s.y)), yMax = Math.max(...samples.map(s => s.y))
    if (n >= 3 && yMax - yMin >= 0.2) {
      // enough spread to learn the slope too: weighted least squares of rx on y, blended in as evidence builds
      const my = samples.reduce((a, s, i) => a + w[i] * s.y, 0) / sw
      const mr = samples.reduce((a, s, i) => a + w[i] * s.rx, 0) / sw
      let sxy = 0, sxx = 0
      samples.forEach((s, i) => { sxy += w[i] * (s.y - my) * (s.rx - mr); sxx += w[i] * (s.y - my) ** 2 })
      if (sxx > 1e-9) {
        const ls = mr + (sxy / sxx) * (y - my)
        const lam = Math.min(1, (n - 2) / 4)
        rx = lam * ls + (1 - lam) * rx
      }
    }
    const fl = samples.map(s => (s.ry / Math.max(1e-4, s.rx)) / aspect / priorFlat(s.y))
    flat = priorFlat(y) * Math.min(1.6, Math.max(0.5, Math.exp(fl.reduce((a, v, i) => a + w[i] * Math.log(Math.max(0.05, v)), 0) / sw)))
  }
  rx = Math.min(0.09, Math.max(0.008, rx))
  return { radiusX: rx, radiusY: Math.max(0.008, rx * aspect * Math.min(1.4, Math.max(0.25, flat))) }
}

/** The point a shape is scaled around when pinched: its anchor, so it grows in place instead of drifting. */
export function pivotOf(s: Shape): NPoint {
  switch (s.type) {
    case 'zone': {
      const n = Math.max(1, s.points.length)
      return [s.points.reduce((a, p) => a + p[0], 0) / n, s.points.reduce((a, p) => a + p[1], 0) / n]
    }
    case 'curve': return [(s.from[0] + s.to[0]) / 2, (s.from[1] + s.to[1]) / 2]
    case 'connector': return [0, 0]
    default: return s.pos
  }
}

export const MIN_PINCH = 0.01
/**
 * What a two-finger pinch does to a shape: scale it by `f` about its own anchor and slide it by (dx, dy), all in 0..1 units.
 * Returns the patch to apply (null when the shape cannot be pinched, e.g. text or a connector, which follows its markers).
 * Markers keep their ellipse shape; a body shape grows in length; zones and lines scale their points.
 */
export function pinchPatch(s: Shape, f: number, dx = 0, dy = 0): Partial<Shape> | null {
  if (!isFinite(f) || f <= 0) return null
  const pv = pivotOf(s)
  const sc = (p: NPoint): NPoint => [pv[0] + (p[0] - pv[0]) * f + dx, pv[1] + (p[1] - pv[1]) * f + dy]
  const mv = (p: NPoint): NPoint => [p[0] + dx, p[1] + dy]
  switch (s.type) {
    case 'highlight':
      return {
        pos: mv(s.pos),
        radiusX: Math.max(MIN_PINCH, Math.min(0.12, s.radiusX * f)),
        radiusY: Math.max(MIN_PINCH, Math.min(0.2, s.radiusY * f)),
      } as Partial<Shape>
    case 'cone': return { pos: mv(s.pos), length: Math.max(0.03, Math.min(0.9, s.length * f)) } as Partial<Shape>
    case 'zone': return { points: s.points.map(sc) } as Partial<Shape>
    case 'curve': return { from: sc(s.from), control: sc(s.control), to: sc(s.to) } as Partial<Shape>
    case 'label': case 'number': return { pos: mv(s.pos) } as Partial<Shape>
    default: return null
  }
}
