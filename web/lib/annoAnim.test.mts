// Run: node --test lib/annoAnim.test.mts   (Node 22+ strips the types itself)
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  resolveShape, applyEdit, alphaAt, spanEnd, staggerReveal, isAnimated, removeKeyframe, setWindow,
  markerSize, keyframeAt, duplicateShape, clearReveal, shiftShape, addKeyframeAt,
} from './annoAnim.ts'

/* eslint-disable @typescript-eslint/no-explicit-any */
const hl = (over: any = {}): any => ({
  id: 'h1', type: 'highlight', pos: [0.2, 0.5], radiusX: 0.03, radiusY: 0.05,
  color: '#facc15', opacity: 0.85, dash: 'solid', style: 'circle', ...over,
})
const near = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} ≉ ${b}`)

test('a shape with no animation fields is static and always fully visible', () => {
  const s = hl()
  for (const t of [0, 1, 100]) {
    const r = resolveShape(s, t)!
    assert.equal(r.a, 1); assert.deepEqual((r.s as any).pos, [0.2, 0.5])
  }
  assert.equal(isAnimated([s]), false)
})

test('auto-key: an edit at the start changes the base, an edit later makes a keyframe', () => {
  const s = hl()
  const atStart = applyEdit(s, { pos: [0.25, 0.5] } as any, 0, true) as any
  assert.deepEqual(atStart.pos, [0.25, 0.5]); assert.equal(atStart.kf, undefined)
  const moved = applyEdit(atStart, { pos: [0.6, 0.4] } as any, 1.5, true) as any
  assert.deepEqual(moved.pos, [0.25, 0.5])           // base untouched
  assert.equal(moved.kf.length, 1); assert.equal(moved.kf[0].t, 1.5)
  assert.deepEqual(moved.kf[0].v, { pos: [0.6, 0.4] })
})

test('dragging repeatedly at the same moment updates one keyframe, not many', () => {
  let s = hl()
  for (let i = 0; i < 10; i++) s = applyEdit(s, { pos: [0.3 + i * 0.01, 0.5] } as any, 2, true)
  assert.equal((s as any).kf.length, 1)
  s = applyEdit(s, { pos: [0.9, 0.9] } as any, 2.03, true)   // within KEY_EPS
  assert.equal((s as any).kf.length, 1)
})

test('with auto-key off, edits change the starting state unless a keyframe sits at the playhead', () => {
  const s = hl()
  const e = applyEdit(s, { pos: [0.7, 0.7] } as any, 2, false) as any
  assert.deepEqual(e.pos, [0.7, 0.7]); assert.equal(e.kf, undefined)
})

test('non-animatable edits (colour excepted) never create keyframes', () => {
  const s = applyEdit(hl(), { playerName: 'Müller', dash: 'dashed' } as any, 3, true) as any
  assert.equal(s.kf, undefined); assert.equal(s.playerName, 'Müller'); assert.equal(s.dash, 'dashed')
})

test('two-point move eases and reaches both ends exactly; only changed props are keyed', () => {
  const s = applyEdit(hl(), { pos: [0.6, 0.5] } as any, 2, true)
  near((resolveShape(s, 0)!.s as any).pos[0], 0.2)
  near((resolveShape(s, 2)!.s as any).pos[0], 0.6)
  near((resolveShape(s, 1)!.s as any).pos[0], 0.4)          // smoothstep midpoint
  assert.equal((resolveShape(s, 1)!.s as any).radiusX, 0.03) // untouched prop stays put
  assert.deepEqual((resolveShape(s, 9)!.s as any).pos, [0.6, 0.5]) // holds after the last keyframe
})

test('three keyframes pass exactly through each keyframe', () => {
  let s = hl()
  s = applyEdit(s, { pos: [0.4, 0.3] } as any, 1, true)
  s = applyEdit(s, { pos: [0.5, 0.8] } as any, 2, true)
  for (const [t, p] of [[0, [0.2, 0.5]], [1, [0.4, 0.3]], [2, [0.5, 0.8]]] as const) {
    const r = (resolveShape(s, t)!.s as any).pos
    near(r[0], p[0]); near(r[1], p[1])
  }
  const mid = (resolveShape(s, 1.5)!.s as any).pos
  assert.ok(mid[0] > 0.4 && mid[0] < 0.5 && mid[1] > 0.3 && mid[1] < 0.8)
})

test('colour animates between keyframes', () => {
  const s = applyEdit(hl({ color: '#000000' }), { color: '#ffffff' } as any, 2, true)
  assert.equal((resolveShape(s, 1)!.s as any).color, '#808080')
})

test('cone angle takes the short way round', () => {
  const cone: any = { id: 'c', type: 'cone', pos: [0.5, 0.5], angle: 3.0, length: 0.1, width: 1, color: '#38bdf8', opacity: 0.3, dash: 'solid' }
  const s = applyEdit(cone, { angle: -3.0 } as any, 1, true)
  const a = (resolveShape(s, 0.5)!.s as any).angle
  assert.ok(Math.abs(a) > 3.0, `expected to pass through ±π, got ${a}`)
})

test('appear / disappear windows fade and hide', () => {
  const s = setWindow(hl(), 1, 3)
  assert.equal(resolveShape(s, 0.5), null)
  const f = resolveShape(s, 1.125)!; near(f.a, 0.5)
  assert.equal(alphaAt(s, 2), 1)
  near(alphaAt(s, 2.875), 0.5)
  assert.equal(resolveShape(s, 3.1), null)
  assert.equal(alphaAt(hl(), 0), 1, 'static shape must not fade in at τ = 0')
})

test('a keyframe can never end up at or before the appear time', () => {
  let s = applyEdit(hl(), { pos: [0.9, 0.9] } as any, 1, true)
  s = setWindow(s, 2, undefined)
  assert.ok((s as any).kf[0].t > 2)
})

test('curve draw-on progress', () => {
  const c: any = { id: 'c', type: 'curve', from: [0, 0], control: [0.5, 0], to: [1, 0], style: 'pass', color: '#fff', opacity: 1, dash: 'solid', draw: true, t0: 1 }
  near(resolveShape(c, 1.3)!.p, 0.5); near(resolveShape(c, 2)!.p, 1)
})

test('staggerReveal orders by creation, connectors wait for their ends, and clearReveal undoes it', () => {
  const a = hl({ id: 'a' }), b = hl({ id: 'b' })
  const conn: any = { id: 'k', type: 'connector', fromId: 'a', toId: 'b', color: '#f0f', opacity: 1, dash: 'dashed' }
  const out: any[] = staggerReveal([a, b, conn]) as any
  assert.ok(out[0].t0 < out[1].t0)
  assert.ok(out[2].t0 > out[1].t0)
  assert.ok(isAnimated(out) && spanEnd(out) > out[2].t0)
  assert.equal(isAnimated(clearReveal(out)), false)
})

test('removeKeyframe, keyframeAt, duplicateShape', () => {
  let s = applyEdit(hl(), { pos: [0.5, 0.5] } as any, 1, true)
  assert.ok(keyframeAt(s, 1.02))
  s = removeKeyframe(s, 1)
  assert.equal((s as any).kf, undefined)
  const d: any = duplicateShape(applyEdit(hl(), { pos: [0.5, 0.5] } as any, 1, true))
  assert.notEqual(d.id, 'h1'); assert.equal(d.kf, undefined)
})

test('marker size: prior, one correction rescales every depth, several corrections learn the slope', () => {
  const A = 16 / 9
  const prior = markerSize([], 0.5, A)
  assert.ok(prior.radiusX > 0.016 && prior.radiusX < 0.042)
  // the coach makes one near marker 1.5× the default: a far one scales up too, and the near one is reproduced
  const nearDefault = markerSize([], 0.9, A)
  const sample = { y: 0.9, rx: nearDefault.radiusX * 1.5, ry: nearDefault.radiusY * 1.5 }
  const after = markerSize([sample], 0.9, A)
  near(after.radiusX, sample.rx, 1e-4)
  const farAfter = markerSize([sample], 0.2, A), farBefore = markerSize([], 0.2, A)
  near(farAfter.radiusX / farBefore.radiusX, 1.5, 1e-3)
  // four samples on a steeper line than the prior: prediction in between follows that line
  const line = (y: number) => 0.01 + 0.07 * y
  const many = [0.1, 0.4, 0.7, 0.95].map(y => ({ y, rx: line(y), ry: line(y) * A * 0.6 }))
  near(markerSize(many, 0.55, A).radiusX, line(0.55), 0.006)
  // sizes stay sane
  const huge = markerSize([{ y: 0.5, rx: 5, ry: 5 }], 0.5, A)
  assert.ok(huge.radiusX <= 0.09)
})

test('shiftShape moves appear, disappear and keyframes together and never goes below zero', () => {
  let s: any = applyEdit(setWindow(hl(), 1, 4), { pos: [0.9, 0.9] } as any, 2, true)
  s = shiftShape(s, 0.5) as any
  assert.equal(s.t0, 1.5); assert.equal(s.t1, 4.5); assert.equal(s.kf[0].t, 2.5)
  s = shiftShape(s, -10) as any
  assert.equal(s.t0, undefined); assert.equal(s.t1, 3); assert.equal(s.kf[0].t, 1)
})

test('addKeyframeAt pins the current look so the next edit moves from there', () => {
  let s: any = applyEdit(hl(), { pos: [0.6, 0.5] } as any, 2, true)
  s = addKeyframeAt(s, 1)
  assert.equal(s.kf.length, 2)
  near((resolveShape(s, 1)!.s as any).pos[0], s.kf[0].v.pos[0])
  assert.equal(addKeyframeAt(s, 1.01).kf.length, 2, 'already a keyframe there')
  assert.equal(addKeyframeAt(hl(), 0).kf, undefined, 'nothing to pin at the start')
})
