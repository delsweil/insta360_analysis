import test from 'node:test'
import assert from 'node:assert/strict'
import { plan, frameAt, tween, ease, lerpAngle, along, pathBend, moveOf, holdOf, clampSec, formatTime, DEFAULT_MOVE_S, DEFAULT_HOLD_S, type Step } from './anim.ts'
import { defaultBackground, type Scene, type SceneShape } from './scene.ts'

const bg = defaultBackground('free')
const scene = (shapes: SceneShape[], caption = ''): Scene => ({ background: bg, shapes, caption })
const pl = (id: string, x: number, y: number, extra: object = {}): SceneShape => ({ id, type: 'player', pos: [x, y], team: 'a', ...extra } as SceneShape)
const ln = (id: string, kind: 'pass' | 'return' | 'move', from: [number, number], to: [number, number], extra: object = {}): SceneShape => ({ id, type: 'line', kind, from, to, bend: 0, ...extra } as SceneShape)
const pos = (s: SceneShape) => (s as { pos: number[] }).pos
const near = (p: number[], q: number[], e = 1e-6) => Math.abs(p[0] - q[0]) < e && Math.abs(p[1] - q[1]) < e

test('time line: first scene is held, the others are moved into and held; defaults and clamping', () => {
  const steps: Step[] = [{ scene: scene([]) }, { scene: scene([]), timing: { move_s: 2, hold_s: 3 } }, { scene: scene([]) }]
  const p = plan(steps)
  assert.deepEqual(p.segs.map(s => [s.index, s.kind]), [[0, 'hold'], [1, 'move'], [1, 'hold'], [2, 'move'], [2, 'hold']])
  assert.ok(Math.abs(p.total - (DEFAULT_HOLD_S + 2 + 3 + DEFAULT_MOVE_S + DEFAULT_HOLD_S)) < 1e-9)
  const r1 = (x: number) => Math.round(x * 10) / 10
  assert.deepEqual(p.starts.map(r1), [0, DEFAULT_HOLD_S + 2, DEFAULT_HOLD_S + 2 + 3 + DEFAULT_MOVE_S].map(r1))
  assert.equal(moveOf(null), DEFAULT_MOVE_S); assert.equal(holdOf({ hold_s: 0 }), 0.2); assert.equal(holdOf({ hold_s: 999 }), 30); assert.equal(clampSec(NaN, 1), 1)
  assert.equal(plan([]).total, 0)
})
test('easing starts and ends gently and is monotonic', () => {
  assert.equal(ease(0), 0); assert.equal(ease(1), 1); assert.ok(Math.abs(ease(0.5) - 0.5) < 1e-9)
  let prev = -1; for (let i = 0; i <= 20; i++) { const v = ease(i / 20); assert.ok(v >= prev); prev = v }
})
test('the same player moves from its place in one scene to its place in the next', () => {
  const A = scene([pl('p', 2, 2)]), B = scene([pl('p', 10, 6)])
  assert.ok(near(pos(tween(A, B, 0).shapes[0]), [2, 2])); assert.ok(near(pos(tween(A, B, 1).shapes[0]), [10, 6]))
  const mid = pos(tween(A, B, 0.5).shapes[0]); assert.ok(near(mid, [6, 4]))
  assert.equal(tween(A, B, 0.3).shapes.length, 1) // matched by id: one player, not two
})
test('a run line the coach drew bends the player\'s path, a pass line bends the ball\'s; straight otherwise', () => {
  const A = scene([pl('p', 2, 10), { id: 'b', type: 'ball', pos: [3, 10] }, ln('run', 'move', [2, 10], [12, 10], { bend: 3 }), ln('pass', 'pass', [3, 10], [12, 10], { bend: -4 })])
  const B = scene([pl('p', 12, 10), { id: 'b', type: 'ball', pos: [12, 10] }])
  const f = tween(A, B, 0.5)
  const p = f.shapes.find(s => s.id === 'p')!, b = f.shapes.find(s => s.id === 'b')!
  assert.ok(Math.abs(pos(p)[1] - 10) > 1, 'player leaves the straight line'); assert.ok(Math.abs(pos(b)[1] - 10) > 1, 'ball too')
  assert.ok(Math.sign(pos(p)[1] - 10) === -Math.sign(pos(b)[1] - 10), 'opposite bends go opposite ways')
  assert.ok(near(pos(tween(A, B, 1).shapes.find(s => s.id === 'p')!), [12, 10], 1e-6)) // still ends exactly at its place
  const noLine = tween(scene([pl('q', 2, 10)]), scene([pl('q', 12, 10)]), 0.5).shapes[0]; assert.ok(near(pos(noLine), [7, 10]))
  assert.equal(pathBend(A, pl('p', 2, 10) as never, [2, 10], [12, 10]), 3)
  assert.equal(pathBend(A, pl('x', 18, 25) as never, [18, 25], [12, 10]), 0) // a line elsewhere does not apply
})
test('new shapes fade in, vanished ones fade out, lines of the old scene go quickly', () => {
  const A = scene([pl('gone', 3, 3), ln('l', 'pass', [3, 3], [8, 3])]), B = scene([pl('new', 9, 9)])
  const mid = tween(A, B, 0.5)
  const g = mid.shapes.find(s => s.id === 'gone')!, n = mid.shapes.find(s => s.id === 'new')!, l = mid.shapes.find(s => s.id === 'l')!
  assert.ok(Math.abs((g.alpha ?? 1) - 0.5) < 1e-9 && Math.abs((n.alpha ?? 0) - 0.5) < 1e-9); assert.equal(l.alpha, 0)
  const start = tween(A, B, 0); assert.equal(start.shapes.find(s => s.id === 'gone')!.alpha, 1); assert.equal(start.shapes.find(s => s.id === 'l')!.alpha, 1)
  const end = tween(A, B, 1); assert.equal(end.shapes.find(s => s.id === 'new')!.alpha, 1); assert.equal(end.shapes.find(s => s.id === 'gone')!.alpha, 0)
})
test('lines attached to players follow them while they move', () => {
  const A = scene([pl('a', 2, 2), pl('b', 10, 2), ln('l', 'pass', [2, 2], [10, 2], { fromId: 'a', toId: 'b' })])
  const B = scene([pl('a', 2, 12), pl('b', 10, 12), ln('l', 'pass', [2, 12], [10, 12], { fromId: 'a', toId: 'b' })])
  const l = tween(A, B, 0.5).shapes.find(s => s.id === 'l') as Extract<SceneShape, { type: 'line' }>
  assert.ok(near(l.from, [2, 7]) && near(l.to, [10, 7]))
})
test('equipment turns the short way and keeps its scale; background and caption switch at the half way point', () => {
  assert.equal(lerpAngle(350, 10, 0.5), 0); assert.equal(Math.round(lerpAngle(10, 350, 0.5)), 0); assert.equal(lerpAngle(0, 90, 1), 90)
  const A = scene([{ id: 'g', type: 'equip', kind: 'goal', pos: [5, 5], angle: 350, scale: 1 } as SceneShape], 'eins')
  const B = { ...scene([{ id: 'g', type: 'equip', kind: 'goal', pos: [5, 5], angle: 10, scale: 2 } as SceneShape], 'zwei'), background: defaultBackground('pitch') }
  const m = tween(A, B, 0.5); assert.equal((m.shapes[0] as { angle: number }).angle, 0); assert.equal((m.shapes[0] as { scale: number }).scale, 1.5)
  assert.equal(tween(A, B, 0.4).caption, 'eins'); assert.equal(tween(A, B, 0.6).caption, 'zwei'); assert.equal(tween(A, B, 0.6).background.type, 'pitch')
})
test('frameAt: holds show the scene itself, moves are in between, the end stays on the last scene', () => {
  const steps: Step[] = [{ scene: scene([pl('p', 0, 0)], 'a'), timing: { hold_s: 1 } }, { scene: scene([pl('p', 10, 0)], 'b'), timing: { move_s: 2, hold_s: 1 } }]
  assert.equal(frameAt(steps, 0.5).kind, 'hold'); assert.equal(frameAt(steps, 0.5).index, 0); assert.equal(frameAt(steps, 0.5).scene, steps[0].scene)
  const m = frameAt(steps, 2); assert.equal(m.kind, 'move'); assert.ok(near(pos(m.scene.shapes[0]), [5, 0], 1e-6))
  assert.equal(frameAt(steps, 3.5).scene, steps[1].scene); assert.equal(frameAt(steps, 999).index, 1); assert.equal(frameAt(steps, -5).index, 0)
  assert.deepEqual(frameAt([], 0).scene.shapes, [])
})
test('along follows a polyline by length; time formatting', () => {
  assert.deepEqual(along([[0, 0], [10, 0], [10, 10]], 0.5), [10, 0]); assert.deepEqual(along([[0, 0], [10, 0]], 1), [10, 0]); assert.deepEqual(along([[3, 3]], 0.5), [3, 3])
  assert.equal(formatTime(75), '1:15'); assert.equal(formatTime(0), '0:00')
})

test('tween: a carried ball travels with its carrier', () => {
  const ball = (x: number, y: number): SceneShape => ({ id: 'b', type: 'ball', pos: [x, y], carriedBy: 'p' } as SceneShape)
  const A = scene([pl('p', 0, 0), ball(0.5, 0)]), B = scene([pl('p', 10, 4), ball(10.5, 4)])
  const m = tween(A, B, 0.5)
  const p = m.shapes.find(s => s.id === 'p') as any, b = m.shapes.find(s => s.id === 'b') as any
  assert.ok(Math.abs(b.pos[0] - p.pos[0] - 0.5) < 1e-9 && Math.abs(b.pos[1] - p.pos[1]) < 1e-9)
})
