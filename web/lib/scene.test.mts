import test from 'node:test'
import assert from 'node:assert/strict'
import {
  defaultBackground, viewBoxOf, symbolUnit, formatMetres, dimText, gridSizeText, clampFrameSize, snapPt, snapStep, clampPt,
  makeShape, isMeaningful, moveShape, removeShape, duplicateShape, copyScene, emptyScene, hitTest, handlesOf, handleAt, handlePatch,
  lineEndTarget, drawOrder, moveShapes, duplicateShapes, removeShapes, colorShapes, scaleShapes, shapesInRect, shapeBox, rotPt, gridCorners, gridCentre, inGrid, gridResized, rotateShapes, gridSplits, gridLattice, gridCellText, addFieldCones, shapesInGrid, cloneShapes, fieldShape, sizeHandle, scaleOf, equipDims, isLight, rotHandle, equipSize, updateShape, hist, hPush, hUndo, hRedo,
  type SceneShape, type Scene,
} from './scene.ts'
import { dist, curvePoints, trimPolyline, resample, wobble, wavy, arrowHead, blobPath, decimate, pathLength, hashStr } from './sketch.ts'

const bg = defaultBackground('free') // 20 x 30 m
const player = (id: string, x: number, y: number): SceneShape => ({ id, type: 'player', pos: [x, y], team: 'a' })
const line = (id: string, from: [number, number], to: [number, number], extra: Partial<SceneShape> = {}): SceneShape => ({ id, type: 'line', kind: 'pass', from, to, bend: 0, ...extra } as SceneShape)

test('frames: sizes, view box margin, symbol unit', () => {
  assert.deepEqual(defaultBackground('pitch'), { type: 'pitch', width_m: 105, height_m: 68 })
  assert.deepEqual(defaultBackground('halfpitch'), { type: 'halfpitch', width_m: 68, height_m: 52.5 })
  const v = viewBoxOf(bg)
  assert.ok(v.x < 0 && v.w > bg.width_m && v.h > bg.height_m)
  assert.ok(symbolUnit(defaultBackground('pitch')) > symbolUnit(bg))
  assert.equal(clampFrameSize(500), 120); assert.equal(clampFrameSize(0), 3); assert.equal(clampFrameSize(NaN), 3); assert.equal(clampFrameSize(22.46), 22.5)
})

test('metres are formatted the German way', () => {
  assert.equal(formatMetres(12), '12 m'); assert.equal(formatMetres(12.5), '12,5 m'); assert.equal(formatMetres(7.32), '7,3 m'); assert.equal(formatMetres(0.04), '0 m')
  assert.equal(dimText({ id: 'd', type: 'dim', a: [0, 0], b: [3, 4] }), '5 m')
  assert.equal(dimText({ id: 'd', type: 'dim', a: [0, 0], b: [3, 4], text: ' 4,5 m ' }), '4,5 m')
  assert.equal(dimText({ id: 'd', type: 'dim', a: [0, 0], b: [3, 4], text: '   ' }), '5 m')
  assert.equal(gridSizeText({ id: 'g', type: 'grid', a: [10, 20], b: [0, 0], showSize: true }), '10 × 20 m')
})

test('snapping and clamping', () => {
  assert.deepEqual(snapPt([3.4, 7.6], 1), [3, 8]); assert.deepEqual(snapPt([12, 13], 5), [10, 15])
  assert.equal(snapStep(bg), 1); assert.equal(snapStep(defaultBackground('pitch')), 5)
  const c = clampPt([-100, 999], bg); assert.ok(c[0] > -5 && c[1] < 40)
})

test('making shapes and dropping accidental taps', () => {
  const l = makeShape('pass', [1, 1])!; assert.equal(l.type, 'line'); assert.equal(isMeaningful(l, 0.5), false)
  assert.equal(isMeaningful({ ...(l as object), to: [5, 1] } as SceneShape, 0.5), true)
  assert.equal(isMeaningful(makeShape('player', [1, 1])!, 0.5), true)
  assert.equal(isMeaningful(makeShape('text', [1, 1])!, 0.5), false)
  assert.equal(makeShape('select', [0, 0]), null)
  const g = makeShape('grid', [0, 0])!; assert.equal(isMeaningful({ ...(g as object), b: [4, 0] } as SceneShape, 0.5), false)
})

test('moving a player carries attached lines, moving a line detaches it', () => {
  const s = [player('p1', 2, 2), player('p2', 10, 2), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' })]
  const m = moveShape(s, 'p1', 1, 3)
  assert.deepEqual((m[0] as { pos: number[] }).pos, [3, 5])
  const l = m[2] as Extract<SceneShape, { type: 'line' }>
  assert.deepEqual(l.from, [3, 5]); assert.deepEqual(l.to, [10, 2]); assert.equal(l.fromId, 'p1')
  const m2 = moveShape(s, 'l1', 1, 1)[2] as Extract<SceneShape, { type: 'line' }>
  assert.deepEqual(m2.from, [3, 3]); assert.equal('fromId' in m2, false); assert.equal('toId' in m2, false)
  assert.equal(moveShape(s, 'nope', 1, 1), s)
})

test('deleting a player detaches lines but keeps them', () => {
  const s = [player('p1', 2, 2), player('p2', 10, 2), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' })]
  const r = removeShape(s, 'p1'); assert.equal(r.length, 2)
  const l = r[1] as Extract<SceneShape, { type: 'line' }>
  assert.equal('fromId' in l, false); assert.equal(l.toId, 'p2'); assert.deepEqual(l.from, [2, 2])
})

test('duplicate gets a new id, an offset, and no attachments', () => {
  const l = line('l1', [2, 2], [10, 2], { fromId: 'p1' })
  const d = duplicateShape(l, 1, 1) as Extract<SceneShape, { type: 'line' }>
  assert.notEqual(d.id, 'l1'); assert.deepEqual(d.from, [3, 3]); assert.equal('fromId' in d, false)
})

test('copying a scene keeps the shape ids (same id = same player in the next step)', () => {
  const sc: Scene = { ...emptyScene('free'), shapes: [player('p1', 1, 1)], caption: 'x' }
  const c = copyScene(sc); assert.equal(c.shapes[0].id, 'p1'); assert.notEqual(c.shapes, sc.shapes)
  ;(c.shapes[0] as { pos: number[] }).pos[0] = 9; assert.equal((sc.shapes[0] as { pos: number[] }).pos[0], 1)
})

test('hit-testing: topmost wins, grids only at the edge, touch slop helps', () => {
  const u = symbolUnit(bg)
  const s: SceneShape[] = [
    { id: 'g', type: 'grid', a: [2, 2], b: [18, 28], showSize: true },
    player('p', 10, 10),
    { id: 'b', type: 'ball', pos: [10 + u * 1.1, 10] },
  ]
  assert.equal(hitTest(s, [10, 10], 0.3, bg), 'p')
  assert.equal(hitTest(s, [10 + u * 1.1, 10], 0.1, bg), 'b') // the ball is a separate target at the player's edge
  assert.equal(hitTest(s, [10, 20], 0.3, bg), null)            // inside the grid, not on an edge: nothing
  assert.equal(hitTest(s, [2.1, 15], 0.3, bg), 'g')            // on the grid edge
  assert.equal(hitTest(s, [10 - u - 0.5, 10], 0.3, bg), null)
  assert.equal(hitTest(s, [10 - u - 0.5, 10], 1.5, bg), 'p')   // more slop (finger)
})

test('hit-testing lines, rotated goals and text', () => {
  const s: SceneShape[] = [
    line('l', [0, 0], [10, 0]),
    { id: 'goal', type: 'equip', kind: 'goal', pos: [10, 20], angle: 90 } as SceneShape,
    { id: 't', type: 'text', pos: [5, 25], text: '2 shift', size: 1 },
  ]
  assert.equal(hitTest(s, [5, 0.2], 0.4, bg), 'l'); assert.equal(hitTest(s, [5, 1.5], 0.4, bg), null)
  // a goal turned 90 degrees is 2 m wide and 7.32 m tall
  assert.equal(hitTest(s, [10, 20 + 3.4], 0.2, bg), 'goal'); assert.equal(hitTest(s, [10 + 3.4, 20], 0.2, bg), null)
  assert.equal(hitTest(s, [5, 25], 0.2, bg), 't')
})

test('line handles: ends attach to a nearby player, bend is signed, tiny bends snap to straight', () => {
  const ps = [player('p1', 2, 2), player('p2', 10, 2)]
  const l = line('l', [5, 5], [8, 5]) as Extract<SceneShape, { type: 'line' }>
  const all = [...ps, l]
  const h = handlesOf(l, bg); assert.deepEqual(h.map(x => x.key), ['from', 'to', 'bend'])
  assert.equal(handleAt(l, bg, [5.1, 5.1], 0.5), 'from')
  const a = handlePatch(l, 'to', [10.3, 2.2], all, bg) as { to: number[]; toId?: string }
  assert.equal(a.toId, 'p2'); assert.deepEqual(a.to, [10, 2])
  const f = handlePatch(l, 'to', [15, 15], all, bg) as { to: number[]; toId?: string }
  assert.equal(f.toId, undefined); assert.deepEqual(f.to, [15, 15])
  const up = handlePatch(l, 'bend', [6.5, 8], all, bg) as { bend: number }, down = handlePatch(l, 'bend', [6.5, 2], all, bg) as { bend: number }
  assert.ok(up.bend * down.bend < 0)
  assert.equal((handlePatch(l, 'bend', [6.5, 5.02], all, bg) as { bend: number }).bend, 0)
})

test('an attached line end has its handle at the player\'s edge, so the player stays grabbable', () => {
  const u = symbolUnit(bg)
  const ps = [player('p1', 2, 2), player('p2', 12, 2)]
  const l = line('l', [2, 2], [12, 2], { fromId: 'p1', toId: 'p2' })
  const all = [...ps, l]
  const withPlayers = handlesOf(l, bg, all), without = handlesOf(l, bg)
  assert.ok(Math.abs(withPlayers[0].pos[0] - (2 + u * 1.25)) < 0.2 && Math.abs(withPlayers[1].pos[0] - (12 - u * 1.25)) < 0.2)
  assert.deepEqual(without[0].pos, [2, 2])
  assert.equal(handleAt(l, bg, [2, 2], 0.5, all), null)              // on the player's centre: no handle
  assert.equal(handleAt(l, bg, [2 + u * 1.25, 2], 0.5, all), 'from') // at the arrow end: the handle
})

test('goal rotation handle snaps to 5 degrees', () => {
  const g = { id: 'g', type: 'equip', kind: 'kleinfeldgoal', pos: [10, 10], angle: 0 } as SceneShape
  assert.equal((handlePatch(g, 'rot', [20, 10], [g], bg) as { angle: number }).angle, 90)
  assert.equal((handlePatch(g, 'rot', [10, 20], [g], bg) as { angle: number }).angle, 180)
  assert.equal((handlePatch(g, 'rot', [10, 0], [g], bg) as { angle: number }).angle, 0)
  assert.equal((handlePatch(g, 'rot', [0, 10], [g], bg) as { angle: number }).angle, 270)
})

test('the rotation handle swings to the other side when it would fall outside the picture', () => {
  const g = { id: 'g', type: 'equip', kind: 'kleinfeldgoal', pos: [10, 1.2], angle: 0 } as Extract<SceneShape, { type: 'equip' }>
  const r = rotHandle(g, bg)
  assert.equal(r.flipped, true); assert.ok(r.pos[1] > g.pos[1])
  assert.equal(rotHandle({ ...g, pos: [10, 15] }, bg).flipped, false)
  // dragging the flipped handle to straight below the goal keeps the goal facing up
  assert.equal((handlePatch(g, 'rot', [10, 9], [g], bg) as { angle: number }).angle, 0)
  assert.equal((handlePatch(g, 'rot', [20, 1.2], [g], bg) as { angle: number }).angle, 270)
})

test('grid corner handles resize', () => {
  const g = { id: 'g', type: 'grid', a: [2, 2], b: [10, 10], showSize: true } as SceneShape
  assert.equal(handlesOf(g, bg).length, 5) // four corners and the rotation handle
  assert.deepEqual({ ...g, ...handlePatch(g, 'b', [12, 14], [g], bg) }, { ...g, a: [2, 2], b: [12, 14] })
  assert.deepEqual({ ...g, ...handlePatch(g, 'ab', [1, 12], [g], bg) }, { ...g, a: [1, 2], b: [10, 12] })
})

test('dragging a new line end snaps to a player', () => {
  const r = lineEndTarget([player('p', 10, 10)], [10.5, 10.5], bg); assert.equal(r.id, 'p'); assert.deepEqual(r.pos, [10, 10])
  assert.equal(lineEndTarget([player('p', 10, 10)], [15, 15], bg).id, undefined)
})

test('goal sizes are real; draw order puts grids behind and text on top', () => {
  const u = symbolUnit(defaultBackground('pitch'))
  assert.equal(equipSize('goal', u).w, 7.32); assert.equal(equipSize('kleinfeldgoal', u).w, 5)
  assert.ok(equipSize('minigoal', 0.6).w >= 1.2)
  const s: SceneShape[] = [{ id: 't', type: 'text', pos: [0, 0], text: 'a', size: 1 }, player('p', 1, 1), { id: 'g', type: 'grid', a: [0, 0], b: [5, 5], showSize: false }]
  assert.deepEqual(drawOrder(s).map(x => x.id), ['g', 'p', 't'])
})

test('updateShape removes cleared fields', () => {
  const s = [line('l', [0, 0], [4, 0], { fromId: 'p1' })]
  const u = updateShape(s, 'l', { fromId: undefined } as Partial<SceneShape>)[0]
  assert.equal('fromId' in u, false)
})

test('history: push, undo, redo, new push drops the redo branch', () => {
  let h = hist(1); h = hPush(h, 2); h = hPush(h, 3)
  h = hUndo(h); assert.equal(h.present, 2); h = hUndo(h); assert.equal(h.present, 1); h = hUndo(h); assert.equal(h.present, 1)
  h = hRedo(h); assert.equal(h.present, 2); h = hPush(h, 9); assert.equal(h.future.length, 0); assert.equal(hRedo(h).present, 9)
})

// sketch helpers
test('sketch: curve bend moves the middle by exactly the bend, straight when 0', () => {
  const p = curvePoints([0, 0], [10, 0], 2, 24); assert.ok(Math.abs(Math.abs(p[12][1]) - 2) < 1e-6)
  const q = curvePoints([0, 0], [10, 0], 0, 8); assert.ok(q.every(pt => Math.abs(pt[1]) < 1e-9))
})
test('sketch: trim, resample and wobble keep the ends where they are', () => {
  const t = trimPolyline([[0, 0], [10, 0]], 1, 2); assert.deepEqual(t[0], [1, 0]); assert.ok(Math.abs(t[t.length - 1][0] - 8) < 1e-9)
  const tiny = trimPolyline([[0, 0], [1, 0]], 5, 5); assert.equal(tiny.length, 2)
  const r = resample([[0, 0], [10, 0]], 2); assert.equal(r.length, 6); assert.deepEqual(r[r.length - 1], [10, 0])
  const w = wobble([[0, 0], [20, 0]], 7, 0.3)
  assert.deepEqual(w[0], [0, 0]); assert.ok(Math.abs(w[w.length - 1][0] - 20) < 1e-6 && Math.abs(w[w.length - 1][1]) < 1e-6)
  assert.ok(w.some(p => Math.abs(p[1]) > 0.02) && w.every(p => Math.abs(p[1]) <= 0.3 + 1e-9))
  assert.deepEqual(wobble([[0, 0], [20, 0]], 7, 0.3), w) // deterministic: no shimmer
  assert.notDeepEqual(wobble([[0, 0], [20, 0]], 8, 0.3), w)
  assert.deepEqual(wobble([[0, 0], [20, 0]], 7, 0), [[0, 0], [20, 0]])
})
test('sketch: wavy line oscillates around the path and ends on it', () => {
  const w = wavy([[0, 0], [12, 0]], 0.4, 2)
  assert.ok(Math.max(...w.map(p => p[1])) > 0.3 && Math.min(...w.map(p => p[1])) < -0.3)
  assert.ok(Math.abs(w[0][1]) < 1e-6 && Math.abs(w[w.length - 1][1]) < 0.05)
})
test('sketch: arrow head, blob, decimate, hash', () => {
  assert.match(arrowHead([0, 0], [10, 0], 1), /^M[\d. -]+L10 0L/)
  assert.match(blobPath([5, 5], 1, 3, 0), /a1 1/); assert.ok(blobPath([5, 5], 1, 3, 0.1).endsWith('Z'))
  assert.equal(decimate([[0, 0], [0.01, 0], [0.02, 0], [1, 0], [1.01, 0]], 0.5).length, 3)
  assert.ok(hashStr('a') !== hashStr('b')); assert.ok(pathLength([[0, 0], [3, 4]]) === 5 && dist([0, 0], [3, 4]) === 5)
})

// group operations, colour, size, rectangle selection
test('group move keeps attached lines on players that move with them, lets go of the rest', () => {
  const shapes = [player('p1', 2, 2), player('p2', 10, 2), player('p3', 15, 15), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' }), line('l2', [10, 2], [15, 15], { fromId: 'p2', toId: 'p3' })]
  const out = moveShapes(shapes, ['p1', 'p2', 'l1'], 1, 1)
  const l1 = out.find(s => s.id === 'l1') as Extract<SceneShape, { type: 'line' }>
  assert.deepEqual(l1.from, [3, 3]); assert.deepEqual(l1.to, [11, 3]); assert.equal(l1.fromId, 'p1'); assert.equal(l1.toId, 'p2')
  const l2 = out.find(s => s.id === 'l2') as Extract<SceneShape, { type: 'line' }>
  assert.deepEqual(l2.from, [11, 3]); assert.deepEqual(l2.to, [15, 15]) // unselected line follows only its moved end
  const lone = moveShapes(shapes, ['l1'], 1, 0) // a selected line alone detaches
  assert.equal((lone.find(s => s.id === 'l1') as Extract<SceneShape, { type: 'line' }>).fromId, undefined)
})
test('duplicating a group remaps attachments inside the group and drops the others', () => {
  const shapes = [player('p1', 2, 2), player('p2', 10, 2), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' })]
  const r = duplicateShapes(shapes, ['p1', 'l1'], 1, 1)
  assert.equal(r.shapes.length, 5); assert.equal(r.newIds.length, 2)
  const nl = r.shapes.find(s => s.id === r.newIds[1]) as Extract<SceneShape, { type: 'line' }>
  assert.equal(nl.fromId, r.newIds[0]); assert.equal(nl.toId, undefined)
  assert.ok(new Set(r.shapes.map(s => s.id)).size === 5)
})
test('removeShapes deletes several and detaches lines', () => {
  const shapes = [player('p1', 2, 2), player('p2', 10, 2), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' })]
  const out = removeShapes(shapes, ['p1', 'p2'])
  assert.equal(out.length, 1); assert.equal((out[0] as Extract<SceneShape, { type: 'line' }>).fromId, undefined)
})
test('colour and scale apply to the right shapes', () => {
  const shapes: SceneShape[] = [player('p', 1, 1), { id: 'b', type: 'ball', pos: [2, 2] }, line('l', [0, 0], [5, 5]), { id: 'g', type: 'equip', kind: 'goal', pos: [10, 10], angle: 0 }, { id: 't', type: 'text', pos: [3, 3], text: 'x', size: 1 }]
  const c = colorShapes(shapes, ['p', 'b', 'l', 'g'], '#ef4444')
  assert.equal((c[0] as { color?: string }).color, '#ef4444'); assert.equal('color' in c[1], false); assert.equal((c[2] as { color?: string }).color, '#ef4444'); assert.equal((c[3] as { color?: string }).color, '#ef4444')
  assert.equal('color' in colorShapes(c, ['p'], undefined)[0], false)
  const z = scaleShapes(shapes, ['p', 'b', 'g', 't', 'l'], 2)
  assert.equal(scaleOf(z[0]), 2); assert.equal(scaleOf(z[1]), 2); assert.equal(scaleOf(z[3]), 2); assert.equal((z[4] as { size: number }).size, 2)
  assert.deepEqual(z[2], shapes[2]) // lines are not scaled
  assert.equal(scaleOf(scaleShapes(shapes, ['p'], 100)[0]), 4); assert.equal(scaleOf(scaleShapes(shapes, ['p'], 0.001)[0]), 0.4)
  const eq = z[3] as Extract<SceneShape, { type: 'equip' }>
  assert.ok(Math.abs(equipDims(eq, symbolUnit(bg)).w - 7.32 * 2) < 1e-9)
})
test('scaled symbols are easier to hit and the size handle follows them', () => {
  const small: SceneShape = player('p', 10, 10)
  const big: SceneShape = { ...player('p', 10, 10), scale: 2 } as SceneShape
  const u = symbolUnit(bg), p: [number, number] = [10 + u * 1.7, 10]
  assert.equal(hitTest([small], p, 0.01, bg), null); assert.equal(hitTest([big], p, 0.01, bg), 'p')
  const h0 = sizeHandle(small as Extract<SceneShape, { type: 'player' }>, bg), h1 = sizeHandle(big as Extract<SceneShape, { type: 'player' }>, bg)
  assert.ok(dist(h1, [10, 10]) > dist(h0, [10, 10]))
  // dragging the handle out to twice the distance doubles the size, back in shrinks it
  const patch = handlePatch(small, 'size', [10 + (u * 0.9 * 2 + u * 0.5), 10], [small], bg) as { scale: number }
  assert.ok(Math.abs(patch.scale - 2) <= 0.05)
  assert.equal((handlePatch(small, 'size', [10, 10], [small], bg) as { scale: number }).scale, 0.4)
  assert.ok(handlesOf(small, bg).some(h => h.key === 'size'))
})
test('rectangle selection takes what lies completely inside', () => {
  const shapes: SceneShape[] = [player('a', 3, 3), player('b', 8, 8), player('c', 18, 25), { id: 'g', type: 'grid', a: [1, 1], b: [12, 12], showSize: true }, line('l', [4, 4], [9, 9])]
  assert.deepEqual(shapesInRect(shapes, [0, 0], [10, 10], bg).sort(), ['a', 'b', 'l'])
  assert.deepEqual(shapesInRect(shapes, [10, 10], [0, 0], bg).sort(), ['a', 'b', 'l']) // any corner order
  assert.deepEqual(shapesInRect(shapes, [0, 0], [13, 13], bg).sort(), ['a', 'b', 'g', 'l'])
  assert.deepEqual(shapesInRect(shapes, [3, 3], [4, 4], bg), []) // a player only partly inside is not taken
  const b = shapeBox(player('x', 5, 5), bg); assert.ok(b.x0 < 5 && b.x1 > 5)
})
test('light colours get dark text', () => { assert.ok(isLight('#ffffff') && isLight('#f5c542')); assert.ok(!isLight('#1b2a6b') && !isLight('#ef4444')) })

// rotated fields, group rotation, clipboard, Feldaufbau
const grid = (id: string, a: [number, number], b: [number, number], angle?: number): SceneShape => ({ id, type: 'grid', a, b, showSize: true, ...(angle ? { angle } : {}) } as SceneShape)
const near = (p: number[], q: number[], e = 1e-6) => Math.abs(p[0] - q[0]) < e && Math.abs(p[1] - q[1]) < e
test('rotating points is clockwise on screen and reversible', () => {
  assert.ok(near(rotPt([1, 0], [0, 0], 90), [0, 1])); assert.ok(near(rotPt(rotPt([3, 2], [1, 1], 37), [1, 1], -37), [3, 2]))
})
test('a turned field: corners, inside test and hit test follow the angle', () => {
  const g = grid('g', [0, 0], [10, 4], 90) as Extract<SceneShape, { type: 'grid' }>
  const cs = gridCorners(g) // centre (5,2): the 10 × 4 field now stands upright, 4 wide and 10 high
  const xs = cs.map(c => c[0]), ys = cs.map(c => c[1])
  assert.ok(Math.abs(Math.max(...xs) - Math.min(...xs) - 4) < 1e-6 && Math.abs(Math.max(...ys) - Math.min(...ys) - 10) < 1e-6)
  assert.ok(inGrid(g, [5, -2.5]) && !inGrid(g, [9, 2]))
  assert.equal(hitTest([g], [5, -3], 0.3, bg), 'g') // its (turned) top edge
  assert.equal(hitTest([g], [9.5, 2], 0.3, bg), null) // where the unturned right end would be
  assert.deepEqual(gridCentre(g), [5, 2])
})
test('dragging a corner of a turned field keeps the opposite corner fixed in the picture', () => {
  const g = grid('g', [0, 0], [10, 4], 30)
  const before = gridCorners(g as Extract<SceneShape, { type: 'grid' }>)
  const patch = handlePatch(g, 'b', rotPt([12, 6], [5, 2], 30), [g], defaultBackground('pitch')) as Partial<SceneShape>
  const g2 = { ...g, ...patch } as Extract<SceneShape, { type: 'grid' }>
  const after = gridCorners(g2)
  assert.ok(near(after[0], before[0], 1e-6)) // corner a did not move
  assert.ok(near(after[2], rotPt([12, 6], [5, 2], 30), 1e-6)) // dragged corner is where it was dragged
  const pk = handlePatch(g, 'ab', before[3], [g], defaultBackground('pitch')) as Partial<SceneShape>
  assert.ok(near(gridCorners({ ...g, ...pk } as Extract<SceneShape, { type: 'grid' }>)[1], before[1], 1e-6))
})
test('field rotation handle sets the angle in 5° steps and resizing keeps centre and angle', () => {
  const g = grid('g', [20, 20], [40, 30]) as Extract<SceneShape, { type: 'grid' }>
  const pitch = defaultBackground('pitch')
  assert.ok(handlesOf(g, pitch).some(h => h.key === 'rot'))
  const p = handlePatch(g, 'rot', [50, 25], [g], pitch) as { angle: number }
  assert.equal(p.angle, 90)
  const r = { ...g, angle: 40 } as Extract<SceneShape, { type: 'grid' }>
  const n = { ...r, ...gridResized(r, 12, 6) } as Extract<SceneShape, { type: 'grid' }>
  assert.deepEqual(gridCentre(n), gridCentre(r)); assert.equal(n.angle, 40); assert.equal(Math.abs(n.b[0] - n.a[0]), 12)
})
test('rotating a group turns equipment and fields, moves the rest around the centre', () => {
  const shapes: SceneShape[] = [player('p', 10, 0), { id: 'e', type: 'equip', kind: 'goal', pos: [0, 10], angle: 0 }, grid('g', [0, 0], [4, 2]), line('l', [0, 0], [10, 0], { fromId: undefined })]
  const out = rotateShapes(shapes, ['p', 'e', 'g', 'l'], 90, [0, 0])
  assert.ok(near((out[0] as { pos: number[] }).pos, [0, 10])); assert.equal((out[1] as { angle: number }).angle, 90)
  assert.ok(near((out[1] as { pos: number[] }).pos, [-10, 0]))
  assert.equal((out[2] as { angle?: number }).angle, 90)
  assert.ok(near((out[3] as { to: number[] }).to, [0, 10]))
  const back = rotateShapes(out, ['p', 'e', 'g', 'l'], -90, [0, 0])
  assert.ok(near((back[0] as { pos: number[] }).pos, [10, 0])); assert.equal((back[2] as { angle?: number }).angle, undefined); assert.equal((back[1] as { angle: number }).angle, 0)
  assert.equal((rotateShapes(shapes, ['p'], 90, [0, 0])[1] as { angle: number }).angle, 0) // unselected shapes stay
})
test('"select all in a field" takes the field and what lies inside it, turned or not', () => {
  const shapes: SceneShape[] = [grid('g', [0, 0], [20, 10]), player('in', 5, 5), player('out', 25, 5), { id: 'e', type: 'equip', kind: 'cone', pos: [19, 9], angle: 0 }]
  assert.deepEqual(shapesInGrid(shapes, 'g', bg).sort(), ['e', 'g', 'in'])
  const turned = shapes.map(s => (s.id === 'g' ? grid('g', [0, 0], [20, 10], 90) : s))
  // turned 90° around (10,5): the field now covers x 5..15, y -5..15; the player at (5,5) sits on its left edge
  assert.ok(shapesInGrid(turned, 'g', bg).includes('g')); assert.ok(!shapesInGrid(turned, 'g', bg).includes('out'))
  assert.deepEqual(shapesInGrid(shapes, 'nope', bg), [])
})
test('pasting clones with fresh ids and keeps attachments inside the pasted set', () => {
  const src: SceneShape[] = [player('p1', 2, 2), player('p2', 10, 2), line('l1', [2, 2], [10, 2], { fromId: 'p1', toId: 'p2' })]
  const c = cloneShapes(src, 1, 1)
  assert.equal(c.length, 3); assert.ok(c.every(s => !src.some(o => o.id === s.id)))
  const l = c.find(s => s.type === 'line') as Extract<SceneShape, { type: 'line' }>
  assert.ok(c.some(s => s.id === l.fromId) && c.some(s => s.id === l.toId)); assert.deepEqual((c[0] as { pos: number[] }).pos, [3, 3])
})
test('Feldaufbau: a form becomes a labelled field of its own size inside the frame', () => {
  const pitch = defaultBackground('pitch')
  const f = fieldShape(pitch, '1 · Over the River', 20, 30, 0, '#ef4444') as Extract<SceneShape, { type: 'grid' }>
  assert.equal(f.label, '1 · Over the River'); assert.ok(Math.abs(Math.abs(f.b[0] - f.a[0]) - 20) < 1e-9 && Math.abs(Math.abs(f.b[1] - f.a[1]) - 30) < 1e-9)
  const g = fieldShape(pitch, 'x', 20, 30, 1) as Extract<SceneShape, { type: 'grid' }>
  assert.notDeepEqual(g.a, f.a) // the next one is nudged
})

// split fields and corner cones
test('a field can be divided by cross lines; its parts have their own size', () => {
  const g = grid('g', [0, 0], [20, 15]) as Extract<SceneShape, { type: 'grid' }>
  assert.deepEqual(gridSplits(g), { cols: 1, rows: 1 }); assert.equal(gridCellText(g), '')
  const d = { ...g, cols: 2, rows: 3 } as typeof g
  assert.deepEqual(gridSplits(d), { cols: 2, rows: 3 }); assert.equal(gridCellText(d), 'je 10 × 5 m')
  assert.deepEqual(gridSplits({ ...g, cols: 99, rows: 0 } as typeof g), { cols: 8, rows: 1 })
  assert.equal(gridCellText({ ...g, cols: 3 } as typeof g), 'je 6,7 × 15 m')
})
test('cones go on the corners, and on the dividing lines if asked, also when the field is turned', () => {
  const g = { ...grid('g', [0, 0], [20, 10]), cols: 2, rows: 2 } as Extract<SceneShape, { type: 'grid' }>
  assert.equal(gridLattice(g, false).length, 4); assert.equal(gridLattice(g, true).length, 9)
  const r = addFieldCones([g], 'g', bg, false)
  assert.equal(r.added.length, 4)
  assert.ok(r.shapes.filter(s => s.type === 'equip').every(s => (s as { kind: string }).kind === 'cone'))
  const pos = r.shapes.filter(s => s.type === 'equip').map(s => (s as { pos: number[] }).pos)
  for (const c of [[0, 0], [20, 0], [20, 10], [0, 10]]) assert.ok(pos.some(p => near(p, c)))
  const again = addFieldCones(r.shapes, 'g', bg, false); assert.equal(again.added.length, 0) // nothing doubled
  const more = addFieldCones(r.shapes, 'g', bg, true); assert.equal(more.added.length, 5) // only the 5 new spots
  const turned = { ...g, angle: 90 } as typeof g
  const lat = gridLattice(turned, false)
  assert.ok(lat.some(p => near(p, rotPt([0, 0], [10, 5], 90))))
  assert.deepEqual(addFieldCones([player('p', 1, 1)], 'nope', bg, false).added, [])
})

test('cones placed on a field\'s corners count as part of its content', () => {
  const g = { ...grid('g', [3, 3], [17, 13], 37), cols: 2 } as SceneShape
  const r = addFieldCones([g], 'g', bg, false)
  assert.equal(shapesInGrid(r.shapes, 'g', bg).length, 5)
})

test('cones keep their orientation when a group is turned; goals turn', () => {
  const shapes: SceneShape[] = [{ id: 'c', type: 'equip', kind: 'cone', pos: [10, 0], angle: 0 }, { id: 'g', type: 'equip', kind: 'goal', pos: [0, 10], angle: 0 }]
  const out = rotateShapes(shapes, ['c', 'g'], 90, [0, 0])
  assert.equal((out[0] as { angle: number }).angle, 0); assert.ok(near((out[0] as { pos: number[] }).pos, [0, 10]))
  assert.equal((out[1] as { angle: number }).angle, 90)
})
