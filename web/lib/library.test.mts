import test from 'node:test'
import assert from 'node:assert/strict'
import { completeness, missing, copyOfForm, copyOfScene } from './library.ts'

const full = { title: 'Raute im Aufbau', duration_min: 15, players: '4 v 2', concepts: ['bb-k-raute'], sections: [{ id: 'a', title: 'Aufbau', text: 'x' }, { id: 'b', title: 'Ablauf', text: 'y' }] }
const drawn = [{ shapes: [{ id: 'p', type: 'player', pos: [1, 1], team: 'a' }] }] as never

test('completeness: a full form passes, each gap is named', () => {
  assert.equal(missing(completeness(full, drawn)).length, 0)
  const keys = (f: object, sc = drawn) => missing(completeness({ ...full, ...f }, sc)).map(c => c.key)
  assert.deepEqual(keys({ title: 'Neue Form' }), ['title'])
  assert.deepEqual(keys({ title: 'Neue Übung' }), ['title'])
  assert.deepEqual(keys({ title: ' Neue Spielform ' }), ['title'])
  assert.deepEqual(keys({ concepts: [] }), ['concept'])
  assert.deepEqual(keys({ duration_min: null }), ['duration'])
  assert.deepEqual(keys({ players: '  ' }), ['players'])
  assert.deepEqual(keys({ sections: [full.sections[0], { id: 'c', title: 'Fragen', text: ' ' }] }), ['text'])
  assert.deepEqual(keys({}, [{ shapes: [{ id: 't', type: 'text', pos: [0, 0], text: 'x', size: 1 }] }] as never), ['sketch'])
  assert.deepEqual(keys({}, []), ['sketch'])
})

test('copies: private session form remembers its source', () => {
  const src = { ...full, id: 'F1', session_id: 'S', position: 0, kind: 'exercise', sections: full.sections, library_status: 'library', library_source: null } as never
  const c = copyOfForm(src, 'S2', 3) as Record<string, unknown>
  assert.equal(c.session_id, 'S2'); assert.equal(c.position, 3); assert.equal(c.library_source, 'F1')
  assert.ok(!('library_status' in c))
  const sc = copyOfScene({ id: 'x', form_id: 'F1', position: 1, background: { type: 'free' }, shapes: [], caption: 'c', timing: { move_s: 1 } } as never, 'F2') as Record<string, unknown>
  assert.equal(sc.form_id, 'F2'); assert.deepEqual(sc.timing, { move_s: 1 })
})
