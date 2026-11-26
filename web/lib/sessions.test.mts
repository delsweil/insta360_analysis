import test from 'node:test'
import assert from 'node:assert/strict'
import { parseTags, filterSessions, tagCounts, sortSessions, type TrainingSession } from './sessions.ts'

function mk(p: Partial<TrainingSession>): TrainingSession {
  return {
    id: 'x', title: 'T', session_date: null, focus: '', visibility: 'private', tags: [],
    created_by: 'me', created_at: '2026-01-01T10:00:00Z', updated_at: '2026-01-01T10:00:00Z', ...p,
  }
}

test('parseTags cleans, de-duplicates ignoring case, and strips #', () => {
  assert.deepEqual(parseTags('U13, #Pressing, u13 ; Spielform\n  '), ['U13', 'Pressing', 'Spielform'])
  assert.deepEqual(parseTags(''), [])
  assert.deepEqual(parseTags(' , ;; '), [])
})

test('parseTags limits length and count', () => {
  assert.equal(parseTags('x'.repeat(80))[0].length, 30)
  const many = Array.from({ length: 30 }, (_, i) => `t${i}`).join(',')
  assert.equal(parseTags(many).length, 12)
})

test('parseTags treats German umlauts case-insensitively', () => {
  assert.deepEqual(parseTags('Überzahl, überzahl'), ['Überzahl'])
})

test('filterSessions by scope, tag and text', () => {
  const a = mk({ id: 'a', title: 'Over the River', tags: ['U13', 'Pressing'], visibility: 'shared', created_by: 'other' })
  const b = mk({ id: 'b', title: 'Passspiel', tags: ['U15'], focus: 'Aufbau' })
  const l = [a, b]
  assert.deepEqual(filterSessions(l, { scope: 'mine', tag: null, userId: 'me' }).map(s => s.id), ['b'])
  assert.deepEqual(filterSessions(l, { scope: 'shared', tag: null, userId: 'me' }).map(s => s.id), ['a'])
  assert.deepEqual(filterSessions(l, { scope: 'all', tag: 'u13', userId: 'me' }).map(s => s.id), ['a'])
  assert.deepEqual(filterSessions(l, { scope: 'all', tag: null, userId: 'me', query: 'aufbau' }).map(s => s.id), ['b'])
  assert.deepEqual(filterSessions(l, { scope: 'all', tag: null, userId: 'me', query: 'RIVER' }).map(s => s.id), ['a'])
})

test('tagCounts merges case variants and sorts by use', () => {
  const l = [mk({ tags: ['U13', 'Pressing'] }), mk({ tags: ['u13'] }), mk({ tags: ['Aufbau'] })]
  assert.deepEqual(tagCounts(l), [{ tag: 'U13', count: 2 }, { tag: 'Aufbau', count: 1 }, { tag: 'Pressing', count: 1 }])
})

test('sortSessions: newest date first, undated by creation day', () => {
  const l = [
    mk({ id: '1', session_date: '2026-03-01' }),
    mk({ id: '2', session_date: null, created_at: '2026-05-01T08:00:00Z' }),
    mk({ id: '3', session_date: '2026-04-01' }),
  ]
  assert.deepEqual(sortSessions(l).map(s => s.id), ['2', '3', '1'])
})
