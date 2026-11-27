import test from 'node:test'
import assert from 'node:assert/strict'
import { CONCEPTS, PHASES, normalizeConceptIds, toggleConcept, makeMain, phasesOf, groupsOfPhase, matchesFilter, MAX_CONCEPTS } from './gameModel.ts'

test('game model: ids are unique and every phase has entries', () => {
  assert.equal(new Set(CONCEPTS.map(c => c.id)).size, CONCEPTS.length)
  for (const p of PHASES) assert.ok(CONCEPTS.some(c => c.phase === p.id), p.id)
  assert.equal(CONCEPTS.filter(c => c.phase === 'ballbesitz' && c.kind === 'prinzip').length, 6)
  assert.equal(CONCEPTS.filter(c => c.phase === 'ohneball' && c.kind === 'prinzip').length, 7)
})

test('normalize, toggle and main concept', () => {
  assert.deepEqual(normalizeConceptIds(['bb-k-dritter', 'nope', 'bb-k-dritter', 'bb-p-tempo']), ['bb-k-dritter', 'bb-p-tempo'])
  assert.deepEqual(normalizeConceptIds(null), [])
  let ids: string[] = []
  ids = toggleConcept(ids, 'bb-k-raute'); ids = toggleConcept(ids, 'bb-k-dritter')
  assert.deepEqual(ids, ['bb-k-raute', 'bb-k-dritter'])
  assert.deepEqual(makeMain(ids, 'bb-k-dritter'), ['bb-k-dritter', 'bb-k-raute'])
  assert.deepEqual(toggleConcept(ids, 'bb-k-raute'), ['bb-k-dritter'])
  const many = CONCEPTS.slice(0, 10).map(c => c.id).reduce<string[]>((a, id) => toggleConcept(a, id), [])
  assert.equal(many.length, MAX_CONCEPTS)
})

test('phases, groups and filter', () => {
  assert.deepEqual(phasesOf(['bb-k-raute', 'ob-k-zentrum', 'bb-p-tempo']).map(p => p.id), ['ballbesitz', 'ohneball'])
  const g = groupsOfPhase('ballbesitz').map(x => x.group)
  assert.deepEqual(g, ['Prinzipien', 'Aufbau', 'Kreieren', 'Abschluss'])
  assert.ok(matchesFilter(['bb-k-raute'], { phase: 'ballbesitz' }) && !matchesFilter(['bb-k-raute'], { phase: 'ohneball' }))
  assert.ok(matchesFilter(['bb-k-raute'], { concept: 'bb-k-raute' }) && !matchesFilter(['bb-k-raute'], { concept: 'bb-k-dritter' }))
  assert.ok(matchesFilter([], {}))
})
