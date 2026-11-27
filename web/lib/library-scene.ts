import type { SceneShape } from './scene.ts'

/** A sketch counts as drawn when it holds at least one player, ball, piece of equipment, field or line. */
export function isMeaningfulScene(shapes: SceneShape[] | undefined | null): boolean {
  return (shapes ?? []).some(s => s.type !== 'text' && s.type !== 'dim')
}
