// The library: which forms count as complete enough to propose, and how a library form is copied into a session.
import type { FormSceneRow, LibraryStatus, SessionForm } from './sessions.ts'
import { isMeaningfulScene } from './library-scene.ts'

export const STATUS_LABELS: Record<LibraryStatus, string> = {
  private: 'Entwurf',
  proposed: 'Zur Aufnahme vorgeschlagen',
  library: 'In der Bibliothek',
  archived: 'Archiviert',
}

export type Check = { key: string; label: string; ok: boolean }

const filled = (s: string | undefined | null) => !!s && s.trim().length > 0
export const DEFAULT_TITLES = ['Neue Form', 'Neue Übung', 'Neue Spielform', '']

/** What a form still needs before it can be proposed. Every check must be ok. */
export function completeness(form: Pick<SessionForm, 'title' | 'duration_min' | 'players' | 'sections' | 'concepts'>, scenes: Pick<FormSceneRow, 'shapes'>[]): Check[] {
  const written = (form.sections ?? []).filter(s => filled(s.text)).length
  return [
    { key: 'title', label: 'Eigener Titel', ok: filled(form.title) && !DEFAULT_TITLES.includes(form.title.trim()) },
    { key: 'concept', label: 'Mindestens ein Konzept oder Prinzip aus dem Spielmodell', ok: (form.concepts ?? []).length > 0 },
    { key: 'sketch', label: 'Mindestens eine Skizze mit Inhalt', ok: scenes.some(s => isMeaningfulScene(s.shapes)) },
    { key: 'duration', label: 'Dauer', ok: !!form.duration_min },
    { key: 'players', label: 'Spieler und Gruppen', ok: filled(form.players) },
    { key: 'text', label: 'Mindestens zwei beschriebene Abschnitte (z. B. Aufbau und Ablauf)', ok: written >= 2 },
  ]
}

export const missing = (checks: Check[]) => checks.filter(c => !c.ok)

/** The row to insert when a library form is taken into a session: private again, remembering where it came from. */
export function copyOfForm(src: SessionForm, sessionId: string, position: number): Record<string, unknown> {
  return {
    session_id: sessionId, position, kind: src.kind, title: src.title, duration_min: src.duration_min, players: src.players,
    sections: src.sections, concepts: src.concepts ?? [], library_source: src.library_source && src.library_status !== 'library' ? src.library_source : src.id,
  }
}

export function copyOfScene(src: FormSceneRow, formId: string): Record<string, unknown> {
  return { form_id: formId, position: src.position, background: src.background, shapes: src.shapes, caption: src.caption, ...(src.timing ? { timing: src.timing } : {}) }
}
