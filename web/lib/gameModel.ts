// Das Spielmodell von ASN Pfeil Phönix als Ordnungsstruktur: Phase → Prinzip / Konzept.
// Forms and sessions store only the stable ids below, so names can be changed here without touching saved data.

export type PhaseId = 'ballbesitz' | 'ohneball' | 'umschalten-verteidigung' | 'umschalten-angriff'
export type ConceptKind = 'prinzip' | 'konzept'

export interface Phase { id: PhaseId; name: string; motto?: string; acronym?: string }
export interface Concept {
  id: string
  phase: PhaseId
  kind: ConceptKind
  /** Prinzip: none. Konzept: the zone of the map it sits in (Aufbau / Kreieren / Abschluss, or the block when defending). */
  group: string
  name: string
  hint?: string
}

export const PHASES: Phase[] = [
  { id: 'ballbesitz', name: 'In Ballbesitz', motto: 'Kontrolle, Konzentration, Kreativität', acronym: 'TRAUMTOR' },
  { id: 'ohneball', name: 'Ohne Ball', motto: 'Verbieten, Verzögern, Vorgeben', acronym: 'KOMPAKT' },
  { id: 'umschalten-verteidigung', name: 'Übergang zur Verteidigung' },
  { id: 'umschalten-angriff', name: 'Übergang zum Angriff' },
]

const P = 'Prinzipien'
const c = (id: string, phase: PhaseId, kind: ConceptKind, group: string, name: string, hint?: string): Concept => ({ id, phase, kind, group, name, hint })

export const CONCEPTS: Concept[] = [
  // ── In Ballbesitz: Prinzipien (TRAUMTOR) ──
  c('bb-p-tempo', 'ballbesitz', 'prinzip', P, 'Tempo kontrollieren', 'Erkennen, wann wir das Tempo verlangsamen und wann wir es erhöhen sollten.'),
  c('bb-p-raum', 'ballbesitz', 'prinzip', P, 'Raum schaffen', 'Das Spielfeld maximal nutzen. Breite und Tiefe ausreizen.'),
  c('bb-p-zusammen', 'ballbesitz', 'prinzip', P, 'Alles zusammen', 'Die Mannschaft bewegt sich mit dem Ball von Drittel zu Drittel gemeinsam nach vorne.'),
  c('bb-p-ueberzahl', 'ballbesitz', 'prinzip', P, 'Überzahl erzeugen', 'Bewegung ohne Ball, Platz schaffen, Passwege öffnen, Gegner aus der Ordnung bringen.'),
  c('bb-p-mutig', 'ballbesitz', 'prinzip', P, 'Mutig sein', 'Den Ball immer fordern. Wenn möglich nach vorne spielen.'),
  c('bb-p-taktisch', 'ballbesitz', 'prinzip', P, 'Taktisch reagieren', 'Den Ballverlust antizipieren und so positioniert sein, dass wir angemessen reagieren können.'),
  // ── In Ballbesitz: Konzepte ──
  c('bb-k-hoehe-avs', 'ballbesitz', 'konzept', 'Aufbau', 'Höhe AVs'),
  c('bb-k-diagonale', 'ballbesitz', 'konzept', 'Aufbau', 'Diagonale Pässe'),
  c('bb-k-gegnerdruck', 'ballbesitz', 'konzept', 'Aufbau', 'Nicht ohne Gegnerdruck spielen'),
  c('bb-k-breite-ivs', 'ballbesitz', 'konzept', 'Aufbau', 'Breite IVs'),
  c('bb-k-tw-ueberzahl', 'ballbesitz', 'konzept', 'Aufbau', 'TW nutzen, um Überzahl im Aufbau zu erzeugen'),
  c('bb-k-druck-aufsaugen', 'ballbesitz', 'konzept', 'Aufbau', 'Druck aufsaugen, um hinten Raum zu schaffen'),
  c('bb-k-breite-nutzen', 'ballbesitz', 'konzept', 'Aufbau', 'Breite nutzen'),
  c('bb-k-scannen', 'ballbesitz', 'konzept', 'Aufbau', 'Scannen, Bewegung, Körperhaltung, um zu annehmen'),
  c('bb-k-verlagerung', 'ballbesitz', 'konzept', 'Aufbau', 'Spielverlagerung nach Überladung'),
  c('bb-k-durch-ketten', 'ballbesitz', 'konzept', 'Kreieren', 'Durch die Ketten spielen'),
  c('bb-k-raute', 'ballbesitz', 'konzept', 'Kreieren', 'Raute um den Ballträger'),
  c('bb-k-dritter', 'ballbesitz', 'konzept', 'Kreieren', 'Spiel über den Dritten'),
  c('bb-k-max2', 'ballbesitz', 'konzept', 'Kreieren', 'Max. 2 Spieler in derselben vertikalen Zone'),
  c('bb-k-bewegung-ketten', 'ballbesitz', 'konzept', 'Kreieren', 'Ständige Bewegung zwischen den Ketten'),
  c('bb-k-dreiecke', 'ballbesitz', 'konzept', 'Kreieren', 'Dreiecke in der Breite'),
  c('bb-k-rotieren', 'ballbesitz', 'konzept', 'Kreieren', 'Positionen rotieren'),
  c('bb-k-tiefe', 'ballbesitz', 'konzept', 'Kreieren', 'Tiefe nutzen'),
  c('bb-k-1gg1', 'ballbesitz', 'konzept', 'Abschluss', 'Ins 1gg1 kommen'),
  c('bb-k-halbraum', 'ballbesitz', 'konzept', 'Abschluss', 'Halbraum-Flanken'),
  c('bb-k-strafraum', 'ballbesitz', 'konzept', 'Abschluss', 'Strafraumbesetzung'),
  c('bb-k-laeufe-tiefe', 'ballbesitz', 'konzept', 'Abschluss', 'Läufe in die Tiefe'),
  c('bb-k-flanken-tw', 'ballbesitz', 'konzept', 'Abschluss', 'Flanken zwischen Abwehrreihe und Torwart'),
  c('bb-k-bahnen', 'ballbesitz', 'konzept', 'Abschluss', 'Alle Bahnen besetzen'),
  // ── Ohne Ball: Prinzipien (KOMPAKT) ──
  c('ob-p-kommunikation', 'ohneball', 'prinzip', P, 'Kommunikation', 'Klare Anweisungen stärken die Entschlossenheit.'),
  c('ob-p-pressen', 'ohneball', 'prinzip', P, 'Organisiert pressen', 'Ball und Spieler um den Ball unter Druck setzen. Pressing nur als Team.'),
  c('ob-p-deckung', 'ohneball', 'prinzip', P, 'Mit Deckung und Balance', 'Pässe und Laufwege des Gegners blockieren, um Raum und Tor zu verteidigen.'),
  c('ob-p-platz', 'ohneball', 'prinzip', P, 'Platz beschränken', 'Kompakt um den Ball, damit der Gegner weniger Optionen hat.'),
  c('ob-p-aufmerksam', 'ohneball', 'prinzip', P, 'Aufmerksamkeit und Zurückhaltung', 'Ruhe bewahren, die Lage präzise analysieren.'),
  c('ob-p-kontrolle', 'ohneball', 'prinzip', P, 'Kontrolle und Ballgewinn', 'Druck ausüben, um den Ball zu gewinnen und zu kontern.'),
  c('ob-p-team', 'ohneball', 'prinzip', P, 'Team', 'Wir können nur als Mannschaft gut verteidigen.'),
  // ── Ohne Ball: Konzepte, nach Block ──
  c('ob-k-zentrum', 'ohneball', 'konzept', 'Tiefer Block', 'Zentrum schützen'),
  c('ob-k-rausschieben', 'ohneball', 'konzept', 'Tiefer Block', 'Verteidiger schieben raus, wenn geklärt wurde'),
  c('ob-k-fuessen', 'ohneball', 'konzept', 'Tiefer Block', 'Auf den Füßen bleiben'),
  c('ob-k-viele', 'ohneball', 'konzept', 'Tiefer Block', 'Mit vielen Spielern verteidigen'),
  c('ob-k-keine-schuesse', 'ohneball', 'konzept', 'Tiefer Block', 'Keine Schüsse zulassen'),
  c('ob-k-flanken', 'ohneball', 'konzept', 'Tiefer Block', 'Flanken verhindern'),
  c('ob-k-aussen', 'ohneball', 'konzept', 'Tiefer Block', 'Nach außen drängen'),
  c('ob-k-schieben', 'ohneball', 'konzept', 'Mittel-Block', 'Ball-orientiertes Schieben (3 Bahnen besetzen)'),
  c('ob-k-abstaende', 'ohneball', 'konzept', 'Mittel-Block', 'Horizontale und vertikale Abstände minimieren'),
  c('ob-k-letzte-linie', 'ohneball', 'konzept', 'Mittel-Block', 'Letzte Linie ist immer +1'),
  c('ob-k-deckung-balance', 'ohneball', 'konzept', 'Mittel-Block', 'Deckung und Balance'),
  c('ob-k-gegenpressing', 'ohneball', 'konzept', 'Mittel-Block', 'Aggressives Gegenpressing'),
  c('ob-k-risiko', 'ohneball', 'konzept', 'Hohes Pressing', 'Hohes Risiko, hohe Belohnung'),
  c('ob-k-pressing-team', 'ohneball', 'konzept', 'Hohes Pressing', 'Pressing als Team'),
  c('ob-k-blocken', 'ohneball', 'konzept', 'Hohes Pressing', 'Blocken vor Pressing'),
  c('ob-k-pressing-faelle', 'ohneball', 'konzept', 'Hohes Pressing', 'Pressing-Fälle'),
  c('ob-k-ausloeser', 'ohneball', 'konzept', 'Hohes Pressing', 'Pressing-Auslöser'),
  c('ob-k-deckungsschatten', 'ohneball', 'konzept', 'Hohes Pressing', 'Deckungsschatten nutzen'),
  c('ob-k-tw-zumachen', 'ohneball', 'konzept', 'Hohes Pressing', 'TW zumachen'),
  c('ob-k-ball-pressing', 'ohneball', 'konzept', 'Hohes Pressing', 'Ball-orientiertes Pressing'),
  c('ob-k-aussenlinie', 'ohneball', 'konzept', 'Hohes Pressing', 'Außenlinie als zusätzlichen Verteidiger nutzen'),
  // ── Übergänge: noch keine Prinzipien, daher je ein allgemeiner Eintrag ──
  c('uv-allgemein', 'umschalten-verteidigung', 'konzept', 'Allgemein', 'Umschalten nach Ballverlust', 'Prinzipien und Konzepte dieser Phase sind noch offen.'),
  c('ua-allgemein', 'umschalten-angriff', 'konzept', 'Allgemein', 'Umschalten nach Ballgewinn', 'Prinzipien und Konzepte dieser Phase sind noch offen.'),
]

export const MAX_CONCEPTS = 6
const byId = new Map(CONCEPTS.map(x => [x.id, x]))

export const conceptById = (id: string): Concept | undefined => byId.get(id)
export const phaseById = (id: PhaseId): Phase => PHASES.find(p => p.id === id)!

/** Keep known ids only, without duplicates, in the given order (the first is the main concept), at most MAX_CONCEPTS. */
export function normalizeConceptIds(ids: readonly string[] | null | undefined): string[] {
  const out: string[] = []
  for (const id of ids ?? []) if (byId.has(id) && !out.includes(id)) out.push(id)
  return out.slice(0, MAX_CONCEPTS)
}

/** Toggle an id: add at the end (up to MAX_CONCEPTS), or remove. */
export function toggleConcept(ids: readonly string[], id: string): string[] {
  const cur = normalizeConceptIds(ids)
  return cur.includes(id) ? cur.filter(x => x !== id) : normalizeConceptIds([...cur, id])
}

/** Make `id` the main concept (first). */
export function makeMain(ids: readonly string[], id: string): string[] {
  const cur = normalizeConceptIds(ids)
  return cur.includes(id) ? [id, ...cur.filter(x => x !== id)] : cur
}

/** The phases an item touches, in model order. */
export function phasesOf(ids: readonly string[]): Phase[] {
  const used = new Set(normalizeConceptIds(ids).map(i => byId.get(i)!.phase))
  return PHASES.filter(p => used.has(p.id))
}

/** Concepts of a phase grouped for display: Prinzipien first, then the Konzepte by zone in map order. */
export function groupsOfPhase(phase: PhaseId): { group: string; items: Concept[] }[] {
  const out: { group: string; items: Concept[] }[] = []
  for (const x of CONCEPTS) {
    if (x.phase !== phase) continue
    let g = out.find(o => o.group === x.group)
    if (!g) { g = { group: x.group, items: [] }; out.push(g) }
    g.items.push(x)
  }
  return out
}

/** Does an item with `ids` belong to the filter? A phase filter matches any concept of that phase. */
export function matchesFilter(ids: readonly string[], f: { phase?: PhaseId | null; concept?: string | null }): boolean {
  const cur = normalizeConceptIds(ids)
  if (f.concept) return cur.includes(f.concept)
  if (f.phase) return cur.some(i => byId.get(i)!.phase === f.phase)
  return true
}
