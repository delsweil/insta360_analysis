// Export for analysis: the coach's annotated events and, per moment of the tactical board, simple features of the
// detected players, so one can look for signals (compactness, pressing, team shape ...) that predict the events.
// Two CSV files: `<game>_events.csv` (one row per annotation) and `<game>_frames.csv` (one row per board moment).
import type { Annotation } from '@/lib/supabase'
import type { Board } from '@/lib/board'
import { playersOf, partAt, boxOf } from '@/lib/board'

const csvCell = (v: unknown) => {
  if (v === null || v === undefined) return ''
  const s = typeof v === 'number' ? (Number.isFinite(v) ? String(Math.round(v * 1000) / 1000) : '') : String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
const csv = (head: string[], rows: unknown[][]) =>
  [head.join(','), ...rows.map(r => r.map(csvCell).join(','))].join('\n') + '\n'

const mean = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)
const sd = (v: number[]) => {
  if (v.length < 2) return NaN
  const m = mean(v)
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length)
}

/** Events that are worth predicting come with a window: how long before it a signal may show. */
export const EVENT_LEAD_S = 20

export function eventsCsv(anns: Annotation[]): string {
  const rows = [...anns].sort((a, b) => a.timestamp_sec - b.timestamp_sec).map(a => {
    const m = a.label.match(/^(.*)_(home|away)$/)
    return [a.id, a.timestamp_sec, a.end_timestamp_sec ?? '', a.label, m ? m[1] : a.label, m ? m[2] : '', a.note ?? '']
  })
  return csv(['id', 'video_t', 'end_t', 'label', 'kind', 'team', 'note'], rows)
}

/** Features per board moment. `event_kind`/`event_in` give the next event within EVENT_LEAD_S seconds (and in how many). */
export function framesCsv(board: Board, anns: Annotation[]): string {
  const ev = [...anns].sort((a, b) => a.timestamp_sec - b.timestamp_sec)
  let k = 0
  const head = ['video_t', 'board_t', 'half', 'pan', 'fov', 'n_A', 'n_B', 'n_other', 'n_ref', 'n_GK_A', 'n_GK_B',
    'A_mean_x', 'A_mean_y', 'A_sd_x', 'A_sd_y', 'B_mean_x', 'B_mean_y', 'B_sd_x', 'B_sd_y',
    'A_x_min', 'A_x_max', 'B_x_min', 'B_x_max', 'A_attacks_right', 'next_event_kind', 'next_event_team', 'next_event_in_s']
  const rows = board.frames.map(f => {
    const t = f[0] as number
    const vt = t - board.shift
    const part = partAt(board, t)
    const [x0, x1, y0, y1] = boxOf(board, part)
    const cx = (x0 + x1) / 2
    const cy = (y0 + y1) / 2                      // x, y relative to the pitch centre, in metres
    const ps = playersOf(f)
    const of = (...teams: number[]) => ps.filter(p => teams.includes(p.team))
    const A = of(0, 4), B = of(1, 5)
    const ax = A.map(p => p.x - cx), ay = A.map(p => p.y - cy)
    const bx = B.map(p => p.x - cx), by = B.map(p => p.y - cy)
    while (k < ev.length && ev[k].timestamp_sec < vt) k++
    const nx = k < ev.length && ev[k].timestamp_sec - vt <= EVENT_LEAD_S ? ev[k] : null
    const m = nx ? nx.label.match(/^(.*)_(home|away)$/) : null
    return [vt, t, part?.half ?? '', f[1], f[2], A.length, B.length, of(2).length, of(3).length, of(4).length, of(5).length,
      mean(ax), mean(ay), sd(ax), sd(ay), mean(bx), mean(by), sd(bx), sd(by),
      ax.length ? Math.min(...ax) : NaN, ax.length ? Math.max(...ax) : NaN,
      bx.length ? Math.min(...bx) : NaN, bx.length ? Math.max(...bx) : NaN,
      part?.a_attacks_right === null || part?.a_attacks_right === undefined ? '' : part.a_attacks_right ? 1 : 0,
      nx ? (m ? m[1] : nx.label) : '', nx ? (m ? m[2] : '') : '', nx ? nx.timestamp_sec - vt : '']
  })
  return csv(head, rows)
}

export function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function exportGame(title: string, anns: Annotation[], board: Board | null) {
  const base = title.replace(/[^\w\-]+/g, '_') || 'game'
  download(`${base}_events.csv`, eventsCsv(anns))
  if (board) setTimeout(() => download(`${base}_frames.csv`, framesCsv(board, anns)), 300)
}
