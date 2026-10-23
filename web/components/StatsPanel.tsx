'use client'

import { useEffect, useId, useMemo, useRef, useState } from 'react'
import { type HalfStats, type Stats, type TeamSeries } from '@/lib/stats'
import { PAD, Pitch } from '@/components/BoardPanel'

const NAVY = '#0f2972'
const ORANGE = '#E8780A'

type Theme = { card: string; border: string; text: string; muted: string; faint: string; grid: string; chip: string; chipOn: string; chipOnText: string }
const LIGHT: Theme = { card: '#fff', border: '#E4E6EE', text: '#111318', muted: '#4A4F5C', faint: '#8A8F9E', grid: '#E9ECF3', chip: '#F1F3F9', chipOn: NAVY, chipOnText: '#fff' }
const DARK: Theme = { card: 'rgba(255,255,255,0.06)', border: 'rgba(255,255,255,0.12)', text: '#fff', muted: 'rgba(255,255,255,0.75)', faint: 'rgba(255,255,255,0.45)', grid: 'rgba(255,255,255,0.1)', chip: 'rgba(255,255,255,0.08)', chipOn: ORANGE, chipOnText: '#fff' }

const fmtNum = (v: number, d = 1) => v.toLocaleString('de-DE', { minimumFractionDigits: d, maximumFractionDigits: d })
const clock = (min: number) => `${Math.floor(min)}′`
const mean = (a: (number | null)[]) => {
  const v = a.filter((x): x is number => x != null)
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null
}
/** Centred moving average over `k` values (ignores gaps) so a per-minute line reads as a trend. */
function smooth(a: (number | null)[], k = 3) {
  const h = Math.floor(k / 2)
  return a.map((_, i) => {
    const w = a.slice(Math.max(0, i - h), i + h + 1).filter((x): x is number => x != null)
    return a[i] == null || !w.length ? null : w.reduce((s, x) => s + x, 0) / w.length
  })
}

/** A team colour that can be seen on the card: near-white shirts are darkened on white, near-black ones lightened on dark. */
export function readable(c: string, dark: boolean): string {
  const m = c.match(/^#?([0-9a-f]{6})$/i)
  if (!m) return c
  const n = parseInt(m[1], 16)
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
  const mix = (t: number, to: number) => '#' + [r, g, b].map(v => Math.round(v + (to - v) * t).toString(16).padStart(2, '0')).join('')
  if (!dark && lum > 0.78) return mix(0.5, 90)
  if (dark && lum < 0.2) return mix(0.5, 200)
  return c
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(Math.round(el.getBoundingClientRect().width)))
    ro.observe(el)
    setW(Math.round(el.getBoundingClientRect().width))
    return () => ro.disconnect()
  }, [])
  return [ref, w] as const
}

function Dot({ c, size = 9 }: { c: string; size?: number }) {
  return <i style={{ display: 'inline-block', width: size, height: size, borderRadius: '50%', background: c, flexShrink: 0 }} />
}

function Card({ th, title, note, right, children }: { th: Theme; title: string; note?: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section style={{ background: th.card, border: `1px solid ${th.border}`, borderRadius: 12, padding: '12px 14px', minWidth: 0 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8, marginBottom: note ? 2 : 8 }}>
        <h3 style={{ margin: 0, fontFamily: 'Bebas Neue, sans-serif', fontWeight: 400, fontSize: 17, letterSpacing: '0.04em', color: th.text === '#fff' ? '#fff' : NAVY }}>{title}</h3>
        {right}
      </div>
      {note && <div style={{ fontSize: 11, color: th.faint, marginBottom: 8, lineHeight: 1.4 }}>{note}</div>}
      {children}
    </section>
  )
}

/** Share of the game each team had the play in the opponent's half. */
function TerritoryBar({ th, h }: { th: Theme; h: HalfStats }) {
  const vals = h.territory.value.filter((v): v is number => v != null)
  const a = vals.length ? vals.filter(v => v > 0).length / vals.length : 0.5
  const A = h.teams.A
  const B = h.teams.B
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, color: th.text }}><Dot c={A.colour} />{A.name}</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600, color: th.text }}>{B.name}<Dot c={B.colour} /></span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', margin: '2px 0 6px' }}>
        <span style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 34, lineHeight: 1, color: th.text }}>{Math.round(a * 100)}<small style={{ fontSize: 18 }}>%</small></span>
        <span style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 34, lineHeight: 1, color: th.text }}>{Math.round((1 - a) * 100)}<small style={{ fontSize: 18 }}>%</small></span>
      </div>
      <div style={{ display: 'flex', height: 10, borderRadius: 5, overflow: 'hidden', gap: 2 }} role="img" aria-label={`Territorium ${Math.round(a * 100)} zu ${Math.round((1 - a) * 100)} Prozent`}>
        <div style={{ width: `${a * 100}%`, background: A.colour, transition: 'width 0.4s' }} />
        <div style={{ flex: 1, background: B.colour }} />
      </div>
    </div>
  )
}

/** A small stat: label, then both teams' numbers. */
type TeamValues = TeamSeries & { values: (number | null)[] }

function Tile({ th, label, hint, a, b, unit, better }: { th: Theme; label: string; hint: string; a: TeamValues; b: TeamValues; unit: string; better?: 'low' | 'high' }) {
  const va = mean(a.values)
  const vb = mean(b.values)
  const win = better && va != null && vb != null && Math.abs(va - vb) > 0.05
    ? ((better === 'low' ? va < vb : va > vb) ? 'A' : 'B') : null
  const num = (v: number | null, c: string, who: 'A' | 'B') => (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 26, lineHeight: 1.05, color: th.text, opacity: win && win !== who ? 0.6 : 1 }}>
        {v == null ? '–' : fmtNum(v)}<small style={{ fontSize: 13, marginLeft: 2, color: th.faint }}>{unit}</small>
      </div>
      <div style={{ height: 3, borderRadius: 2, background: c, width: 26, marginTop: 2 }} />
    </div>
  )
  return (
    <div style={{ background: th.card, border: `1px solid ${th.border}`, borderRadius: 12, padding: '10px 12px', minWidth: 0 }} title={hint}>
      <div style={{ fontSize: 11, fontWeight: 600, color: th.muted, marginBottom: 6 }}>{label}</div>
      <div style={{ display: 'flex', gap: 10 }}>{num(va, a.colour, 'A')}{num(vb, b.colour, 'B')}</div>
    </div>
  )
}

/** Who had the play in whose half, minute by minute. Tap or click a moment to jump there in the video. */
function Momentum({ th, h, videoStart, now, onSeek }: { th: Theme; h: HalfStats; videoStart: number | null; now: number; onSeek: (s: number) => void }) {
  const [ref, w] = useWidth()
  const [hover, setHover] = useState<number | null>(null)
  const { minutes, value } = h.territory
  const pts = minutes.map((m, i) => [m, value[i]] as const).filter((p): p is readonly [number, number] => p[0] != null && p[1] != null)
  const H = 150
  const L = 8, R = 8, T = 18, Bm = 20
  const end = Math.max(h.duration_s / 60, pts.length ? pts[pts.length - 1][0] : 1, 1)
  const x = (m: number) => L + (m / end) * Math.max(1, w - L - R)
  const y = (v: number) => T + ((1 - v) / 2) * (H - T - Bm)
  const y0 = y(0)
  const path = pts.length ? `M${x(pts[0][0])},${y0} ${pts.map(p => `L${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ')} L${x(pts[pts.length - 1][0])},${y0} Z` : ''
  const ticks: number[] = []
  const step = end > 40 ? 15 : 10
  for (let m = 0; m <= end; m += step) ticks.push(m)
  const idAbove = `ab${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const idBelow = idAbove + 'b'
  const nowMin = videoStart != null && now >= videoStart && now <= videoStart + h.duration_s + 5 ? (now - videoStart) / 60 : null

  const minuteAt = (clientX: number, el: Element) => {
    const r = el.getBoundingClientRect()
    return Math.max(0, Math.min(end, ((clientX - r.left - L) / Math.max(1, r.width - L - R)) * end))
  }
  const valueAt = (m: number) => {
    let best = pts[0]
    for (const p of pts) if (Math.abs(p[0] - m) < Math.abs(best[0] - m)) best = p
    return best
  }
  const hv = hover != null && pts.length ? valueAt(hover) : null

  return (
    <div ref={ref} style={{ width: '100%', position: 'relative' }}>
      {w > 0 && pts.length > 1 && (
        <svg
          width={w} height={H} style={{ display: 'block', touchAction: 'pan-y', cursor: videoStart != null ? 'pointer' : 'default' }}
          onPointerMove={e => setHover(minuteAt(e.clientX, e.currentTarget))}
          onPointerLeave={() => setHover(null)}
          onClick={e => { if (videoStart != null) onSeek(videoStart + minuteAt(e.clientX, e.currentTarget) * 60) }}
          role="img" aria-label="Verlauf: welche Mannschaft das Spiel in der gegnerischen Hälfte hatte"
        >
          <defs>
            <clipPath id={idAbove}><rect x={0} y={0} width={w} height={y0} /></clipPath>
            <clipPath id={idBelow}><rect x={0} y={y0} width={w} height={H - y0} /></clipPath>
          </defs>
          {ticks.map(m => (
            <g key={m}>
              <line x1={x(m)} x2={x(m)} y1={T} y2={H - Bm} stroke={th.grid} strokeWidth={1} />
              <text x={x(m)} y={H - 5} fontSize={10} fill={th.faint} textAnchor={m === 0 ? 'start' : 'middle'}>{m}′</text>
            </g>
          ))}
          <path d={path} fill={h.teams.A.colour} fillOpacity={0.9} clipPath={`url(#${idAbove})`} />
          <path d={path} fill={h.teams.B.colour} fillOpacity={0.9} clipPath={`url(#${idBelow})`} />
          <line x1={L} x2={w - R} y1={y0} y2={y0} stroke={th.faint} strokeWidth={1} />
          <text x={L + 2} y={12} fontSize={10} fontWeight={600} fill={th.muted}>▲ {h.teams.A.name} im Vorwärtsgang</text>
          <text x={L + 2} y={H - Bm - 4} fontSize={10} fontWeight={600} fill={th.muted}>▼ {h.teams.B.name}</text>
          {nowMin != null && <line x1={x(nowMin)} x2={x(nowMin)} y1={T} y2={H - Bm} stroke={ORANGE} strokeWidth={2} />}
          {hover != null && hv && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - Bm} stroke={th.text} strokeOpacity={0.55} strokeWidth={1} />
            </g>
          )}
        </svg>
      )}
      {hv && hover != null && (
        <div style={{
          position: 'absolute', top: 0, left: Math.max(0, Math.min(w - 150, x(hover) - 75)), width: 150, textAlign: 'center', pointerEvents: 'none',
          fontSize: 11, fontWeight: 600, color: th.text, background: th.card === '#fff' ? 'rgba(255,255,255,0.95)' : 'rgba(10,16,32,0.92)',
          border: `1px solid ${th.border}`, borderRadius: 6, padding: '2px 6px',
        }}>
          {clock(hover)} · {Math.abs(hv[1]) < 0.05 ? 'ausgeglichen' : `${hv[1] > 0 ? h.teams.A.name : h.teams.B.name} ${Math.round(Math.abs(hv[1]) * 100)} %`}
        </div>
      )}
      {videoStart != null && <div style={{ fontSize: 10, color: th.faint, marginTop: 2 }}>Antippen springt im Video zu dieser Minute.</div>}
    </div>
  )
}

const METRICS = [
  { key: 'spread_m', label: 'Kompaktheit', unit: 'm', note: 'Mittlerer Abstand der Feldspieler zum Mannschaftsschwerpunkt. Kleiner = enger zusammen.', scale: 1 },
  { key: 'width_m', label: 'Breite', unit: 'm', note: 'Wie breit die Mannschaft das Feld macht (Querrichtung).', scale: 1 },
  { key: 'depth_m', label: 'Tiefe', unit: 'm', note: 'Wie weit die Mannschaft in Längsrichtung auseinandergezogen ist.', scale: 1 },
  { key: 'tempo_ms', label: 'Tempo', unit: 'km/h', note: 'Mittlere Laufgeschwindigkeit der Spieler.', scale: 3.6 },
] as const
type MetricKey = typeof METRICS[number]['key']

function ShapeChart({ th, h, metric }: { th: Theme; h: HalfStats; metric: typeof METRICS[number] }) {
  const [ref, w] = useWidth()
  const [hover, setHover] = useState<number | null>(null)
  const sc = metric.scale
  const sa = smooth(h.teams.A[metric.key]).map(v => (v == null ? null : v * sc))
  const sb = smooth(h.teams.B[metric.key]).map(v => (v == null ? null : v * sc))
  const n = Math.max(sa.length, sb.length)
  const all = [...sa, ...sb].filter((v): v is number => v != null)
  const H = 150, L = 34, R = 8, T = 10, Bm = 20
  if (n < 2 || !all.length) return <div style={{ fontSize: 12, color: th.faint }}>Keine Daten.</div>
  let lo = Math.min(...all), hi = Math.max(...all)
  const pad = (hi - lo || 1) * 0.15
  lo -= pad; hi += pad
  const x = (i: number) => L + (i / (n - 1)) * Math.max(1, w - L - R)
  const y = (v: number) => T + ((hi - v) / (hi - lo)) * (H - T - Bm)
  const line = (a: (number | null)[]) => {
    let d = ''
    let pen = false
    a.forEach((v, i) => { if (v == null) { pen = false; return } d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `; pen = true })
    return d
  }
  const yt = [lo + pad, (lo + hi) / 2, hi - pad]
  const xt: number[] = []
  const step = n > 40 ? 15 : 10
  for (let m = 0; m < n; m += step) xt.push(m)
  const at = hover != null ? Math.max(0, Math.min(n - 1, hover)) : null
  return (
    <div ref={ref} style={{ width: '100%', position: 'relative' }}>
      {w > 0 && (
        <svg
          width={w} height={H} style={{ display: 'block', touchAction: 'pan-y' }}
          onPointerMove={e => {
            const r = e.currentTarget.getBoundingClientRect()
            setHover(Math.round(((e.clientX - r.left - L) / Math.max(1, r.width - L - R)) * (n - 1)))
          }}
          onPointerLeave={() => setHover(null)}
          role="img" aria-label={`${metric.label} pro Minute`}
        >
          {yt.map((v, i) => (
            <g key={i}>
              <line x1={L} x2={w - R} y1={y(v)} y2={y(v)} stroke={th.grid} strokeWidth={1} />
              <text x={L - 5} y={y(v) + 3} fontSize={10} fill={th.faint} textAnchor="end">{fmtNum(v, metric.key === 'tempo_ms' ? 1 : 0)}</text>
            </g>
          ))}
          {xt.map(m => <text key={m} x={x(m)} y={H - 5} fontSize={10} fill={th.faint} textAnchor={m === 0 ? 'start' : 'middle'}>{m}′</text>)}
          <path d={line(sb)} fill="none" stroke={h.teams.B.colour} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
          <path d={line(sa)} fill="none" stroke={h.teams.A.colour} strokeWidth={2.4} strokeLinejoin="round" strokeLinecap="round" />
          {at != null && (
            <g>
              <line x1={x(at)} x2={x(at)} y1={T} y2={H - Bm} stroke={th.text} strokeOpacity={0.4} />
              {sa[at] != null && <circle cx={x(at)} cy={y(sa[at] as number)} r={4} fill={h.teams.A.colour} stroke={th.card === '#fff' ? '#fff' : '#0a1020'} strokeWidth={1.5} />}
              {sb[at] != null && <circle cx={x(at)} cy={y(sb[at] as number)} r={4} fill={h.teams.B.colour} stroke={th.card === '#fff' ? '#fff' : '#0a1020'} strokeWidth={1.5} />}
            </g>
          )}
        </svg>
      )}
      {at != null && (
        <div style={{ position: 'absolute', top: 0, left: Math.max(0, Math.min(w - 170, x(at) - 85)), width: 170, textAlign: 'center', pointerEvents: 'none', fontSize: 11, fontWeight: 600, color: th.text, background: th.card === '#fff' ? 'rgba(255,255,255,0.95)' : 'rgba(10,16,32,0.92)', border: `1px solid ${th.border}`, borderRadius: 6, padding: '2px 6px' }}>
          {at}′ · {sa[at] != null ? fmtNum(sa[at] as number) : '–'} / {sb[at] != null ? fmtNum(sb[at] as number) : '–'} {metric.unit}
        </div>
      )}
    </div>
  )
}

/** Where the play was, on the pitch as the camera saw it. */
function HeatMap({ h }: { h: HalfStats }) {
  const [x0, x1, y0, y1] = h.pitch_box_m
  const { x_edges_m: xe, y_edges_m: ye, counts } = h.heat_map
  const url = useMemo(() => {
    const ny = counts.length
    const nx = ny ? counts[0].length : 0
    if (!nx || !ny || typeof document === 'undefined') return null
    let mx = 0
    const flat: number[] = []
    for (const r of counts) for (const v of r) if (v > 0) flat.push(v)
    flat.sort((a, b) => a - b)
    mx = flat.length ? flat[Math.floor(flat.length * 0.985)] || flat[flat.length - 1] : 0
    if (!mx) return null
    const small = document.createElement('canvas')
    small.width = nx; small.height = ny
    const sctx = small.getContext('2d')
    if (!sctx) return null
    const img = sctx.createImageData(nx, ny)
    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const t = Math.min(1, Math.sqrt(counts[r][c] / mx))
        const o = ((ny - 1 - r) * nx + c) * 4 // far side of the pitch at the top
        // dark → yellow → orange → red, more opaque where there is more
        img.data[o] = 255
        img.data[o + 1] = Math.round(235 - 190 * Math.max(0, (t - 0.35) / 0.65))
        img.data[o + 2] = Math.round(60 * (1 - t))
        img.data[o + 3] = t < 0.04 ? 0 : Math.round(60 + 170 * t)
      }
    }
    sctx.putImageData(img, 0, 0)
    const big = document.createElement('canvas')
    const k = 5
    big.width = nx * k; big.height = ny * k
    const ctx = big.getContext('2d')
    if (!ctx) return null
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    try { ctx.filter = `blur(${k * 1.1}px)` } catch { /* older browsers: unblurred */ }
    ctx.drawImage(small, 0, 0, big.width, big.height)
    return big.toDataURL('image/png')
  }, [counts])

  const TOP = 7 // room above the pitch for the attack arrows
  const vx0 = x0 - PAD, vy1 = y1 + PAD + TOP
  const vbW = x1 - x0 + 2 * PAD
  const vbH = y1 - y0 + 2 * PAD + TOP
  const right = h.team_a_attacks_right
  const ay = -(y1 + PAD + 2.4)
  const arrow = (dir: 1 | -1, c: string, name: string, at: number) => (
    <g>
      <path d={`M${at - dir * 8},${ay} L${at + dir * 8},${ay} M${at + dir * 8 - dir * 2},${ay - 1.5} L${at + dir * 8},${ay} L${at + dir * 8 - dir * 2},${ay + 1.5}`} stroke={c} strokeWidth={0.9} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <text x={at} y={ay - 2.4} fontSize={3.2} fontWeight={700} fill="#fff" textAnchor="middle">{name}</text>
    </g>
  )
  return (
    <svg viewBox={`${vx0} ${-vy1} ${vbW} ${vbH}`} style={{ display: 'block', width: '100%', borderRadius: 8, background: '#2a7344' }} role="img" aria-label="Wärmekarte: wo das Spiel stattfand">
      <rect x={vx0} y={-vy1} width={vbW} height={vbH} fill="#1f5a35" />
      <g transform="scale(1,-1)"><Pitch box={h.pitch_box_m} grass={1} /></g>
      {url && xe.length > 1 && ye.length > 1 && (
        <image href={url} x={xe[0]} y={-ye[ye.length - 1]} width={xe[xe.length - 1] - xe[0]} height={ye[ye.length - 1] - ye[0]} preserveAspectRatio="none" />
      )}
      <g transform="scale(1,-1)"><Pitch box={h.pitch_box_m} grass={0} /></g>
      {arrow(right ? 1 : -1, h.teams.A.colour, h.teams.A.name, x0 + (x1 - x0) * (right ? 0.25 : 0.75))}
      {arrow(right ? -1 : 1, h.teams.B.colour, h.teams.B.name, x0 + (x1 - x0) * (right ? 0.75 : 0.25))}
      <text x={vx0 + vbW - 1} y={-vy1 + vbH - 1} fontSize={2.6} fill="#fff" fillOpacity={0.6} textAnchor="end">▼ Kamera</text>
    </svg>
  )
}

/** The statistics of a game: territory, momentum, team shape, heat map. `dark` = over the video (phone HUD). */
export default function StatsPanel({ stats, dark = false, halfStart, now, onSeek }: {
  stats: Stats
  dark?: boolean
  /** video time at which a half starts (from the board data), or null when unknown */
  halfStart: (half: string) => number | null
  now: number
  onSeek: (sec: number) => void
}) {
  const th = dark ? DARK : LIGHT
  const halves = Object.keys(stats.halves).sort()
  const [half, setHalf] = useState(halves[0])
  const [metric, setMetric] = useState<MetricKey>('spread_m')
  const raw = stats.halves[half] ?? stats.halves[halves[0]]
  const cur = useMemo<HalfStats>(() => ({
    ...raw,
    teams: { A: { ...raw.teams.A, colour: readable(raw.teams.A.colour, dark) }, B: { ...raw.teams.B, colour: readable(raw.teams.B.colour, dark) } },
  }), [raw, dark])
  const m = METRICS.find(x => x.key === metric) ?? METRICS[0]
  const A = cur.teams.A
  const B = cur.teams.B
  const wide = !dark

  // the tiles' numbers in their display unit
  const series = (key: MetricKey, t: TeamSeries): TeamValues => ({ ...t, values: t[key].map(v => (v == null ? null : v * (METRICS.find(x => x.key === key)?.scale ?? 1))) })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {halves.length > 1 && (
        <div role="tablist" style={{ display: 'flex', gap: 6 }}>
          {halves.map(h => (
            <button
              key={h} role="tab" aria-selected={h === half} onClick={() => setHalf(h)}
              style={{
                flex: dark ? 1 : undefined, padding: '7px 14px', borderRadius: 99, border: 'none', cursor: 'pointer',
                fontSize: 12, fontWeight: 600, fontFamily: 'DM Sans, sans-serif',
                background: h === half ? th.chipOn : th.chip, color: h === half ? th.chipOnText : th.muted,
              }}
            >
              {h}. Halbzeit
            </button>
          ))}
        </div>
      )}

      <Card th={th} title="Wer hatte das Spiel?" note="Anteil der Zeit, in der die Mannschaft das Spiel in der gegnerischen Hälfte hatte.">
        <TerritoryBar th={th} h={cur} />
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: wide ? 'repeat(4, 1fr)' : '1fr 1fr', gap: 8 }}>
        {METRICS.map(mt => (
          <Tile key={mt.key} th={th} label={`${mt.label} Ø`} hint={mt.note} unit={mt.unit}
            a={series(mt.key, A)} b={series(mt.key, B)} better={mt.key === 'spread_m' ? 'low' : mt.key === 'tempo_ms' ? 'high' : undefined} />
        ))}
      </div>
      <TeamLegend th={th} A={A} B={B} />

      <Card th={th} title="Spielverlauf" note={`Über der Linie hatte ${A.name} das Spiel in der gegnerischen Hälfte, darunter ${B.name}.`}>
        <Momentum th={th} h={cur} videoStart={halfStart(half)} now={now} onSeek={onSeek} />
      </Card>

      <div style={{ display: 'grid', gridTemplateColumns: wide ? '1.25fr 1fr' : '1fr', gap: 10, alignItems: 'start' }}>
        <Card th={th} title="Mannschaftsform" note={`${m.note} Geglättet über 3 Minuten.`}>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
            {METRICS.map(mt => (
              <button
                key={mt.key} onClick={() => setMetric(mt.key)}
                style={{
                  padding: '5px 11px', borderRadius: 99, border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 600,
                  fontFamily: 'DM Sans, sans-serif', background: mt.key === metric ? th.chipOn : th.chip, color: mt.key === metric ? th.chipOnText : th.muted,
                }}
              >
                {mt.label}
              </button>
            ))}
          </div>
          <ShapeChart th={th} h={cur} metric={m} />
        </Card>
        <Card th={th} title="Wärmekarte" note="Wo das Spiel stattfand. Pfeile zeigen die Angriffsrichtung.">
          <HeatMap h={cur} />
        </Card>
      </div>
    </div>
  )
}

function TeamLegend({ th, A, B }: { th: Theme; A: TeamSeries; B: TeamSeries }) {
  return (
    <div style={{ display: 'flex', gap: 14, fontSize: 11, color: th.muted, marginTop: -4 }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><Dot c={A.colour} />{A.name} links</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}><Dot c={B.colour} />{B.name} rechts</span>
    </div>
  )
}

/** One-line summary for the collapsed card header. */
export function statsHeadline(stats: Stats): { a: string; b: string; ca: string; cb: string; share: number } {
  const hs = Object.values(stats.halves)
  const first = hs[0]
  let n = 0, pos = 0
  for (const h of hs) for (const v of h.territory.value) if (v != null) { n++; if (v > 0) pos++ }
  // team A is the same team in every half
  return { a: first.teams.A.name, b: first.teams.B.name, ca: first.teams.A.colour, cb: first.teams.B.colour, share: n ? pos / n : 0.5 }
}

/** Desktop: a card that opens under the video. Collapsed it shows the territory split; open/closed is remembered on this device. */
export function StatsCard({ stats, halfStart, now, onSeek }: {
  stats: Stats
  halfStart: (half: string) => number | null
  now: number
  onSeek: (sec: number) => void
}) {
  const [open, setOpen] = useState(() => {
    if (typeof window === 'undefined') return false
    try { return localStorage.getItem('stats-open') === '1' } catch { return false }
  })
  const hl = statsHeadline(stats)
  hl.ca = readable(hl.ca, false); hl.cb = readable(hl.cb, false)
  return (
    <div style={{ background: '#fff', border: '1px solid #E4E6EE', borderRadius: 12, overflow: 'hidden' }}>
      <button
        onClick={() => {
          setOpen(v => {
            try { localStorage.setItem('stats-open', v ? '0' : '1') } catch { /* private window */ }
            return !v
          })
        }}
        aria-expanded={open}
        style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', background: 'transparent', border: 'none', cursor: 'pointer', fontFamily: 'DM Sans, sans-serif', textAlign: 'left' }}
      >
        <span style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 18, letterSpacing: '0.04em', color: NAVY, whiteSpace: 'nowrap' }}>📊 Statistik</span>
        {!open ? (
          <span style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: '#4A4F5C', whiteSpace: 'nowrap' }}>{Math.round(hl.share * 100)}%</span>
            <span style={{ flex: 1, display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', gap: 2, maxWidth: 260 }}>
              <span style={{ width: `${hl.share * 100}%`, background: hl.ca }} />
              <span style={{ flex: 1, background: hl.cb }} />
            </span>
            <span style={{ fontSize: 11, fontWeight: 600, color: '#4A4F5C', whiteSpace: 'nowrap' }}>{Math.round((1 - hl.share) * 100)}%</span>
            <span style={{ fontSize: 11, color: '#8A8F9E', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>Spielanteil {hl.a} · {hl.b}</span>
          </span>
        ) : <span style={{ flex: 1 }} />}
        <span style={{ fontSize: 12, color: '#8A8F9E', whiteSpace: 'nowrap' }}>{open ? 'Zuklappen ▴' : 'Anzeigen ▾'}</span>
      </button>
      {open && (
        <div style={{ padding: '0 14px 14px' }}>
          <StatsPanel stats={stats} halfStart={halfStart} now={now} onSeek={onSeek} />
        </div>
      )}
    </div>
  )
}
