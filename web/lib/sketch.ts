// Hand-drawn look for the scene editor: pure functions that turn geometry into SVG path strings.
// Everything is deterministic (seeded), so a drawing never shimmers between renders and the same scene always looks the same.
// Units are whatever the points are in (the editor uses metres); `amp` etc. must be given in the same unit.

export type Pt = [number, number]

export function hashStr(s: string): number {
  let h = 7
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h) + 1
}

export function rng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const r2 = (n: number) => Math.round(n * 100) / 100
const P = (p: Pt) => `${r2(p[0])} ${r2(p[1])}`

export function polyPath(pts: Pt[]): string {
  if (!pts.length) return ''
  return 'M' + pts.map(P).join('L')
}

export function dist(a: Pt, b: Pt): number { return Math.hypot(b[0] - a[0], b[1] - a[1]) }

/** Points on the quadratic curve from a to b whose middle is pushed `bend` away from the straight line (bend 0 = straight). */
export function curvePoints(a: Pt, b: Pt, bend: number, n = 24): Pt[] {
  const L = dist(a, b)
  if (L < 1e-9) return [a, b]
  const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L
  const c: Pt = [(a[0] + b[0]) / 2 + nx * bend * 2, (a[1] + b[1]) / 2 + ny * bend * 2]
  const out: Pt[] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n, u = 1 - t
    out.push([u * u * a[0] + 2 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2 * u * t * c[1] + t * t * b[1]])
  }
  return out
}

export function pathLength(pts: Pt[]): number {
  let s = 0
  for (let i = 1; i < pts.length; i++) s += dist(pts[i - 1], pts[i])
  return s
}

/** Cut `startTrim` off the start and `endTrim` off the end of a polyline. Returns the middle part (at least two points). */
export function trimPolyline(pts: Pt[], startTrim: number, endTrim: number): Pt[] {
  const total = pathLength(pts)
  if (total <= startTrim + endTrim + 1e-6) return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : pts
  const from = startTrim, to = total - endTrim
  const out: Pt[] = []
  let acc = 0
  const at = (p: Pt, q: Pt, d: number): Pt => { const L = dist(p, q) || 1; return [p[0] + ((q[0] - p[0]) * d) / L, p[1] + ((q[1] - p[1]) * d) / L] }
  for (let i = 1; i < pts.length; i++) {
    const seg = dist(pts[i - 1], pts[i])
    const s0 = acc, s1 = acc + seg
    if (s1 >= from && s0 <= to) {
      if (!out.length) out.push(s0 >= from ? pts[i - 1] : at(pts[i - 1], pts[i], from - s0))
      out.push(s1 <= to ? pts[i] : at(pts[i - 1], pts[i], to - s0))
    }
    acc = s1
  }
  return out.length >= 2 ? out : [pts[0], pts[pts.length - 1]]
}

/** Resample a polyline at equal steps (the last point is always kept). */
export function resample(pts: Pt[], step: number): Pt[] {
  if (pts.length < 2 || step <= 0) return pts
  const out: Pt[] = [pts[0]]
  let carry = 0
  let prev = pts[0]
  for (let i = 1; i < pts.length; i++) {
    let cur = pts[i]
    let d = dist(prev, cur)
    while (d > 0 && carry + d >= step) {
      const u = (step - carry) / d
      const q: Pt = [prev[0] + (cur[0] - prev[0]) * u, prev[1] + (cur[1] - prev[1]) * u]
      out.push(q)
      prev = q
      d = dist(prev, cur)
      carry = 0
    }
    carry += d
    prev = cur
    cur = pts[i]
  }
  const last = pts[pts.length - 1]
  if (dist(out[out.length - 1], last) > step * 0.25) out.push(last)
  else out[out.length - 1] = last
  return out
}

/** Smooth 1-D noise: random values on a lattice every `wl` units of arc length, blended with a smoothstep. Range about -1..1. */
function noise(seed: number, wl: number): (s: number) => number {
  const cache = new Map<number, number>()
  const val = (i: number) => {
    let v = cache.get(i)
    if (v === undefined) { v = rng(seed * 7919 + i * 104729)() * 2 - 1; cache.set(i, v) }
    return v
  }
  return (s: number) => {
    const x = s / wl, i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f)
    return val(i) * (1 - u) + val(i + 1) * u
  }
}

/** Offset a polyline sideways by smooth noise, so a straight line looks drawn by hand. amp 0 returns the line unchanged. */
export function wobble(pts: Pt[], seed: number, amp: number): Pt[] {
  if (amp <= 0 || pts.length < 2) return pts
  const total = pathLength(pts)
  const dense = resample(pts, Math.max(amp * 3, total / 40))
  const nz = noise(seed, Math.max(amp * 9, total / 3.2))
  const out: Pt[] = []
  let s = 0
  for (let i = 0; i < dense.length; i++) {
    if (i > 0) s += dist(dense[i - 1], dense[i])
    const a = dense[Math.max(0, i - 1)], b = dense[Math.min(dense.length - 1, i + 1)]
    const L = dist(a, b) || 1
    const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L
    const taper = Math.min(1, s / (amp * 3), (total - s) / (amp * 3)) // keep the ends where the coach put them
    const o = nz(s) * amp * Math.max(0, taper)
    out.push([dense[i][0] + nx * o, dense[i][1] + ny * o])
  }
  return out
}

/** A wavy line (the notation for player movement). */
export function wavy(pts: Pt[], amp: number, wavelength: number): Pt[] {
  if (pts.length < 2) return pts
  const R = resample(pts, wavelength / 10)
  const out: Pt[] = []
  let s = 0
  const total = pathLength(R)
  for (let i = 0; i < R.length; i++) {
    if (i > 0) s += dist(R[i - 1], R[i])
    const a = R[Math.max(0, i - 1)], b = R[Math.min(R.length - 1, i + 1)]
    const L = dist(a, b) || 1
    const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L
    const taper = Math.min(1, s / (wavelength * 0.5), (total - s) / (wavelength * 0.5))
    const o = amp * Math.sin((s / wavelength) * Math.PI * 2) * Math.max(0, taper)
    out.push([R[i][0] + nx * o, R[i][1] + ny * o])
  }
  return out
}

/** Open arrowhead at `tip`, pointing away from `from`. */
export function arrowHead(from: Pt, tip: Pt, len: number, seed = 1, jitter = 0): string {
  const a = Math.atan2(tip[1] - from[1], tip[0] - from[0]), w = 0.46
  const g = rng(seed), j = () => (jitter ? (g() - 0.5) * jitter : 0)
  const p1: Pt = [tip[0] - Math.cos(a - w) * len + j(), tip[1] - Math.sin(a - w) * len + j()]
  const p2: Pt = [tip[0] - Math.cos(a + w) * len + j(), tip[1] - Math.sin(a + w) * len + j()]
  return `M${P(p1)}L${P(tip)}L${P(p2)}`
}

/** A closed, slightly irregular disc. jit 0 gives a perfect circle. */
export function blobPath(c: Pt, r: number, seed: number, jit: number): string {
  if (jit <= 0) return `M${r2(c[0] - r)} ${r2(c[1])}a${r2(r)} ${r2(r)} 0 1 0 ${r2(2 * r)} 0a${r2(r)} ${r2(r)} 0 1 0 ${r2(-2 * r)} 0Z`
  const g = rng(seed), n = 20, pts: Pt[] = []
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2, rr = r * (1 + (g() - 0.5) * jit)
    pts.push([c[0] + Math.cos(a) * rr, c[1] + Math.sin(a) * rr * 0.97])
  }
  return polyPath(pts) + 'Z'
}

/** Rectangle outline as four separately wobbled edges (a hand-drawn box). */
export function boxPath(a: Pt, b: Pt, seed: number, amp: number): string {
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1])
  const corners: Pt[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
  let d = ''
  for (let i = 0; i < 4; i++) d += polyPath(wobble([corners[i], corners[(i + 1) % 4]], seed + i, amp))
  return d
}

/** Decimate a freehand stroke: keep a point only when it moved at least `minStep` from the last kept one. */
export function decimate(pts: Pt[], minStep: number): Pt[] {
  if (pts.length < 3) return pts
  const out: Pt[] = [pts[0]]
  for (let i = 1; i < pts.length - 1; i++) if (dist(out[out.length - 1], pts[i]) >= minStep) out.push(pts[i])
  out.push(pts[pts.length - 1])
  return out
}

/** Smooth a freehand stroke into a path with quadratic curves through the midpoints. */
export function smoothPath(pts: Pt[]): string {
  if (pts.length < 3) return polyPath(pts)
  let d = `M${P(pts[0])}`
  for (let i = 1; i < pts.length - 1; i++) {
    const m: Pt = [(pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2]
    d += `Q${P(pts[i])} ${P(m)}`
  }
  return d + `L${P(pts[pts.length - 1])}`
}
