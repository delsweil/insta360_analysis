// Renders the animation to a video (MP4) or GIF in the browser, so it can be sent by WhatsApp or shown on a screen.
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { Muxer, ArrayBufferTarget } from 'mp4-muxer'
import { GIFEncoder, quantize, applyPalette } from 'gifenc'
import SceneCanvas from '@/components/SceneCanvas'
import { frameAt, plan, type Step } from '@/lib/anim'
import { viewBoxOf } from '@/lib/scene'

export type ExportFormat = 'mp4' | 'gif'
export const FPS = { mp4: 30, gif: 12 }

async function dataUrl(path: string): Promise<string> {
  const buf = new Uint8Array(await (await fetch(path)).arrayBuffer())
  let bin = ''
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000))
  return `data:font/woff2;base64,${btoa(bin)}`
}

async function fontCss(): Promise<string> {
  try {
    const [sans, hand] = await Promise.all([dataUrl('/fonts/dm-sans-latin-600-normal.woff2'), dataUrl('/fonts/caveat-latin-700-normal.woff2')])
    return `@font-face{font-family:'DM Sans';font-weight:100 900;src:url(${sans}) format('woff2')}@font-face{font-family:'Caveat';font-weight:100 900;src:url(${hand}) format('woff2')}`
  } catch { return '' }
}

/** The pixel size of the video: the longer side is about 1280, both even (required by H.264). */
export function exportSize(steps: Step[], longest = 1280): { w: number; h: number } {
  const vb = viewBoxOf(steps[0].scene.background)
  const r = vb.w / vb.h
  let w = r >= 1 ? longest : longest * r, h = r >= 1 ? longest / r : longest
  w = Math.max(2, Math.round(w / 2) * 2); h = Math.max(2, Math.round(h / 2) * 2)
  return { w, h }
}

function svgOf(scene: Step['scene'], w: number, h: number, css: string): string {
  let s = renderToStaticMarkup(createElement(SceneCanvas, { scene, plain: true, style: { width: w, height: h } }))
  s = s.replace(/var\(--font-hand\)/g, "'Caveat'").replace(/var\(--font-dm-sans\)/g, "'DM Sans'")
  s = s.replace('<svg ', `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" `)
  const i = s.indexOf('>')
  return `${s.slice(0, i + 1)}<style>${css}</style><rect x="-10000" y="-10000" width="20000" height="20000" fill="#fbfbf8"/>${s.slice(i + 1)}`
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error('Bild konnte nicht erzeugt werden')); im.src = src })
}

export function canExportMp4(): boolean {
  return typeof window !== 'undefined' && 'VideoEncoder' in window
}

export async function exportAnimation(steps: Step[], format: ExportFormat, onProgress: (p: number) => void, signal?: { cancelled: boolean }): Promise<Blob> {
  const pl = plan(steps)
  const { w, h } = exportSize(steps, format === 'gif' ? 720 : 1280)
  const fps = FPS[format]
  const frames = Math.max(1, Math.ceil(pl.total * fps) + 1)
  const css = await fontCss()
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!

  let muxer: Muxer<ArrayBufferTarget> | null = null
  let encoder: VideoEncoder | null = null
  const gif = format === 'gif' ? GIFEncoder() : null
  if (format === 'mp4') {
    muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: 'avc', width: w, height: h }, fastStart: 'in-memory' })
    encoder = new VideoEncoder({ output: (c, m) => muxer!.addVideoChunk(c, m), error: e => { throw e } })
    encoder.configure({ codec: 'avc1.640028', width: w, height: h, bitrate: 3_000_000, framerate: fps })
  }

  for (let i = 0; i < frames; i++) {
    if (signal?.cancelled) throw new Error('abgebrochen')
    const t = Math.min(pl.total, i / fps)
    const url = URL.createObjectURL(new Blob([svgOf(frameAt(steps, t).scene, w, h, css)], { type: 'image/svg+xml' }))
    try {
      const im = await loadImage(url)
      ctx.fillStyle = '#fbfbf8'; ctx.fillRect(0, 0, w, h)
      ctx.drawImage(im, 0, 0, w, h)
    } finally { URL.revokeObjectURL(url) }
    if (encoder) {
      const vf = new VideoFrame(canvas, { timestamp: Math.round((i * 1e6) / fps), duration: Math.round(1e6 / fps) })
      encoder.encode(vf, { keyFrame: i % fps === 0 })
      vf.close()
      if (encoder.encodeQueueSize > 8) await new Promise(r => setTimeout(r, 0))
    } else if (gif) {
      const rgba = ctx.getImageData(0, 0, w, h).data
      const palette = quantize(rgba, 128)
      gif.writeFrame(applyPalette(rgba, palette), w, h, { palette, delay: Math.round(1000 / fps), repeat: 0 })
    }
    onProgress((i + 1) / frames)
  }
  if (encoder && muxer) {
    await encoder.flush(); muxer.finalize()
    return new Blob([muxer.target.buffer], { type: 'video/mp4' })
  }
  gif!.finish()
  return new Blob([gif!.bytes() as BlobPart], { type: 'image/gif' })
}
