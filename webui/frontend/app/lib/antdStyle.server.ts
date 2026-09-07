/**
 * Locates the antd stylesheet baked by `scripts/genAntdCss.tsx`.
 *
 * With `theme.zeroRuntime` on (see `msaTheme.ts`) antd/x never emit component
 * CSS at runtime, so this file IS the component styling — the root loader
 * hands its href to the document `<head>`. Server-only (`.server` suffix keeps
 * `node:fs` out of the client bundle).
 *
 * Only the manifest lives here; the stylesheet it points at sits in
 * `public/assets/`, the one path served as `immutable` (`frontend/server.js`).
 * The manifest is deliberately kept out of there — see the script.
 */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

interface AntdCssManifest {
  href: string
  hash: string
  /** `path relative to the frontend root` → hash of its content at bake time,
   * written by the script from its own `BAKE_INPUTS` list. */
  sources?: Record<string, string>
}

/** `public/` is the source of truth while developing (a stale `build/` from an
 * earlier production build would otherwise win); the built client directory is
 * checked first in production, where `public/` may not be deployed at all. */
const CANDIDATES = import.meta.env.DEV
  ? ['public/antd/manifest.json', 'build/client/antd/manifest.json']
  : ['build/client/antd/manifest.json', 'public/antd/manifest.json']

let cached: string | null | undefined
let lastStaleWarning: string | null = null

/** `pnpm dev` bakes once before starting Vite, so editing the theme afterwards
 * leaves the served stylesheet a step behind — with zeroRuntime on that shows
 * up as styles that quietly do not match the tokens, which is hard to place.
 * Comparing the recorded input hashes turns it into a line in the log. */
function warnIfStale(manifest: AntdCssManifest) {
  const stale = Object.entries(manifest.sources ?? {}).filter(([rel, hash]) => {
    try {
      const current = fs.readFileSync(path.resolve(process.cwd(), rel), 'utf8')
      return createHash('sha256').update(current).digest('hex').slice(0, 8) !== hash
    } catch {
      // The file moved or was renamed — the manifest is out of date either way.
      return true
    }
  })
  if (!stale.length) {
    lastStaleWarning = null
    return
  }
  // read() runs per request in dev; only speak up when the set changes.
  const signature = stale.map(([rel]) => rel).join()
  if (signature === lastStaleWarning) return
  lastStaleWarning = signature
  console.warn(
    `[antd] the baked stylesheet predates ${signature} — run \`pnpm gen:antd-css\` (no restart needed)`
  )
}

function read(): string | null {
  for (const candidate of CANDIDATES) {
    try {
      const raw = fs.readFileSync(path.resolve(process.cwd(), candidate), 'utf8')
      const manifest = JSON.parse(raw) as AntdCssManifest
      if (manifest.href) {
        if (import.meta.env.DEV) warnIfStale(manifest)
        return manifest.href
      }
    } catch {
      // Try the next location.
    }
  }
  // Loud but non-fatal: the app still renders, only unstyled — which is a much
  // easier symptom to diagnose with this line in the log.
  console.error(
    '[antd] no baked stylesheet found — run `pnpm gen:antd-css` (zeroRuntime is on, so antd emits no CSS at runtime)'
  )
  return null
}

/** Href of the generated antd stylesheet, or `null` when it has not been
 * generated. Re-read on every call in dev so a re-bake needs no restart. */
export function getAntdCssHref(): string | null {
  if (import.meta.env.DEV) return read()
  if (cached === undefined) cached = read()
  return cached
}
