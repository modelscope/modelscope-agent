/**
 * Production entry: React Router SSR and the API on ONE port.
 *
 * This is `react-router-serve` plus the one thing it structurally cannot do,
 * built on the same pieces (express, compression, morgan, express.static) so
 * the parts that were already right keep behaving exactly as they did.
 *
 * Why it has to exist at all: `react-router-serve`'s whole routing table ends at
 * `app.all("/{*splat}", createRequestHandler(...))`, a catch-all with no exit for
 * `/api/*`. Those requests never leave the SSR process — React Router answers
 * them with its own 404. Nothing noticed, because something else always covered
 * it: vite's `server.proxy` in dev (vite.config.ts), nginx in the container. So
 * `pnpm start` was quietly broken on its own, and any launcher spawning
 * `react-router-serve` walks into it: both processes come up healthy, every
 * browser fetch of a relative `/api` path 404s.
 *
 * One public port is a requirement, not a convenience: the browser half of
 * `app/lib/api.ts` uses relative `/api` URLs on purpose, so the API must be
 * same-origin. Handing the browser an absolute backend URL instead would mean
 * CORS plus a second port the user has to know about — and with port probing
 * (a busy 8000 becoming 8001) it would have to be discovered at runtime too.
 *
 * Layout, in the order a request is tried:
 *   /api/*      -> reverse-proxied to MS_AGENT_API_BASE_URL, ahead of compression
 *   /assets/*   -> build/client/assets, content-hashed, immutable
 *   <any file>  -> build/client, then public/ (favicon, baked antd css, ...)
 *   everything  -> React Router SSR
 *
 * Inherited from nginx, which this file also replaced (see Dockerfile):
 * `client_max_body_size 50m`, `X-Real-IP` / `X-Forwarded-*`, unbuffered
 * proxying, and no read timeout on a streaming turn.
 *
 * Note there are two same-named `createRequestListener`s in this dependency
 * tree — the one in `@remix-run/node-fetch-server` (which react-router-serve
 * uses) takes a fetch function, this one takes `{ build, mode }`.
 */
import { createRequestListener } from '@react-router/node'
import compression from 'compression'
import express from 'express'
import morgan from 'morgan'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const CLIENT_DIR = fileURLToPath(new URL('./build/client/', import.meta.url))
const PUBLIC_DIR = fileURLToPath(new URL('./public/', import.meta.url))
const SERVER_BUILD = new URL('./build/server/index.js', import.meta.url)

// `MS_AGENT_API_BASE_URL` on purpose: the launcher already exports it for the
// dev server's proxy, so the production entry reading the same variable means
// the backend's address is declared in exactly one place regardless of how the
// stack was started. (SSR loaders read MS_AGENT_FRONTEND_API_BASE_URL instead —
// they bypass this proxy entirely, see app/lib/env.ts.) Both carry the project
// prefix so a shared shell or a PaaS injecting a generic API_BASE_URL cannot
// silently re-aim this proxy.
const API_TARGET = new URL(
  process.env.MS_AGENT_API_BASE_URL || 'http://127.0.0.1:8000'
)
const API_PORT = Number(API_TARGET.port || 80)
// Derived from the API's port rather than fixed, so the pair reads as a pair:
// an API on 8000 puts the app on 8001, and re-aiming the proxy moves the app
// with it. A hardcoded 3000 had no relation to the port it proxies to, which
// made it one more number to be told about. Only a bare `pnpm start` reaches
// this — the launcher probes for free ports and the container pins them, and
// both pass PORT explicitly. Note those two invert the pairing on purpose: the
// port they publish is the app's, so the API sits at app + 1 there.
const PORT = Number(process.env.PORT || API_PORT + 1)
const HOST = process.env.HOST || '127.0.0.1'

// nginx's `client_max_body_size 50m`, which vanished with it. Fixed, as it was
// there: workspace uploads are the only thing that gets near it, and a limit
// that moves per-deployment is one more thing an upload bug can hide behind.
const MAX_BODY_MB = 50
const MAX_BODY_BYTES = MAX_BODY_MB * 1024 * 1024

/** Per-connection headers, which a proxy must consume rather than forward
 * (RFC 9110 §7.6.1). Dropping `transfer-encoding` is safe and required: node
 * re-derives the framing for the upstream request from what we hand it. */
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade'
])

// Keep-alive so a chat turn's many small calls reuse one upstream socket. This
// is what nginx's `proxy_set_header Connection ""` was for.
const agent = new http.Agent({ keepAlive: true, maxSockets: 128 })

/** Refuse an oversized body the way nginx did: 413, then close. */
function reject413(req, res) {
  if (!res.headersSent) {
    res.writeHead(413, {
      'content-type': 'application/json; charset=utf-8',
      connection: 'close'
    })
  }
  res.end(
    JSON.stringify({
      code: 413,
      message: `Request body exceeds the ${MAX_BODY_MB} MB limit.`
    }),
    // Only tear the socket down once the response has actually flushed —
    // destroying it earlier is how a 413 turns into a bare connection reset
    // that the client reports as a network error instead of a limit.
    () => req.socket?.destroy()
  )
}

/** Declared-size check, global like nginx's directive. Anything with a
 * `content-length` (every browser upload: fetch with a File, FormData) is
 * rejected before a single byte of body is read. Chunked bodies have no size to
 * check here and are counted as they stream, in `proxyApi`. */
function enforceBodyLimit(req, res, next) {
  const declared = Number(req.headers['content-length'])
  if (Number.isFinite(declared) && declared > MAX_BODY_BYTES) {
    return reject413(req, res)
  }
  next()
}

/** Reverse-proxy one request to FastAPI, streaming both directions.
 *
 * Streaming is the whole point: `/api/chat` is SSE and the UI renders tokens as
 * they arrive, so nothing here may buffer — this is nginx's `proxy_buffering
 * off`. `pipe` in both directions covers it (and keeps large uploads off the
 * heap); `setNoDelay` stops Nagle from holding a lone token back waiting for a
 * fuller packet. Mounted ahead of `compression()` so an event stream is never
 * handed to a gzip transform, whatever the filter says. */
function proxyApi(req, res) {
  const headers = {}
  for (const [key, value] of Object.entries(req.headers)) {
    if (!HOP_BY_HOP.has(key)) headers[key] = value
  }

  const remote = req.socket.remoteAddress ?? ''
  const proto = req.socket.encrypted ? 'https' : 'http'
  // The client's Host is preserved rather than rewritten to the upstream's,
  // matching the `proxy_set_header Host $host` this replaced: it is what lets
  // the backend build an absolute URL that points at the port the browser can
  // actually reach. X-Forwarded-For appends, like $proxy_add_x_forwarded_for.
  headers['x-real-ip'] = remote
  headers['x-forwarded-for'] = req.headers['x-forwarded-for']
    ? `${req.headers['x-forwarded-for']}, ${remote}`
    : remote
  headers['x-forwarded-proto'] = proto
  headers['x-forwarded-host'] = req.headers.host ?? ''

  const upstream = http.request(
    {
      agent,
      host: API_TARGET.hostname,
      port: API_PORT,
      method: req.method,
      // originalUrl: `app.use('/api', ...)` strips the mount path from req.url,
      // and the backend needs the whole thing including the query string.
      path: req.originalUrl,
      headers
    },
    (up) => {
      res.writeHead(up.statusCode ?? 502, up.headers)
      // An SSE response's headers must reach the browser before the first
      // event, which can be seconds away while the model warms up.
      res.flushHeaders()
      up.pipe(res)
    }
  )

  // nginx allowed 3600s; node's default socket timeout would sever a long
  // agent turn mid-stream, so drop it entirely and let the client's disconnect
  // (handled below) be what ends the request.
  upstream.setTimeout(0)
  upstream.on('error', (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'application/json; charset=utf-8' })
      res.end(
        JSON.stringify({ code: 502, message: `Backend unreachable: ${err.message}` })
      )
      return
    }
    // Mid-stream: the status line is long gone, so the only honest signal left
    // is ending the body. Writing a JSON error here would corrupt it.
    res.end()
  })

  // A browser that navigates away mid-turn must not leave the upstream request
  // (and the agent turn behind it) running with nowhere to write.
  res.on('close', () => {
    if (!res.writableEnded) upstream.destroy()
  })

  // Streaming counterpart to enforceBodyLimit, for bodies that arrived chunked
  // and therefore had no length to check. Registered before `pipe` in the same
  // tick, so both consumers see every chunk.
  if (!req.headers['content-length']) {
    let seen = 0
    req.on('data', (chunk) => {
      seen += chunk.length
      if (seen > MAX_BODY_BYTES) {
        upstream.destroy()
        reject413(req, res)
      }
    })
  }
  req.pipe(upstream)
}

const app = express()
app.disable('x-powered-by')

// nginx applied its body limit to every location, not just /api.
app.use(enforceBodyLimit)

// Before compression, and the only branch that gets to skip it.
app.use('/api', proxyApi)

app.use(
  compression({
    // Belt and braces: /api never reaches this middleware, but an event stream
    // that somehow did would be buffered into silence by gzip.
    filter: (req, res) =>
      !String(res.getHeader('content-type') ?? '').includes('text/event-stream') &&
      compression.filter(req, res)
  })
)

// /assets/* filenames carry a content hash, so they can never go stale.
app.use(
  '/assets',
  express.static(path.join(CLIENT_DIR, 'assets'), { immutable: true, maxAge: '1y' })
)
// Everything else in the build: favicon, and `antd/manifest.json`, which is
// deliberately outside /assets because its name is fixed while its contents
// change (see app/lib/antdStyle.server.ts). One hour, matching what
// react-router-serve gave these same files when it served them from public/.
app.use(express.static(CLIENT_DIR, { maxAge: '1h' }))
// public/ is a superset-free fallback: `react-router build` copies it into
// build/client, so this only matters for a tree where the build ran elsewhere.
// Absolute path, not react-router-serve's cwd-relative one, so the entry works
// from any working directory (the launcher spawns it with cwd=frontend, the
// container does not).
app.use(express.static(PUBLIC_DIR, { maxAge: '1h', fallthrough: true }))

// One line per request, unconditional — both things this replaced logged every
// request too (react-router-serve's own morgan, nginx's `access_log`). Placed
// after the static handlers, as in react-router-serve: a file that gets served
// never reaches it, so the log stays about navigations and API calls.
app.use(morgan('tiny'))

app.all(
  '/{*splat}',
  createRequestListener({
    build: await import(SERVER_BUILD.href),
    mode: 'production'
  })
)

const server = http.createServer(app)

// Node caps a single request at 300 s by default, which would guillotine both a
// long agent turn's SSE stream and a slow upload. Header timeouts stay on —
// they only bound the request line, never the body.
server.requestTimeout = 0
server.on('connection', (socket) => socket.setNoDelay(true))

// Port selection belongs to whoever spawns this (the launcher probes for a free
// one), but probe-then-spawn always has a race window, so fail readably instead
// of dumping a node stack trace on the user.
server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`[webui] port ${PORT} is already in use on ${HOST}.`)
    process.exit(1)
  }
  throw err
})

server.listen(PORT, HOST, () => {
  // The launcher sets MS_AGENT_WEBUI_BANNER=0 and prints one URL itself, after
  // the API answers. Two URLs in one terminal is what makes people open the
  // wrong one, and this one is reachable a moment before the API behind it is.
  if (process.env.MS_AGENT_WEBUI_BANNER !== '0') {
    console.log(`[webui] http://${HOST}:${PORT}  (api -> ${API_TARGET.origin})`)
  }
})

for (const signal of ['SIGTERM', 'SIGINT']) {
  process.once(signal, () => {
    // Idle keep-alive sockets (browser tabs, and our own upstream pool) would
    // otherwise hold `close` open until they time out on their own.
    server.closeIdleConnections?.()
    agent.destroy()
    server.close(() => process.exit(0))
  })
}
