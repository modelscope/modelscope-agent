/**
 * Shared cache for the MCP reachability sweep (`GET /api/mcps/health`).
 *
 * Three surfaces show the same verdicts — the global MCP page, a project's MCP
 * tab, and the composer's MCP pill — and each used to sweep from its own mount.
 * So every switch between them re-asked, and the pill plus the tab mounted
 * together on a project page asked twice at once.
 *
 * The backend does cache probes, but not in a way that makes those repeats
 * free: a reachable server is remembered for 60 s, an unreachable one for only
 * 10 s, deliberately (a pinned failure withholds that server's tools from a
 * whole conversation). So as soon as anything is down, a switch lands on a
 * cache miss — measured at 8.7 s for five dead servers against 5 ms warm, with
 * every card spinning for the duration.
 *
 * This module holds the last sweep in one client-side place: a mount reads it
 * instead of asking, and a request only goes out when that copy is older than
 * FRESH_MS or when the server set actually changed (a mutation here, or
 * `msa:mcp-skill-changed` from elsewhere). Concurrent mounts share the one
 * in-flight request.
 *
 * State is module-level, so it lives as long as the tab and a reload starts
 * cold on purpose. That also means it must never be written during render:
 * under SSR this module is shared by every request, so the sweep is only ever
 * kicked off from an effect, leaving the server-side copy permanently empty.
 */
import { useCallback, useEffect, useSyncExternalStore } from 'react'
import { api } from './api'
import { useOnMcpSkillChanged } from './events'
import type { McpHealth } from './types'

/** How long a landed sweep is served without asking again. Matched to the
 * backend's success TTL: past it the backend re-probes anyway, so caching for
 * longer would only buy staleness. */
const FRESH_MS = 60_000

export interface McpHealthState {
  /** Verdicts by MCP id, from the last sweep that landed. */
  rows: Record<string, McpHealth>
  /** A sweep is out, or none has landed yet. Consumers spin only for ids with
   * no verdict yet, so a revalidation never blanks what is already on screen. */
  sweeping: boolean
}

// The server render, and the client's hydration render with it: no verdicts,
// sweep understood to be pending — which is what the HTML says. Cached verdicts
// then arrive in the re-render right after mount, so hydration cannot mismatch.
const PENDING: McpHealthState = { rows: {}, sweeping: true }

let state: McpHealthState = PENDING
let fetchedAt = 0
let inFlight: Promise<void> | null = null
const listeners = new Set<() => void>()

function set(next: McpHealthState) {
  state = next
  listeners.forEach((notify) => notify())
}

function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

const getSnapshot = () => state
const getServerSnapshot = () => PENDING

/** Sweep unless a fresh copy is already cached. `force` is for the case where
 * the cached answer is known to describe the wrong set of servers. */
export function sweepMcpHealth(force = false): Promise<void> {
  // Whoever asked first is already fetching; joining it is the point.
  if (inFlight) return inFlight
  if (!force && fetchedAt && Date.now() - fetchedAt < FRESH_MS) {
    if (state.sweeping) set({ ...state, sweeping: false })
    return Promise.resolve()
  }
  set({ ...state, sweeping: true })
  inFlight = api
    .listMcpHealth()
    .then((rows) => {
      fetchedAt = Date.now()
      set({
        rows: Object.fromEntries(rows.map((h) => [h.id, h])),
        sweeping: false
      })
    })
    // Keep the previous verdicts when a sweep fails: a blip must not turn every
    // card's status into "unknown".
    .catch(() => set({ ...state, sweeping: false }))
    .finally(() => {
      inFlight = null
    })
  return inFlight
}

/** Record one server's verdict — a deep check or a manual reconnect, both more
 * authoritative than the sweep's shallow probe. It deliberately does NOT
 * refresh the sweep's timestamp: one row says nothing about the others. */
export function putMcpHealth(row: McpHealth) {
  set({ ...state, rows: { ...state.rows, [row.id]: row } })
}

/** Subscribe to the shared verdicts. Mounting costs a request only when the
 * cached sweep is stale; `refresh(true)` forces one after a mutation. */
export function useMcpHealth() {
  const shared = useSyncExternalStore(
    subscribe,
    getSnapshot,
    getServerSnapshot
  )
  useEffect(() => {
    void sweepMcpHealth()
  }, [])
  // Config changed somewhere else, so the cached verdicts describe the old set.
  useOnMcpSkillChanged(
    useCallback(() => {
      void sweepMcpHealth(true)
    }, [])
  )
  return { ...shared, refresh: sweepMcpHealth, put: putMcpHealth }
}
