import { useCallback, useEffect, useRef } from 'react'
import { useRevalidator } from 'react-router'
import { api } from '~/lib/api'
import { useOnSessionDone } from '~/lib/events'

/**
 * Retire the unread dot of the session being viewed.
 *
 * Opening a conversation IS reading it, so the flag a background turn left
 * behind (backend `chat._drain_abandoned_turn`) is cleared on mount. Gated on
 * the loader having actually seen it set: clearing has to be followed by a
 * revalidate to repaint the sidebar, and that re-runs the whole app-layout
 * loader set — sessions with no dot are the overwhelming majority and must not
 * pay for it on every open.
 *
 * The second call covers a turn that finishes while the view is open. The
 * attached viewer and the server's drain task both wake on the same turn-end
 * event, so the drain can win the race and flag a session the user is watching
 * right now. No revalidate for that one: the presence heartbeat is already about
 * to run one for the session leaving the running set, and this write lands well
 * before it — so the dot never gets painted in the first place.
 */
export function useMarkSessionRead(sessionId: string, unread: boolean): void {
  const revalidator = useRevalidator()
  // The revalidator object is a fresh identity on every render; keeping the
  // function in a ref keeps it out of the effect's deps, which must be only
  // "which session, and was it unread".
  const revalidateRef = useRef(revalidator.revalidate)
  revalidateRef.current = revalidator.revalidate

  useEffect(() => {
    if (!sessionId || !unread) return
    let cancelled = false
    api
      .markSessionRead(sessionId)
      .then(() => {
        if (!cancelled) revalidateRef.current()
      })
      .catch(() => {
        // Best effort: a stale dot is a far smaller problem than an error page
        // over a conversation that opened fine.
      })
    return () => {
      cancelled = true
    }
  }, [sessionId, unread])

  useOnSessionDone(
    useCallback(
      (sid: string) => {
        if (sid !== sessionId) return
        api.markSessionRead(sessionId).catch(() => {})
      },
      [sessionId]
    )
  )
}
