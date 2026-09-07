import { useEffect, useState } from 'react'

/**
 * Live "did this picture reach the model" state, keyed by workspace path.
 *
 * History carries the answer on each attached file, but a turn in progress does
 * not: the user bubble is rendered from what the composer sent, and the answer
 * is only known once the request has been built. Without this the badge would
 * appear on reload and not before — for the turn the user is actually looking
 * at, which is the one that matters.
 *
 * Same shape as the workspace-file broadcast next door: a window event and a
 * hook, so any bubble on screen picks it up without threading state through the
 * chat provider.
 */
const EVENT = 'msa:image-delivery'

export type DeliveryState = 'delivered' | 'degraded' | 'unreadable'

const live = new Map<string, DeliveryState>()

export function dispatchImageDelivery(path: string, state: string): void {
  if (!path) return
  // Once the model has received a picture, nothing un-receives it: a later turn
  // on a text-only model does not retroactively make it unseen.
  if (live.get(path) === 'delivered') return
  live.set(path, state as DeliveryState)
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(EVENT))
  }
}

/** Current live map. Re-renders on every delivery report. */
export function useImageDelivery(): Map<string, DeliveryState> {
  const [, bump] = useState(0)
  useEffect(() => {
    const onChange = () => bump((n) => n + 1)
    window.addEventListener(EVENT, onChange)
    return () => window.removeEventListener(EVENT, onChange)
  }, [])
  return live
}
