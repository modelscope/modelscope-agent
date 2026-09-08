import { useEffect } from 'react'

/** Broadcast that the active model changed outside the model picker, so any
 * composer on screen refreshes its selection instead of showing a stale one. */
const EVENT = 'msa:model-changed'

export function dispatchModelChanged(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(EVENT))
  }
}

export function useModelChanged(onChange: () => void): void {
  useEffect(() => {
    window.addEventListener(EVENT, onChange)
    return () => window.removeEventListener(EVENT, onChange)
  }, [onChange])
}
