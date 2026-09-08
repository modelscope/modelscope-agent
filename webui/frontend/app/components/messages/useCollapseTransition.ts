import { useCallback, useRef, useState } from 'react'
import type { TransitionEvent } from 'react'

/**
 * Height-collapse animation state for the grid `1fr`/`0fr` accordion trick,
 * scoped so a collapsible animates ONLY its own expand/collapse.
 *
 * The resolved value of `grid-template-rows: 1fr` is a PIXEL height, so a live
 * `transition` on that property re-fires whenever the open row's content
 * resizes — most visibly when a NESTED collapsible expands. Each open ancestor
 * then runs a second, lagging height animation that chases the descendant's
 * mid-flight height; stacked, those reads as a subtle vertical jitter.
 *
 * So we keep the transition present only in the window around THIS element's
 * own toggle and remove it once the toggle settles: while merely open the row
 * has no transition and snaps to fit its content every frame, following a
 * descendant's animation smoothly instead of racing it.
 *
 * Usage — gate the transition utility on `animating` and forward the handler:
 *
 *   const { animating, onTransitionEnd } = useCollapseTransition(expanded)
 *   <div
 *     className={`grid duration-200 ease-in-out ${animating ? 'transition-[grid-template-rows]' : ''}`}
 *     style={{ gridTemplateRows: expanded ? '1fr' : '0fr' }}
 *     onTransitionEnd={onTransitionEnd}
 *   >
 */
export function useCollapseTransition(expanded: boolean): {
  animating: boolean
  onTransitionEnd: (e: TransitionEvent) => void
} {
  const prev = useRef(expanded)
  const [animating, setAnimating] = useState(false)

  // Detect the toggle DURING render (before paint), so the transition class is
  // committed in the SAME update as the `expanded` change — a transition only
  // starts when the property is transitionable in the after-change style, so
  // turning it on now (not in a post-paint effect, which would let the value
  // jump un-animated first) is what makes the element's own toggle animate.
  // This runs for every `expanded` change (header click AND auto-collapse).
  if (prev.current !== expanded) {
    prev.current = expanded
    if (!animating) setAnimating(true)
  }

  // Settle once the element's OWN height transition ends. Descendant transitions
  // bubble here too, so ignore anything whose target isn't this grid container.
  const onTransitionEnd = useCallback((e: TransitionEvent) => {
    if (e.target === e.currentTarget) setAnimating(false)
  }, [])

  return { animating, onTransitionEnd }
}
