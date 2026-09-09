import { Tooltip } from 'antd'
import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type RefObject
} from 'react'
import type { AgentMessage } from '~/lib/agentProvider'
import type { ChatMessageItem, MessageListHandle } from './MessageList'
import './MessageNavRail.css'

/** DOM hook the rail uses to locate a turn's bubble when tracking the reading
 * position. Stamped by `UserBubble`; kept here so the producer and the only
 * consumer name it from one place. */
export const MSG_KEY_ATTR = 'data-msa-msg-key'

/** Width a label expands to, in px. Single source of truth: handed to the
 * stylesheet as a custom property AND used to decide whether a label is
 * ellipsised (and therefore needs a tooltip). */
const LABEL_WIDTH = 208

/**
 * One-line label for a user turn. Mirrors `UserBubble`'s content precedence so
 * the rail says what the bubble says: a segment array (skill pills read as
 * `/name`) wins over plain content, and a turn that is only attachments falls
 * back to their file names instead of showing an empty row.
 */
function labelOf(message: AgentMessage): string {
  const raw = message.segments?.length
    ? message.segments
        .map((s) => (s.type === 'skill' ? `/${s.name || s.id}` : s.text))
        // Space at every segment boundary: the bubble spaces its pills apart
        // with margin, which a flat string has no equivalent for — joining on
        // '' ran the pill straight into the prompt after it (`/docker-expert`
        // + `node...` read as one word, `/docker-expertnode`). Doubles are
        // collapsed just below, so this cannot introduce gaps.
        .join(' ')
    : message.content || (message.files ?? []).map((f) => f.name).join(', ')
  // Rows are single-line, so the newlines a pasted/multi-line prompt carries
  // would otherwise render as gaps inside the ellipsised label.
  return raw.replace(/\s+/g, ' ').trim()
}

function NavRow({
  label,
  active,
  onJump
}: {
  label: string
  active: boolean
  onJump: () => void
}) {
  const labelRef = useRef<HTMLSpanElement>(null)
  const [clipped, setClipped] = useState(false)
  return (
    // Only an ellipsised label gets a tooltip — an empty title renders no
    // tooltip at all, so short rows ("5", "继续") stay quiet.
    <Tooltip
      title={clipped ? label : ''}
      placement="right"
      mouseEnterDelay={0.3}
    >
      <button
        type="button"
        className={`mnr-row${active ? ' mnr-row-active' : ''}`}
        onMouseEnter={() => {
          // `scrollWidth` is the FULL text width whatever the open/close
          // animation is doing (the label never wraps), so comparing it with
          // the final width is stable even mid-expand — comparing it with the
          // live `clientWidth` would misjudge every row entered early.
          const el = labelRef.current
          setClipped(!!el && el.scrollWidth > LABEL_WIDTH + 1)
        }}
        onClick={(e) => {
          // A pointer click leaves the button focused, which kept the panel
          // pinned open after the cursor left (it stayed matched by the
          // focus rule in the stylesheet). `detail > 0` marks a real pointer
          // click; keyboard activation reports 0 and MUST keep its focus, or
          // tabbing would close the panel it is navigating.
          if (e.detail > 0) e.currentTarget.blur()
          onJump()
        }}
      >
        <span className="mnr-dash" />
        <span ref={labelRef} className="mnr-label">
          {label}
        </span>
      </button>
    </Tooltip>
  )
}

/**
 * Table of contents for the conversation, pinned to the message list's left
 * gutter: one tick per user turn, collapsed to a bare column of dashes and
 * expanding on hover into the questions themselves. Clicking a row jumps the
 * list to that turn.
 *
 * Rendered into a box that spans the FULL width of the chat panel but only the
 * MESSAGE LIST's height, so "pinned to the far left, centred on the list, capped
 * at a share of its height" is plain CSS (`inset-y-0` + `items-center` + a
 * percentage `max-height`). Two earlier attempts got this wrong and are worth
 * not repeating: measuring the scroll box and writing `top` from JS centred on
 * whatever `offsetParent` reported (not the real containing block once an
 * ancestor establishes one by `transform`) and re-rendered on every layout
 * change, so the rail drifted while a reply streamed; hosting it inside the
 * CENTRED content column instead made it follow that column inward on wide
 * screens, leaving it floating in the middle of the conversation.
 *
 * The ticks stay put when the panel opens (label to the RIGHT of the tick)
 * rather than being pushed across by the growing labels: the tick under the
 * cursor is the hover target, and moving it out from under the pointer makes
 * the panel collapse and re-open on its own.
 */
export function MessageNavRail({
  items,
  listRef
}: {
  items: ChatMessageItem[]
  listRef: RefObject<MessageListHandle | null>
}) {
  const panelRef = useRef<HTMLDivElement>(null)
  const [activeId, setActiveId] = useState<string | null>(null)

  const turns = items
    .filter(({ message }) => message.role === 'user')
    .map(({ id, message }) => ({ id, label: labelOf(message) || '…' }))

  useEffect(() => {
    let frame = 0
    let box: HTMLElement | null = null
    let attempts = 0
    const keys = turns.map((turn) => turn.id)

    /** The turn's BUBBLE ROOT — the element `scrollTo({ key })` aligns to the
     * top of the viewport. Reading the inner content div instead left the
     * highlight one row behind after a jump, because the bubble's own padding
     * sits above that div and pushed it past the fold test below. */
    const bubbleOf = (key: string) => {
      const el = box?.querySelector(`[${MSG_KEY_ATTR}="${CSS.escape(key)}"]`)
      return el?.closest('.ant-bubble') ?? el
    }

    const track = () => {
      if (!box) return
      const boxTop = box.getBoundingClientRect().top
      let current: string | null = null
      for (const key of keys) {
        const el = bubbleOf(key)
        if (!el) continue
        // Bubbles are in ascending document order, so the last one still at or
        // above the fold is the turn being read; everything after is below it.
        if (el.getBoundingClientRect().top - boxTop <= 8) current = key
        else break
      }
      setActiveId(current ?? keys[0] ?? null)
    }

    const schedule = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(track)
    }

    // Bubble.List publishes its scroll box through React state, so it is absent
    // on the first commit. Retry for a few frames rather than giving up: a
    // replayed session never changes `items`, and a one-shot attach would leave
    // the rail unhighlighted there.
    const attach = () => {
      box = listRef.current?.getScrollBox() ?? null
      if (!box) {
        if (attempts++ > 60) return
        frame = requestAnimationFrame(attach)
        return
      }
      track()
      box.addEventListener('scroll', schedule, { passive: true })
    }
    attach()

    return () => {
      cancelAnimationFrame(frame)
      box?.removeEventListener('scroll', schedule)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [items, listRef])

  // Keep the active tick inside the capped window. The collapsed panel clips
  // instead of scrolling, so without this the "you are here" mark could sit
  // outside the visible column in a long conversation — an `overflow: hidden`
  // box still scrolls programmatically, so the window simply follows along.
  useEffect(() => {
    if (!activeId) return
    panelRef.current
      ?.querySelector('.mnr-row-active')
      ?.scrollIntoView({ block: 'nearest' })
  }, [activeId])

  // A single question needs no navigator — the whole turn is already on screen.
  if (turns.length < 2) return null

  return (
    <nav
      // Spans the list's full height only to centre the panel inside it, and the
      // panel's full width once open, so it must not swallow pointer events over
      // that area: the strip is click-through (see the stylesheet) and the panel
      // itself is the only hover target.
      className="mnr-root absolute inset-y-0 left-1 z-20 flex items-center"
    >
      <div
        ref={panelRef}
        className="mnr-panel"
        style={{ '--mnr-label-w': `${LABEL_WIDTH}px` } as CSSProperties}
      >
        {turns.map((turn) => (
          <NavRow
            key={turn.id}
            label={turn.label}
            active={turn.id === activeId}
            onJump={() => listRef.current?.scrollToKey(turn.id)}
          />
        ))}
      </div>
    </nav>
  )
}
