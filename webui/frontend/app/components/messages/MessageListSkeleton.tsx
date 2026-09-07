import { Skeleton } from 'antd'
import { DeferredSkeleton } from '~/components/common/DeferredSkeleton'
import type { AgentMessage } from '~/lib/agentProvider'
import type { ChatMessageItem } from './MessageList'

/**
 * One skeleton row mimicking a chat bubble: a rounded content block, no avatar
 * (the real Bubble.List renders none). `mine` flips it to the right (user)
 * side, matching the real placement (assistant start / user end).
 */
function SkeletonRow({
  mine,
  rows,
  width
}: {
  mine?: boolean
  rows: number
  /** CSS width value (e.g. "60%") — a fraction of the column so it tracks the
   * real bubbles, which are responsive (assistant body is `w-full`). */
  width: string
}) {
  return (
    <div className={`flex items-start ${mine ? 'justify-end' : ''}`}>
      <div className="rounded-2xl bg-msa-fill-2 px-4 py-3" style={{ width }}>
        <Skeleton active title={false} paragraph={{ rows }} />
      </div>
    </div>
  )
}

// Chars that roughly fill one line of the message column. Used to turn a
// message's text length into a line count and a width fraction.
const CHARS_PER_LINE = 80

/** Approximate the bubble's line count from its content length. Capped at 8:
 * `len / CHARS_PER_LINE` is a crude proxy (markdown structure isn't chars), and
 * the skeleton is bottom-anchored + clipped, so a block taller than the viewport
 * buys nothing. 8 is enough to read as "a long message". */
function estimateRows(len: number): number {
  return Math.max(1, Math.min(8, Math.ceil(len / CHARS_PER_LINE)))
}

/** Bubble width as a percentage of the column, grown from content length.
 * Assistant reaches 100% (its real body is `w-full`, wrapping at column width);
 * a user bubble is right-aligned and content-sized, so it caps lower. A small
 * floor keeps the shortest messages from becoming slivers. */
function estimateWidth(len: number, mine: boolean): string {
  const maxPct = mine ? 72 : 100
  const floorPct = 14
  // ~4 lines' worth of chars spans the full width; shorter scales down.
  const pct = (len / (CHARS_PER_LINE * 4)) * 100
  return `${Math.min(maxPct, Math.max(floorPct, Math.round(pct)))}%`
}

/** Visible text length of a message. `content` alone misses most bubbles: user
 * turns are a couple of chars, and an assistant answer often lives in its parts
 * (summary/text) with an empty `content` — so sizing off `content` collapsed
 * nearly every row to the minimum, making them look identical. Fold in the
 * text-bearing parts so the skeleton's rows/width actually vary per message. */
function messageTextLen(message: AgentMessage): number {
  let len = message.content?.length ?? 0
  for (const part of message.parts ?? []) {
    if (part.kind === 'text' || part.kind === 'thought') {
      len += part.text?.length ?? 0
    }
  }
  return len
}

/**
 * Loading placeholder for the chat message list, shown while history hydrates
 * (see ChatPanel). Renders one skeleton bubble per real message so the count
 * and left/right placement (by `role`) match the conversation about to appear.
 * Occupies the same flex-1 slot as MessageList (the surrounding chat layout
 * provides the centered column and the SSR-safe composer stays mounted below).
 */
export function MessageListSkeleton({ items }: { items: ChatMessageItem[] }) {
  return (
    // The gate occupies the flex-1 slot of the chat layout (same as MessageList).
    // Inside it, the skeleton container uses a pure-CSS trick to auto-adapt:
    //  - flex-initial (0 1 auto): height equals content when short, SHRINKS to
    //    the available space when content exceeds it (no grow, so short content
    //    doesn't fill the entire gate — it stays top-anchored with space below).
    //  - max-h-full: caps the container at the gate's height.
    //  - min-h-0: allows shrinking below intrinsic height.
    //  - overflow-hidden: clips the overflowing top.
    //  - justify-end: always pins rows to the bottom — but when content fits
    //    (container height === content height), there's no extra space, so
    //    justify-end is effectively a no-op and rows appear top-anchored.
    // Result: no JS needed, SSR-safe, both short and long conversations render
    // correctly on the very first frame.
    <DeferredSkeleton className="flex min-h-0 flex-1 flex-col">
      <div className="flex max-h-full min-h-0 flex-initial flex-col justify-end gap-6 overflow-hidden py-6">
        {items.map(({ id, message }) => {
          const mine = message.role === 'user'
          const len = messageTextLen(message)
          return (
            <SkeletonRow
              key={id}
              mine={mine}
              rows={estimateRows(len)}
              width={estimateWidth(len, mine)}
            />
          )
        })}
      </div>
    </DeferredSkeleton>
  )
}
