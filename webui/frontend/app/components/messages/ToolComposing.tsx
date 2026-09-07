import { useT } from '~/lib/i18n'
import type { AgentPart } from '~/lib/agentProvider'
import { toolActionLabel } from '~/lib/toolLabels'
import TodoIcon from '~/assets/icons/todo.svg?react'

type StepPart = Extract<AgentPart, { kind: 'step' }>

/**
 * "The model is writing a tool call" — the one state the turn used to render as
 * nothing at all.
 *
 * Between the assistant's last words and the first tool card, the model streams
 * the call's arguments, and every byte of a written file travels in there. For a
 * round that wrote five long documents that gap measured over a minute of blank
 * space below the text, with only the global "processing Ns" ticker moving.
 *
 * Deliberately the lightest thing that removes the blank: one muted row that
 * borrows the tool-round header's icon and typography, so it reads as the same
 * object the "used N tools" accordion is about to become — not as a new kind of
 * card the user has to learn. It is replaced the instant a real step arrives.
 */
export function ToolComposing({ steps }: { steps: StepPart[] }) {
  const { t } = useT()
  if (!steps.length) return null

  // Name only when the round is a single call: listing five identical
  // "write_file"s says less than the count does.
  const names = [
    ...new Set(steps.map((s) => String(s.step.meta.tool ?? '')).filter(Boolean))
  ]
  const label =
    steps.length === 1 && names.length === 1
      ? // What the tool does, not what it is called: the wire name is a
        // `server---leaf` pair the user never chose and cannot read
        // ("todo_list---todo_write"), and this row is the one place a builtin's
        // raw name used to reach the screen — every real card titles itself
        // from the backend's step kind instead.
        t.chat.toolComposingOne.replace('{tool}', toolActionLabel(t, names[0]))
      : t.chat.toolComposing.replace('{n}', String(steps.length))

  return (
    <div
      className="flex items-center gap-1.5 text-sm font-medium text-msa-text-3"
      aria-live="polite"
    >
      <TodoIcon className="h-5 w-5 shrink-0" />
      <span>{label}</span>
      {/* Three dots on the same pulse as the rest of the streaming UI. */}
      <span className="inline-flex gap-0.5">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1 w-1 animate-pulse rounded-full bg-msa-text-3"
            style={{ animationDelay: `${i * 150}ms` }}
          />
        ))}
      </span>
    </div>
  )
}
