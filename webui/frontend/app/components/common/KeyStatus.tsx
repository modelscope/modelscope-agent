import { Popconfirm, Tooltip } from 'antd'
import type { ReactNode } from 'react'
import RefreshIcon from '~/assets/icons/refresh.svg?react'
import { IconButton } from './IconButton'

/* ================================================================
 * Shared UI for write-only credential fields.
 *
 * A stored API key is never echoed back to the client, so every place that
 * holds one needs the same two affordances: a tag saying whether one is on
 * file, and a way to take it back out again. Both are used by the search
 * settings page and the model-provider modal.
 * ================================================================ */

/**
 * "Configured" / "Not set" pill shown next to a credential field's label.
 *
 * Since the key itself can't be shown, this tag is the only signal that one is
 * stored. The box is fixed so the states differ in colour only: the labels can
 * be 2 characters ("optional") vs 3 ("configured" / "not set"), so a
 * content-sized tag changed width whenever the state flipped and nudged the
 * row. 42px = 3 glyphs at 10px + px-1.5.
 */
export function KeyStatusTag({
  /** Whether a credential is on file — drives the colour only. */
  set,
  children
}: {
  set: boolean
  children: ReactNode
}) {
  return (
    <span
      className={`inline-flex h-5 min-w-[42px] items-center justify-center rounded px-1.5 text-[10px] font-normal leading-none ${
        set
          ? 'bg-msa-fill-5 text-msa-text-brand1'
          : 'bg-msa-fill-2 text-msa-text-3'
      }`}
    >
      {children}
    </span>
  )
}

/**
 * Reset control for a credential field, meant for an `Input.Password` suffix.
 *
 * Without this, "configured" is a one-way door: the key can be replaced but
 * never taken back out, so a key pasted into the wrong place stays there for
 * good. Render it only when there is actually something to remove.
 */
export function KeyResetButton({
  /** Question — keep it short; the consequence belongs in `confirmDesc`. */
  confirmTitle,
  confirmDesc,
  okText,
  tooltip,
  onConfirm
}: {
  confirmTitle: string
  confirmDesc: string
  okText: string
  tooltip: string
  onConfirm: () => void | Promise<void>
}) {
  return (
    <Popconfirm
      // Question in the title, consequence in the description: as one string it
      // rendered as a single ~1100px line, since a Popconfirm sizes itself to
      // its longest text. The width cap keeps that true for a long provider
      // label too, instead of letting the label decide the layout.
      title={<span className="block max-w-[260px]">{confirmTitle}</span>}
      description={<span className="block max-w-[260px]">{confirmDesc}</span>}
      okText={okText}
      okButtonProps={{ danger: true }}
      onConfirm={onConfirm}
    >
      <Tooltip title={tooltip}>
        {/* xs (20px), not sm: an affix wrapper is as tall as its tallest
            child, so a 28px control made the field 38px once configured and
            32px before that — the block visibly jumped on save. Staying inside
            the 22px line box keeps the height constant either way, and a 14px
            glyph matches antd's own eye icon. */}
        <IconButton
          variant="ghost"
          size="xs"
          icon={<RefreshIcon className="h-3.5 w-3.5" />}
        />
      </Tooltip>
    </Popconfirm>
  )
}
