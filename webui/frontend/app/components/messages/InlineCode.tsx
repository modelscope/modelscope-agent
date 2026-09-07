/** Inline-code style wrapper for dynamic entities inside step card headers
 * (file paths, tool/MCP names, skill names, search queries…) — the visual
 * analog of markdown backticks, on the msa fill token so it reads on both the
 * fill-1 shell cards and the fill-2 accordion headers. */
export function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code className="mx-0.5 rounded-md bg-msa-fill-3 px-1.5 py-0.5 align-middle font-mono text-[0.85em]">
      {children}
    </code>
  )
}

/** Layout for a one-line step header — a label plus an InlineCode chip naming
 * what the step acts on — where the chip takes the leftover width and clips ITS
 * OWN text with an ellipsis.
 *
 * Put on the antd `Typography.Text` that wraps such a header INSTEAD of its
 * `ellipsis` prop. That prop looks the same in the steady state (its CSS path
 * applies equivalent clipping to the chip), but on mount it renders a frame
 * through the JS measure path, which counts a React element as length 1 and
 * cannot cut it (`typography/Base/Ellipsis.js` `sliceNodes`) — so the chip is
 * dropped whole and the header flashes "Label ..." with the subject gone. Cards
 * remount mid-stream, which is how that frame ends up on screen. The Typography
 * itself stays: the chip inherits its `code` chrome (the hairline border).
 *
 * The child selectors keep this a wrapper-level fix — none of the call sites
 * that assemble these titles has to opt in, and neither does the next one.
 * `gap-1` stands in for the `{' '}` between label and chip, which stops
 * generating a box once the row is a flex container. Trailing extras (a result
 * count, stacked favicons) are spans too, so they hold their size and the chip
 * is what gives way — the opposite of `ellipsis`, which clipped them off the
 * end of the line. */
export const stepTitleLine =
  'flex min-w-0 flex-1 items-center gap-1 [&>span]:shrink-0 [&>code]:min-w-0 [&>code]:truncate'
