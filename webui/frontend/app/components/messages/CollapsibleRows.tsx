import { Children } from 'react'
import { useState } from 'react'
import './CollapsibleRows.css'
import ExpandIcon from '~/assets/icons/expand.svg?react'

/** Rows shown before the list folds. */
const VISIBLE_LIMIT = 5

/**
 * Dense row list that shows at most {@link VISIBLE_LIMIT} rows and folds the
 * rest behind a chevron. A turn that touched a dozen files otherwise rendered a
 * dozen lines inline and pushed the rest of the conversation off screen — the
 * file count in the card header already tells the whole story, so the tail only
 * needs to be reachable, not always visible.
 *
 * While folded the LAST visible row fades out (see CollapsibleRows.css) and the
 * chevron covers that whole row as an absolutely positioned overlay, so folding
 * costs no extra height and the entire row width is the hit area. That row's own
 * file is therefore not openable while folded — an accepted trade for a
 * comfortable target, since expanding brings it back.
 *
 * The chevron carries no label of any kind, by request: no Tooltip, and no
 * aria-label either, since nothing else in this codebase labels its icon buttons
 * that way (the only aria-* in use marks decorative icons hidden).
 *
 * Note the overlay carries no background except on hover: an earlier attempt
 * painted it with the card's own colour to veil the row, which needs to know
 * `--msa-fill-1` and therefore broke per theme (a light grey band in dark mode).
 * The mask does the veiling; the overlay only carries the glyph.
 *
 * Expanded, the chevron returns to a normal in-flow row — overlaying there
 * would cover a fully-visible file.
 *
 * Lists at or below the limit render exactly as before: no toggle, no fade, no
 * wrapper markup, so short lists are untouched.
 *
 * Shared by the turn's deliverables card (ArtifactFiles) and the multi-file
 * tool card (MultiFileStepCard) — the two are deliberately the same construct,
 * so folding only one of them would read as a bug. Both parents are `relative`
 * so the folded chevron anchors to them.
 */
export function CollapsibleRows({ children }: { children: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false)
  // toArray drops null/undefined children and assigns stable keys, so the
  // caller can map freely without the count drifting from what's rendered.
  const rows = Children.toArray(children)
  const folded = rows.length > VISIBLE_LIMIT
  if (!folded) return <>{rows}</>

  const shown = expanded ? rows : rows.slice(0, VISIBLE_LIMIT)
  const lastIndex = shown.length - 1

  return (
    <>
      {shown.map((row, i) =>
        !expanded && i === lastIndex ? (
          // Wrapper carries the mask so the row component itself stays generic
          // (both call sites pass their own row type).
          <div key={`fade-${i}`} className="cr-fade">
            {row}
          </div>
        ) : (
          row
        )
      )}
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        className={
          expanded
            ? 'group flex w-full cursor-pointer items-center justify-center rounded-lg border-0 bg-transparent py-0.5 hover:bg-msa-fill-4'
            : // Inset by the container's own padding (6px) and 32px tall so the
              // box coincides exactly with the faded row — an absolutely
              // positioned element resolves against the padding box, so
              // `inset-x-0` would overhang the rows by that padding on each side.
              //
              // No `transition-colors`: switching between absolute (folded) and
              // static (expanded) is a layout jump — a colour transition on that
              // same frame produces a visible hover-flash as the old button
              // fades out at its previous position.
              'group absolute inset-x-1.5 bottom-1.5 flex h-8 cursor-pointer items-center justify-center rounded-lg border-0 bg-transparent hover:bg-msa-fill-4'
        }
      >
        {/* Unrotated points up (collapse), rotate-180 points down (expand) —
            the same orientation contract the sidebar and composer use. */}
        <ExpandIcon
          className={`h-4 w-4 text-msa-text-3 transition-transform group-hover:text-msa-text-1 ${
            expanded ? '' : 'rotate-180'
          }`}
        />
      </button>
    </>
  )
}
