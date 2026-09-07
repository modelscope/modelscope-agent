import { Typography } from 'antd'
import { FileTypeIcon } from '~/components/common/FileCard'
import { useT } from '~/lib/i18n'
import { useArtifactDeletedMap, useWorkspaceFileSet } from '~/lib/workspaceFiles'
import { CollapsibleRows } from './CollapsibleRows'
import type { OnOpenFile } from './types'
import CollectionIcon from '~/assets/icons/collection.svg?react'
import JumpIcon from '~/assets/icons/jump.svg?react'

/** One compact file row inside the deliverables card — file-type glyph + name,
 * clickable with a hover bg, degrading to a disabled "deleted" row. Mirrors the
 * multi-file tool card's row (MultiFileRow) so a turn's outputs read the same
 * as a single tool that touched several files, just far denser than the old
 * full-size FileCard grid. */
function ArtifactRow({
  path,
  deleted,
  onOpen
}: {
  path: string
  deleted: boolean
  onOpen?: () => void
}) {
  const { t } = useT()
  const name = path.split('/').pop() || path
  const inner = (
    <>
      <span className="flex h-4 w-4 shrink-0 items-center justify-center text-msa-text-3">
        <FileTypeIcon name={name} className="h-4 w-4" />
      </span>
      <Typography.Text
        ellipsis={{ tooltip: { title: name } }}
        className={`min-w-0 flex-1 !text-sm ${
          deleted ? '!text-msa-text-3' : '!text-msa-text-1'
        }`}
      >
        {name}
      </Typography.Text>
    </>
  )
  if (deleted || !onOpen) {
    return (
      <div
        className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left ${
          deleted ? 'cursor-not-allowed' : ''
        }`}
      >
        {inner}
        {deleted && (
          <span className="shrink-0 text-xs text-msa-text-danger">
            {t.home.fileDeleted}
          </span>
        )}
      </div>
    )
  }
  return (
    <button
      type="button"
      onClick={onOpen}
      title={t.session.openInWorkspace}
      className="group flex w-full cursor-pointer items-center gap-2 rounded-lg border-0 bg-transparent px-2 py-1.5 text-left transition-colors hover:bg-msa-fill-4"
    >
      {inner}
      <JumpIcon className="h-4 w-4 shrink-0 text-msa-text-3" />
    </button>
  )
}

/**
 * The turn's deliverables: workspace files the agent wrote/edited during its
 * tool-call loop (`changed_files` from the loop_end boundary), rendered after
 * the summary as a titled bordered card ("N files changed") holding a compact
 * row list — the same dense framing the multi-file tool card uses, chosen over
 * the full-size FileCard grid which took far too much vertical space for a
 * many-file turn. Rows open the file in the workspace rail, and a file the user
 * has since deleted degrades to a disabled "deleted" row via the live workspace
 * path set. Past five files the tail folds behind a chevron, with the last
 * visible row fading out (see CollapsibleRows), so a many-file turn doesn't
 * bury the conversation.
 */
export function ArtifactFiles({
  paths,
  onOpenFile
}: {
  paths: string[]
  onOpenFile?: OnOpenFile
}) {
  const { t } = useT()
  const fileSet = useWorkspaceFileSet()
  const artifactDeleted = useArtifactDeletedMap()
  if (paths.length === 0) return null

  return (
    <div className="w-full max-w-full overflow-hidden rounded-xl border border-msa-line-1 bg-msa-fill-1">
      {/* Header: "<n> files changed". */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        <CollectionIcon className="h-5 w-5 shrink-0 text-msa-text-2" />
        <span className="text-sm font-medium text-msa-text-1">
          {t.chat.filesChanged.replace('{n}', String(paths.length))}
        </span>
      </div>
      {/* Body: compact rows, one file per line. `relative` anchors the folded
          chevron, which floats over the last row's faded tail. */}
      <div className="relative flex flex-col gap-0.5 px-1.5 pb-1.5">
        <CollapsibleRows>
          {paths.map((path) => {
            // Prefer the authoritative artifact ledger (a real `os.stat` under
            // the exact root the write was recorded against). Only fall back to
            // the fuzzy workspace-set membership when the ledger doesn't know
            // this path — a bare `!fileSet.has(path)` mis-flags live files as
            // deleted (curated/lagging listing, differing path roots).
            const ledger = artifactDeleted?.get(path)
            const deleted =
              ledger !== undefined
                ? ledger
                : fileSet
                  ? !fileSet.has(path)
                  : false
            return (
              <ArtifactRow
                key={path}
                path={path}
                deleted={deleted}
                onOpen={onOpenFile ? () => onOpenFile(path) : undefined}
              />
            )
          })}
        </CollapsibleRows>
      </div>
    </div>
  )
}
