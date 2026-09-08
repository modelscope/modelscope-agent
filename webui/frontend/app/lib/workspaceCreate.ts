import { ApiError, api } from './api'
import { dispatchWorkspaceChanged } from './events'

/** Why a create didn't happen. The caller words it — the sentences live in the
 * locale files, not in here. */
export type CreateFailure = 'invalid' | 'exists' | 'failed'

export type CreateResult =
  | { ok: true; path: string }
  | { ok: false; reason: CreateFailure }

/** Join a workspace directory ('' = root, a trailing slash tolerated — the
 * project overview tracks its browsed folder that way) and a name. */
export function joinWorkspacePath(dir: string, name: string): string {
  const base = dir.replace(/\/+$/, '')
  return base ? `${base}/${name}` : name
}

/**
 * What's wrong with `name` as a new entry in `dir`, or null when nothing is.
 *
 * Split out from the create below so a name the user can see is taken never
 * costs a round trip.
 */
function checkEntryName(
  dir: string,
  name: string,
  existingPaths: string[]
): CreateFailure | null {
  const trimmed = name.trim()
  if (!trimmed) return 'invalid'
  // Mirror of the backend's `_safe` guard: an absolute path or a `.`/`..` hop
  // resolves outside the workspace and comes back a 400. Catching it here turns
  // that into something the user can fix.
  const segments = trimmed.split('/')
  if (
    trimmed.startsWith('/') ||
    segments.some((s) => s === '' || s === '.' || s === '..')
  )
    return 'invalid'
  if (existingPaths.includes(joinWorkspacePath(dir, trimmed))) return 'exists'
  return null
}

/**
 * Create an empty file (or a folder) in a project workspace and broadcast it.
 *
 * Shared by every surface that offers it — the session rail's tree and the
 * project overview's file table, which routes there — so the name rules can't
 * drift between them.
 *
 * Nested names are deliberately accepted: `docs/notes.md` creates `docs/` on the
 * way, because the SDK's `write_file` mkdirs the parents. That is what an
 * editor's new-file row does, and refusing it here would be our own
 * restriction, not the backend's.
 */
export async function createWorkspaceEntry({
  projectId,
  dir,
  name,
  kind,
  existingPaths
}: {
  projectId: string
  dir: string
  name: string
  kind: 'file' | 'folder'
  /** Every path in the workspace, so a taken name is refused before the round
   *  trip instead of coming back as a server error. */
  existingPaths: string[]
}): Promise<CreateResult> {
  const trimmed = name.trim()
  const bad = checkEntryName(dir, trimmed, existingPaths)
  if (bad) return { ok: false, reason: bad }
  const path = joinWorkspacePath(dir, trimmed)
  try {
    // 409 is silenced: a name we can SEE is taken never gets here, so a conflict
    // means the listing was stale — reported below in our own words rather than
    // as a second, server-worded toast.
    await api.createWorkspaceFile(
      projectId,
      { path, kind, content: '' },
      { silent: [409] }
    )
  } catch (err) {
    return {
      ok: false,
      reason: err instanceof ApiError && err.status === 409 ? 'exists' : 'failed'
    }
  }
  // Broadcast instead of a local reload: every listener refreshes (the tree, the
  // overview table, chat file cards). Files carry their path so listeners merge
  // it optimistically, flipping a card to "exists" without awaiting its refetch.
  dispatchWorkspaceChanged(kind === 'file' ? [path] : undefined)
  return { ok: true, path }
}
