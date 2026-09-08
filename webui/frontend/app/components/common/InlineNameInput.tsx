import { useEffect, useRef, useState } from 'react'

/**
 * The name editor a file row shows in place of its name: used both to rename an
 * existing entry and to name a brand-new one — an empty row added to the list,
 * the way an editor creates files (no dialog).
 *
 * Autofocuses and pre-selects the base name, excluding the extension, so typing
 * replaces `notes` and leaves `.md` alone. Enter/blur commits, Escape cancels; a
 * `done` guard prevents Escape's blur from also committing.
 */
export function InlineNameInput({
  initial,
  onCommit,
  onCancel
}: {
  initial: string
  onCommit: (value: string) => void
  onCancel: () => void
}) {
  const [value, setValue] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)
  const done = useRef(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.focus()
    const dot = initial.lastIndexOf('.')
    if (dot > 0) el.setSelectionRange(0, dot)
    else el.select()
  }, [initial])
  const commit = () => {
    if (done.current) return
    done.current = true
    onCommit(value)
  }
  const cancel = () => {
    if (done.current) return
    done.current = true
    onCancel()
  }
  return (
    <input
      ref={ref}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === 'Enter') {
          e.preventDefault()
          commit()
        } else if (e.key === 'Escape') {
          e.preventDefault()
          cancel()
        }
      }}
      onBlur={commit}
      // Loud on purpose: this row is the one thing waiting on the user, and a
      // bare <input> would otherwise be the quietest thing on screen — the
      // app-wide focus ring opts text fields out (app.css), since antd draws
      // its own, and this field is not an antd one. Brand border plus the same
      // pale fill the tree uses for the active row.
      className="mr-2 min-w-0 flex-1 rounded border border-msa-text-brand1 bg-msa-bg-1 px-1 text-sm text-msa-text-1 ring-2 ring-msa-fill-4 outline-none"
    />
  )
}
