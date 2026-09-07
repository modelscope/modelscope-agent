import { App, Button, Modal } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { CodeEditor } from '~/components/common/CodeEditor'
import { api } from '~/lib/api'
import { useT } from '~/lib/i18n'
import type { Mcp, Scope } from '~/lib/types'
import { fromMcpServers, toMcpServers } from './mcpJson'
import { ScopeBadge } from './ScopeBadge'

interface McpJsonModalProps {
  open: boolean
  scope: Scope
  scopeBadge?: string
  items: Mcp[]
  onSaved: () => void
  onClose: () => void
}

/**
 * Edits a whole scope's MCP servers as one `mcp.json` document, in a dialog.
 *
 * It used to replace the card list in place, which meant the list — the thing
 * the document is a view OF — was gone while editing it, and the surrounding
 * toolbar had to be disabled by hand to keep the two editing paths apart. A
 * modal gets both for free: the cards stay visible behind it, and its mask is
 * what stops anyone from adding a server underneath a pending document.
 */
export function McpJsonModal({
  open,
  scope,
  scopeBadge,
  items,
  onSaved,
  onClose
}: McpJsonModalProps) {
  const { t } = useT()
  const { message } = App.useApp()
  const original = useMemo(
    () => JSON.stringify(toMcpServers(items), null, 2),
    [items]
  )
  const [text, setText] = useState(original)
  const [saving, setSaving] = useState(false)

  // Reseed on every open, not just when `items` change: this component outlives
  // the dialog (only the body is destroyed), so an abandoned edit would still be
  // sitting in `text` the next time the dialog is opened.
  useEffect(() => {
    if (open) setText(original)
  }, [open, original])

  const dirty = text !== original

  const save = async () => {
    let parsed
    try {
      parsed = fromMcpServers(text)
    } catch (e) {
      message.error(`${t.resources.jsonInvalid} (${(e as Error).message})`)
      return
    }
    setSaving(true)
    try {
      // ONE atomic call. Deleting every server and re-creating them from the
      // document (what this did before) meant a rename left the old server
      // behind whenever its delete was lost to a concurrent one, and a single
      // rejected entry wiped the whole scope, since the deletes had landed.
      await api.replaceMcps(
        scope,
        parsed.map((m) => ({ ...m, scope }))
      )
      onSaved()
      // Dismiss on success — the dialog's only other way out is "cancel", which
      // would read as "discard" right after a save that already landed.
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      // Saving replaces every server in ONE scope, so the title says which —
      // same badge the add/edit dialog carries.
      title={
        <div className="flex items-center gap-2">
          <span>{t.resources.viaJson}</span>
          <ScopeBadge>{scopeBadge}</ScopeBadge>
        </div>
      }
      // The three buttons this view has always had, now the dialog's footer.
      footer={
        <>
          <Button onClick={onClose}>{t.resources.cancel}</Button>
          <Button disabled={!dirty} onClick={() => setText(original)}>
            {t.resources.jsonReset}
          </Button>
          <Button
            type="primary"
            loading={saving}
            disabled={!dirty}
            onClick={save}
          >
            {t.resources.jsonSave}
          </Button>
        </>
      }
      // Unmounts monaco with the dialog instead of keeping an editor alive for
      // every scope the user has ever opened.
      destroyOnHidden
      width={720}
    >
      <div className="overflow-hidden rounded-xl border border-msa-line-1">
        <CodeEditor
          value={text}
          onChange={setText}
          language="json"
          height={460}
        />
      </div>
    </Modal>
  )
}
