import { Popover } from 'antd'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router'
import { useT } from '~/lib/i18n'
import type { Project, Skill } from '~/lib/types'
import { PillButton } from './PillButton'

interface SkillSelectorProps {
  items: Skill[]
  project?: Project | null
}

/**
 * Read-only view of the skills active for this chat. Enablement lives ONLY in
 * project settings and global settings — the composer has no per-chat toggles;
 * it lists what those settings resolved to and links to the right settings
 * page (project tab when a project is selected, else global).
 */
export function SkillSelector({ items, project }: SkillSelectorProps) {
  const { t } = useT()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)

  // Only skills enabled in global/project settings apply to the chat.
  const enabledItems = useMemo(() => items.filter((s) => s.enabled), [items])

  const settingsPath = project
    ? `/projects/${project.id}?tab=skills`
    : '/settings/mcp-skills?tab=skills'

  const content = (
    <div className="w-[min(280px,calc(100vw-32px))]">
      {/* Header: title only — enablement is settings-driven. With nothing
          enabled the title itself states that, so no list section follows. */}
      <div className="px-[14px] py-[14px]">
        {enabledItems.length ? (
          <span className="text-xs text-msa-text-2">
            {t.home.skillPopoverTitle}
          </span>
        ) : (
          <span className="text-xs font-medium text-msa-text-3">
            {t.home.skillPopoverEmpty}
          </span>
        )}
      </div>

      <div className="h-px bg-msa-line-1" />

      {/* List — omitted entirely when empty; a placeholder row would only
          repeat what the header already said. */}
      {enabledItems.length ? (
        <>
          <div className="max-h-[280px] overflow-y-auto p-[6px]">
            {enabledItems.map((it) => (
              <div
                key={it.id}
                className="flex items-center gap-2 rounded-[8px] px-[10px] py-2.5"
              >
                <span className="min-w-0 flex-1 truncate text-sm text-msa-text-1">
                  {it.name}
                </span>
              </div>
            ))}
          </div>

          <div className="h-px bg-msa-line-1" />
        </>
      ) : null}

      {/* Footer: settings link */}
      <div className="px-[14px] py-[14px]">
        <span
          className="cursor-pointer text-xs text-msa-text-brand1"
          onClick={() => {
            setOpen(false)
            navigate(settingsPath)
          }}
        >
          {t.home.skillSettings}
        </span>
      </div>
    </div>
  )

  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      trigger="click"
      classNames={{ container: 'p-0' }}
      content={content}
    >
      <PillButton
        open={open}
        icon={
          <span className="h-2 w-2 inline-block rounded-full bg-msa-green-5" />
        }
      >
        {enabledItems.length} {t.home.skillPill}
      </PillButton>
    </Popover>
  )
}
