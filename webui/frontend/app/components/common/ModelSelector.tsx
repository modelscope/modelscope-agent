import { CheckOutlined } from '@ant-design/icons'
import { Popover } from 'antd'
import { Fragment, useMemo, useState } from 'react'
import { ProviderTags } from '~/components/models/ProviderTags'
import { useT } from '~/lib/i18n'
import type { AgentSettings, Model, Provider } from '~/lib/types'
import { PillButton } from './PillButton'
import { EmptyState } from './EmptyState'
import { DeferredSkeleton } from './DeferredSkeleton'
import './ModelSelector.css'
import JumpIcon from '~/assets/icons/jump.svg?react'

interface ModelSelectorProps {
  /** null while the lists are still loading — the panel then shows a
   * skeleton instead of an "empty" state that isn't true yet. */
  models: Model[] | null
  providers: Provider[] | null
  settings: AgentSettings | null
  onSelectModel: (providerId: string, modelId: string) => void
}

export function ModelSelector({
  models,
  providers,
  settings,
  onSelectModel
}: ModelSelectorProps) {
  const { t } = useT()
  const [open, setOpen] = useState(false)
  const [activeProviderId, setActiveProviderId] = useState<string | null>(null)

  const defaultModel = useMemo(
    () => models?.find((m) => m.id === settings?.default_model_id) ?? null,
    [models, settings]
  )

  // Sync active provider with the current default provider when opening.
  const effectiveProviderId =
    activeProviderId ??
    defaultModel?.provider_id ??
    settings?.default_provider_id ??
    providers?.[0]?.id ??
    null

  const activeProvider = useMemo(
    () => providers?.find((p) => p.id === effectiveProviderId) ?? null,
    [providers, effectiveProviderId]
  )

  const providerModels = useMemo(
    () =>
      (models ?? [])
        .filter((m) => m.provider_id === effectiveProviderId)
        .sort((a, b) =>
          (a.display_name || a.name).localeCompare(b.display_name || b.name)
        ),
    [models, effectiveProviderId]
  )

  const handleSelectModel = (model: Model) => {
    onSelectModel(model.provider_id, model.id)
    setOpen(false)
  }

  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      trigger="click"
      classNames={{
        container: 'p-0'
      }}
      content={
        <div className="flex h-[300px] w-[min(661px,calc(100vw-32px))] ">
          {/* Left: providers.
              Fixed width, not shrink-0 alone: without a cap the column expands
              to fit the widest provider name, which squeezes the models column
              (min-w-0 flex-1) down to a few characters. 280px keeps common
              provider names on one line next to their status tags, while the
              row's own `truncate` handles longer canonical or custom names.
              The models column keeps ~380px, still ample for model names. */}
          <div className="flex w-[280px] shrink-0 flex-col gap-1 border-r border-msa-line-1 p-[6px] h-full overflow-y-auto">
            {(providers ?? []).map((p) => {
              const selected = p.id === effectiveProviderId
              return (
                <button
                  key={p.id}
                  type="button"
                  title={p.name}
                  onClick={() => setActiveProviderId(p.id)}
                  className={`flex w-full cursor-pointer items-center gap-2 rounded-[8px] border-0 px-[10px] py-[13px] text-left text-sm font-medium transition-colors ${
                    selected
                      ? 'bg-msa-fill-4 text-msa-text-brand1'
                      : 'bg-msa-fill-0 text-msa-text-1 hover:bg-msa-fill-4 hover:text-msa-text-brand1'
                  }`}
                >
                  {/* No `flex-1` on the name: it would claim the row's slack and
                      push the tags over to the arrow, reading as if they
                      belonged to it. Shrinking (the flex default) still lets
                      `truncate` cut a long name, and `ml-auto` keeps the arrow
                      pinned right. Same tags as the settings provider list, so
                      "built-in" and "key on file" mean the same thing here. */}
                  <span className="min-w-0 truncate">{p.name}</span>
                  <ProviderTags provider={p} />
                  <JumpIcon className="ml-auto h-[15px] w-[15px] shrink-0 text-msa-text-3" />
                </button>
              )
            })}
          </div>

          {/* Right: models */}
          <div className="min-w-0 flex-1 overflow-y-auto p-[6px]">
            {models === null || providers === null ? (
              <DeferredSkeleton rows={5} className="p-2" />
            ) : activeProvider ? (
              providerModels.length === 0 ? (
                <div className="flex h-full items-center justify-center">
                  <EmptyState
                    size="sm"
                    description={t.modelsAdmin.modelsEmpty}
                  />
                </div>
              ) : (
                <div className="msel-list flex flex-col">
                  {providerModels.map((m, idx) => {
                    const selected = m.id === settings?.default_model_id
                    return (
                      <Fragment key={m.id}>
                        {idx > 0 && (
                          <div className="msel-divider mx-[10px] my-[1px] h-px bg-msa-line-1 transition-opacity" />
                        )}
                        <button
                          type="button"
                          title={m.display_name || m.name}
                          onClick={() => handleSelectModel(m)}
                          className={`flex w-full cursor-pointer items-center justify-between gap-2 rounded-[8px] border-0 px-[10px] py-[13px] text-left text-sm font-medium transition-colors ${
                            selected
                              ? 'msel-selected bg-msa-fill-4 text-msa-text-brand1'
                              : 'bg-msa-fill-0 text-msa-text-1 hover:bg-msa-fill-4'
                          }`}
                        >
                          <span className="flex min-w-0 flex-1 flex-col">
                            <span className="truncate">
                              {m.display_name || m.name}
                            </span>
                            {m.display_name && (
                              <span className="truncate text-xs font-normal text-msa-text-3">
                                {m.name}
                              </span>
                            )}
                          </span>
                          {selected && (
                            <CheckOutlined className="shrink-0 h-[15px] w-[15px] text-msa-text-brand1" />
                          )}
                        </button>
                      </Fragment>
                    )
                  })}
                </div>
              )
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-msa-text-3">
                —
              </div>
            )}
          </div>
        </div>
      }
    >
      <PillButton
        open={open}
        icon={
          <span className="h-2 w-2 inline-block rounded-full bg-msa-purple-5" />
        }
      >
        {defaultModel?.display_name ?? t.home.modelUnset}
      </PillButton>
    </Popover>
  )
}
