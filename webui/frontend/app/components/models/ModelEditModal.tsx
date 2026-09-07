import { App, AutoComplete, Form, Input, Modal, Select, Typography } from 'antd'
import { useEffect, useState } from 'react'
import { CodeEditor } from '~/components/common/CodeEditor'
import { MsaSwitch } from '~/components/common/MsaSwitch'
import { api } from '~/lib/api'
import { useT } from '~/lib/i18n'
import type { GenerationDefaults, Model, Provider } from '~/lib/types'

/**
 * Params the runtime applies on its own, rendered into the editor so the box
 * starts as a working example instead of an empty `{}`.
 *
 * Only the canonical `reasoning_effort` knob is pre-filled, never the lowered
 * wire form: writing the wire form back would freeze today's dialect into the
 * model's saved settings, and the whole point of the knob is that the SDK
 * re-decides the dialect per endpoint. `auto` means "no opinion", so saving it
 * unchanged is a genuine no-op.
 */
function draftParams(defaults: GenerationDefaults | null): string {
  return JSON.stringify(
    { reasoning_effort: defaults?.effort ?? 'auto' },
    null,
    2
  )
}

interface Props {
  open: boolean
  /** Provider currently selected when "Add Model" was clicked. */
  defaultProvider: Provider | null
  /** Existing model to edit, or null for create mode. */
  model: Model | null
  providers: Provider[]
  onClose: () => void
  onSaved: (model: Model) => void
}

interface FormValues {
  provider_id: string
  name: string
  display_name: string
  /**
   * Whether this model may be shown image attachments.
   *
   * Deliberately on the CREATE form rather than buried in advanced settings:
   * whether a model can read a picture is not predictable from its name (on one
   * provider `qwen3.8-max` can and `qwen3.7-max` cannot), so the person adding
   * the model is the one who knows. Left untouched it stays unset, and the SDK
   * decides — provider capability first, then learning from a refusal.
   */
  supports_vision: boolean
}

export function ModelEditModal({
  open,
  defaultProvider,
  model,
  providers,
  onClose,
  onSaved
}: Props) {
  const { t } = useT()
  const { message } = App.useApp()
  const [form] = Form.useForm<FormValues>()
  const [advancedJson, setAdvancedJson] = useState('{}')
  const [submitting, setSubmitting] = useState(false)
  const [modelOptions, setModelOptions] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(false)
  const [defaults, setDefaults] = useState<GenerationDefaults | null>(null)
  const [providerId, setProviderId] = useState('')

  const isEdit = !!model

  useEffect(() => {
    if (!open) return
    if (model) {
      form.setFieldsValue({
        provider_id: model.provider_id,
        name: model.name,
        display_name: model.display_name,
        supports_vision: model.supports_vision ?? false
      })
      setProviderId(model.provider_id)
      // An existing model keeps whatever it was saved with; only a model that
      // has never been configured gets the pre-filled draft (below, once the
      // effective defaults arrive).
      setAdvancedJson(
        Object.keys(model.advanced_params ?? {}).length
          ? JSON.stringify(model.advanced_params, null, 2)
          : ''
      )
    } else {
      const pid = defaultProvider?.id ?? providers[0]?.id ?? ''
      form.setFieldsValue({
        provider_id: pid,
        name: '',
        display_name: '',
        supports_vision: false
      })
      setProviderId(pid)
      setAdvancedJson('')
    }
    setDefaults(null)
  }, [open, model, defaultProvider, providers, form])

  // The dialect (and therefore what the box should show) is decided by the
  // provider's endpoint, so re-resolve whenever the provider changes. Failure
  // degrades to the plain empty object the box always had.
  useEffect(() => {
    if (!open || !providerId) return
    let cancelled = false
    api
      .getGenerationDefaults(providerId, model?.name ?? '', { silent: true })
      .then((d) => {
        if (cancelled) return
        setDefaults(d)
        setAdvancedJson((current) => (current ? current : draftParams(d)))
      })
      .catch(() => {
        if (!cancelled) setAdvancedJson((current) => current || '{}')
      })
    return () => {
      cancelled = true
    }
  }, [open, providerId, model])

  // Add mode only: pull available model ids from the provider's standard
  // /models endpoint for autocomplete. Failure degrades silently to empty.
  useEffect(() => {
    if (!open || model) {
      setModelOptions([])
      setLoadingModels(false)
      return
    }
    const providerId = defaultProvider?.id ?? providers[0]?.id
    if (!providerId) {
      setModelOptions([])
      return
    }
    let cancelled = false
    setLoadingModels(true)
    api
      .listProviderModels(providerId, { silent: true })
      .then((ids) => {
        if (!cancelled) setModelOptions(ids)
      })
      .catch(() => {
        if (!cancelled) setModelOptions([])
      })
      .finally(() => {
        if (!cancelled) setLoadingModels(false)
      })
    return () => {
      cancelled = true
    }
  }, [open, model, defaultProvider, providers])

  const submit = async () => {
    const v = await form.validateFields()
    let advanced: Record<string, unknown> = {}
    try {
      const parsed = JSON.parse(advancedJson || '{}')
      if (typeof parsed !== 'object' || Array.isArray(parsed)) {
        throw new Error('Must be a JSON object')
      }
      advanced = parsed
    } catch (e) {
      message.error(`${t.resources.jsonInvalid} (${(e as Error).message})`)
      return
    }
    setSubmitting(true)
    try {
      let saved: Model
      // The switch is a plain boolean, default off, and is always sent. It used
      // to need a "was it touched?" guard because an unset model stored `null`
      // ("let the SDK decide") which a Switch can only draw as OFF — so saving
      // an unrelated field could turn that display artifact into a real `false`.
      // There is no `null` state any more: unset simply means off, so writing
      // it back changes nothing and the guard is gone with it.
      if (model) {
        saved = await api.updateModel(model.id, {
          display_name: v.display_name,
          advanced_params: advanced,
          supports_vision: v.supports_vision
        })
      } else {
        saved = await api.createModel({
          provider_id: v.provider_id,
          name: v.name,
          display_name: v.display_name,
          advanced_params: advanced,
          supports_vision: v.supports_vision
        })
      }
      onSaved(saved)
    } catch {
      // API errors surface via the global toast (see root ApiErrorBridge).
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      open={open}
      onCancel={onClose}
      title={
        isEdit ? t.modelsAdmin.editModelTitle : t.modelsAdmin.addModelTitle
      }
      okText={t.modelsAdmin.confirm}
      cancelText={t.modelsAdmin.cancel}
      onOk={submit}
      okButtonProps={{ loading: submitting }}
      destroyOnHidden
      width={480}
    >
      <Form form={form} layout="vertical">
        <Form.Item
          label={t.modelsAdmin.apiProviderLabel}
          name="provider_id"
          rules={[{ required: true }]}
        >
          <Select
            disabled
            options={providers.map((p) => ({
              value: p.id,
              label: p.name,
              disabled: !p.enabled
            }))}
          />
        </Form.Item>
        <Form.Item
          label={t.modelsAdmin.modelName}
          name="name"
          rules={[{ required: true, max: 160 }]}
          extra={
            // No colour here: `Form.Item`'s extra slot already paints antd's
            // description colour — the same token `Typography type="secondary"`
            // uses for the hint under the JSON editor, so the two match without
            // naming a value. Only the size is overridden.
            <span className="text-[11px]">{t.modelsAdmin.modelNameHint}</span>
          }
        >
          {isEdit ? (
            <Input disabled />
          ) : (
            <AutoComplete
              options={modelOptions.map((id) => ({ value: id }))}
              showSearch={{
                filterOption: (input, option) =>
                  (option?.value ?? '')
                    .toString()
                    .toLowerCase()
                    .includes(input.toLowerCase())
              }}
              notFoundContent={
                loadingModels ? t.modelsAdmin.modelsLoading : null
              }
            >
              {/* Custom child input so we can turn OFF the browser's native
                  autofill dropdown (saved form-history values like "123"/"test"
                  otherwise overlap our model-id suggestion list). AutoComplete
                  still owns value/onChange via Form.Item; this only customizes
                  the rendered input. */}
              <Input
                placeholder={t.modelsAdmin.modelNamePlaceholder}
                autoComplete="off"
              />
            </AutoComplete>
          )}
        </Form.Item>
        <Form.Item
          label={t.modelsAdmin.modelDisplayNameLabel}
          name="display_name"
        >
          <Input />
        </Form.Item>
        <Form.Item
          label={t.modelsAdmin.supportsVisionLabel}
          name="supports_vision"
          valuePropName="checked"
          extra={
            <span className="text-[11px]">
              {t.modelsAdmin.supportsVisionHint}
            </span>
          }
        >
          <MsaSwitch />
        </Form.Item>
        <Form.Item label={t.modelsAdmin.advancedJson}>
          <div className="overflow-hidden rounded-md border border-msa-line-1">
            <CodeEditor
              value={advancedJson}
              onChange={setAdvancedJson}
              language="json"
              height={140}
            />
          </div>
          <Typography.Paragraph
            type="secondary"
            className="!mt-1 !mb-0 !text-[11px]"
          >
            {/* Two lines, not five: what you may set, then what is actually
                being sent. Everything else about vendor quirks lives in the
                user guide — a settings dialog is not the place to explain that
                neighbouring tiers collapse. */}
            {t.modelsAdmin.generationParamsHint}
            {defaults && (
              <>
                {' '}
                {t.modelsAdmin.thinkingKnobHint.replace(
                  '{options}',
                  defaults.effort_options.join(' / ')
                )}
                <br />
                {/* What the knob resolves to on THIS endpoint. Computed by the
                    same SDK call the request path uses, so it can't drift. */}
                {Object.keys(defaults.wire_params).length
                  ? t.modelsAdmin.thinkingWireOn.replace(
                      '{json}',
                      JSON.stringify(defaults.wire_params)
                    )
                  : t.modelsAdmin.thinkingWireNone}
                {defaults.extra_hint &&
                  ` · ${t.modelsAdmin.thinkingExtraHint.replace(
                    '{keys}',
                    defaults.extra_hint
                  )}`}
              </>
            )}
          </Typography.Paragraph>
        </Form.Item>
      </Form>
    </Modal>
  )
}
