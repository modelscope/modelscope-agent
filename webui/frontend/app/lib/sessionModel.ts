import { useEffect } from 'react'
import { api } from '~/lib/api'
import { dispatchModelChanged } from '~/lib/modelChanged'

/**
 * Put a session back on the model it was last held with.
 *
 * The active model is a single global setting, so opening an old conversation
 * used to run it on whatever had been selected most recently anywhere else.
 * That is not a neutral default: it changes what the model can see (a text-only
 * model degrades every image in the history), and it discards the provider's
 * prefix cache for that conversation, so the next turn pays full price for a
 * context it had already warmed.
 *
 * Runs once per session id, and only when the stored model actually differs —
 * a redundant PUT would invalidate live agent runtimes for nothing.
 */
export function useRestoreSessionModel(modelId: string | undefined): void {
  useEffect(() => {
    if (!modelId) return
    let cancelled = false
    ;(async () => {
      try {
        const settings = await api.getAgentSettings()
        if (cancelled || settings.default_model_id === modelId) return
        await api.putAgentSettings({ ...settings, default_model_id: modelId })
        if (!cancelled) dispatchModelChanged()
      } catch {
        // Best effort: failing to restore the model must not block the page.
      }
    })()
    return () => {
      cancelled = true
    }
  }, [modelId])
}
