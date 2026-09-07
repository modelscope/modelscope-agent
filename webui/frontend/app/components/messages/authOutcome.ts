import type { AgentStep } from '~/lib/agentProvider'
import type { Dict } from '~/lib/i18n'

/**
 * Header wording for a call that was refused.
 *
 * An ask that simply expired says so instead of reading "rejected", which
 * claims a decision somebody took — and that difference is the whole answer to
 * "why didn't this run?": a timeout means approve the retry faster, a rejection
 * means it was refused on purpose. The backend marks it (`reason: "timeout"`,
 * see the runtime's permission handler) on the live announcement AND in the
 * persisted record, so a replayed card keeps the distinction.
 *
 * Refusals with no such mark (a rule denied the call, so nobody was ever asked;
 * or a pre-`reason` session log) keep the plain rejected wording.
 */
export function deniedNote(t: Dict, meta: AgentStep['meta']): string {
  return String(meta.reason ?? '') === 'timeout'
    ? t.chat.authTimedOut
    : t.chat.authRejected
}
