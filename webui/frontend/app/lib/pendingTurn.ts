import type { AgentMessage } from "./agentProvider";

/**
 * The question of a turn in flight, held OUTSIDE the React tree.
 *
 * For the first second or two of a turn (longer on a cold session) that row
 * exists nowhere a remounting view can read it. The server does not have it
 * yet — the SDK appends the user row to the session log when the driver picks
 * the prompt up — and the only other copy is the optimistic bubble inside the
 * chat store, which dies with the panel: `ChatView` keys `ChatPanel` on the
 * session id, so navigating to another conversation, a project page or settings
 * unmounts it. Coming back remounts and re-seeds from the route loader, whose
 * history predates the question, and the turn then streams an answer under a
 * conversation that never shows what was asked — while the composer sits in its
 * running state. Switching away again or refreshing "fixed" it only because by
 * then the row had landed.
 *
 * Keyed by session id and module-level, so it outlives that unmount. Readers
 * are expected to de-duplicate against what they already render (the question
 * is not removed the instant the server copy appears), which also makes the
 * cleanup below a memory concern rather than a correctness one.
 */
const questions = new Map<string, AgentMessage>();

/**
 * A brand-new chat has no session id until the backend assigns one on the
 * `session` frame, so its question parks here until `claimQuestion`.
 */
let unclaimed: AgentMessage | null = null;

/** Remember what was just sent, so a remount can render it before the server
 * history catches up. */
export function rememberQuestion(
  sessionId: string | null | undefined,
  message: AgentMessage
): void {
  if (sessionId) questions.set(sessionId, message);
  else unclaimed = message;
}

/** File the parked question under the id the backend just assigned. */
export function claimQuestion(sessionId: string): void {
  if (unclaimed) {
    questions.set(sessionId, unclaimed);
    unclaimed = null;
  }
}

export function pendingQuestion(
  sessionId: string | null | undefined
): AgentMessage | undefined {
  return sessionId ? questions.get(sessionId) : undefined;
}

/** The turn is over (or its row is on the server): the copy is redundant. */
export function forgetQuestion(sessionId: string | null | undefined): void {
  if (sessionId) questions.delete(sessionId);
}
