import type { ReactNode } from 'react'

/** The `global` / `project` pill that dialogs put next to their title, so a
 * document or form that acts on a whole scope says which one before it is
 * submitted. Shared so the two MCP dialogs cannot drift apart visually. */
export function ScopeBadge({ children }: { children?: ReactNode }) {
  if (!children) return null
  return (
    <span className="rounded bg-msa-fill-purple px-1.5 py-0.5 text-[10px] text-msa-text-brand1">
      {children}
    </span>
  )
}
