---
title: Use useHydrated for SSR/CSR Divergence
impact: MEDIUM
tags: [rendering, hydration]
---

# Use useHydrated for SSR/CSR Divergence

Use `useHydrated` to render safe SSR fallbacks without mismatches.

## Pattern

```tsx
import { useHydrated } from "remix-utils/use-hydrated";

export function CopyButton({ text }: { text: string }) {
  let hydrated = useHydrated();
  return (
    <button
      type="button"
      disabled={!hydrated}
      onClick={() => navigator.clipboard.writeText(text)}
    >
      Copy
    </button>
  );
}
```

## Rules

1. Return the same fallback on SSR and first CSR render
2. Switch to client UI after hydration
