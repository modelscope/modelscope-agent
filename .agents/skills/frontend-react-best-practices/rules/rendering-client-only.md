---
title: Use ClientOnly for Browser-Only UI
impact: MEDIUM
tags: [rendering, hydration]
---

# Use ClientOnly for Browser-Only UI

Render browser-only components on the client with a stable SSR fallback.

## Pattern

```tsx
import { ClientOnly } from "remix-utils/client-only";

export function MapSection() {
  return (
    <ClientOnly fallback={<StaticMapPreview />}>
      {() => <InteractiveMap />}
    </ClientOnly>
  );
}
```

## Rules

1. Always provide a fallback to avoid layout shift
2. Use ClientOnly for DOM APIs or browser-only libraries
