---
title: Place Error Boundaries at Feature Boundaries
impact: HIGH
tags: [errors, boundaries, resilience]
---

# Place Error Boundaries at Feature Boundaries

Add error boundaries around independent features, not just at the top and not around every leaf.

## Why

- A single top-level boundary brings down the whole app on any error
- Too many boundaries create partially broken UI and confusing states
- Feature-level boundaries isolate failures while keeping the rest usable

## Pattern

```tsx
// Good: boundaries at feature seams
<ErrorBoundary fallback={<SidebarError />}> 
  <Sidebar />
</ErrorBoundary>

<ErrorBoundary fallback={<FeedError />}> 
  <Feed />
</ErrorBoundary>

<ErrorBoundary fallback={<TrendsError />}>
  <Trends />
</ErrorBoundary>
```

## Heuristic

Ask: “If this component crashes, should its siblings also crash?”

- If yes, put the boundary higher
- If no, put the boundary at this feature boundary

## Rules

1. Avoid only a single top-level error boundary
2. Avoid wrapping every component with a boundary
3. Place boundaries around independent feature areas
4. Use feature-specific fallbacks to prevent broken UX
