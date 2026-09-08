---
title: Name useEffect Functions
impact: MEDIUM
tags: [hooks, useEffect, debugging, readability]
---

# Name useEffect Functions

Use named function declarations instead of arrow functions in `useEffect`. Also name cleanup functions.

## Why

1. **Stack traces**: Named functions appear in error stack traces, making debugging easier
2. **Self-documentation**: The function name explains what the effect does
3. **Single responsibility**: Naming encourages one concern per effect
4. **Code review**: Easier to understand effect purpose at a glance

## Bad: Anonymous Arrow Functions

```tsx
// Bad: anonymous functions hide intent
useEffect(() => {
  document.title = title;
}, [title]);

useEffect(() => {
  let handler = (e: KeyboardEvent) => {
    if (e.key === "Escape") onClose();
  };
  window.addEventListener("keydown", handler);
  return () => window.removeEventListener("keydown", handler);
}, [onClose]);
```

When these effects error, the stack trace shows `anonymous` or `<anonymous>`.

## Good: Named Function Declarations

```tsx
// Good: named functions are self-documenting
useEffect(
  function syncDocumentTitle() {
    document.title = title;
  },
  [title],
);

useEffect(
  function handleEscapeKey() {
    let handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);

    return function removeEscapeKeyHandler() {
      window.removeEventListener("keydown", handler);
    };
  },
  [onClose],
);
```

Stack traces now show `syncDocumentTitle` or `handleEscapeKey`.

## Pattern Examples

### Data Synchronization

```tsx
useEffect(
  function syncLocalStorage() {
    localStorage.setItem("preferences", JSON.stringify(preferences));
  },
  [preferences],
);
```

### Subscriptions

```tsx
useEffect(function subscribeToOnlineStatus() {
  function handleOnline() {
    setIsOnline(true);
  }
  function handleOffline() {
    setIsOnline(false);
  }

  window.addEventListener("online", handleOnline);
  window.addEventListener("offline", handleOffline);

  return function unsubscribeFromOnlineStatus() {
    window.removeEventListener("online", handleOnline);
    window.removeEventListener("offline", handleOffline);
  };
}, []);
```

### Form Reset (Remix pattern)

```tsx
useEffect(
  function resetFormOnSuccess() {
    if (fetcher.state === "idle" && fetcher.data?.ok) {
      formRef.current?.reset();
    }
  },
  [fetcher.state, fetcher.data],
);
```

### Third-Party Integration

```tsx
useEffect(function initializeMap() {
  if (!mapRef.current) return;

  let map = new MapLibrary(mapRef.current, { center });
  mapInstanceRef.current = map;

  return function destroyMap() {
    map.destroy();
  };
}, []);
```

### Analytics

```tsx
useEffect(
  function trackPageView() {
    analytics.page(pathname);
  },
  [pathname],
);
```

## Naming Conventions

| Effect Purpose | Name Pattern                                             |
| -------------- | -------------------------------------------------------- |
| Sync data      | `sync[What]` - `syncDocumentTitle`, `syncLocalStorage`   |
| Subscribe      | `subscribeTo[What]` - `subscribeToOnlineStatus`          |
| Initialize     | `initialize[What]` - `initializeMap`, `initializeChart`  |
| Handle event   | `handle[What]` - `handleEscapeKey`, `handleResize`       |
| Track/log      | `track[What]` - `trackPageView`, `logError`              |
| Reset          | `reset[What]` - `resetFormOnSuccess`                     |
| Cleanup        | `destroy[What]`, `remove[What]`, `unsubscribeFrom[What]` |

## Multiple Effects

Named functions make it clear why you have separate effects:

```tsx
function UserProfile({ userId }: Props) {
  useEffect(
    function fetchUserData() {
      // Fetch user when userId changes
    },
    [userId],
  );

  useEffect(
    function trackProfileView() {
      // Analytics - separate concern
      analytics.track("profile_viewed", { userId });
    },
    [userId],
  );

  useEffect(function setupKeyboardShortcuts() {
    // Keyboard handling - separate concern
  }, []);
}
```

## Rules

1. Always use named function declarations in `useEffect`, not arrow functions
2. Name cleanup functions too (`return function cleanup() { ... }`)
3. Use descriptive names that explain the effect's purpose
4. One concern per effect - if you can't name it clearly, split it
5. Follow naming conventions: `sync*`, `subscribeTo*`, `initialize*`, `handle*`, `track*`
