---
title: Limit useEffect Usage
impact: HIGH
impactDescription: prevents bugs and improves performance
tags: react, hooks, useEffect, patterns
---

## Limit useEffect Usage

Use `useEffect` only when absolutely necessary. Prefer derived state, event handlers, or other patterns.

### Why

1. **Effects are escape hatches** - For synchronizing with external systems, not for React logic
2. **Common source of bugs** - Missing dependencies, infinite loops, stale closures
3. **Performance overhead** - Runs after render, can cause extra re-renders
4. **Usually unnecessary** - Most "effects" are better expressed differently

### When NOT to Use useEffect

#### Deriving State from Props/State

```tsx
// Bad: useEffect to derive state
function FilteredList({ items, query }: Props) {
  let [filtered, setFiltered] = useState(items);

  useEffect(() => {
    setFiltered(items.filter((item) => item.name.includes(query)));
  }, [items, query]);

  return <List items={filtered} />;
}

// Good: derive during render
function FilteredList({ items, query }: Props) {
  let filtered = items.filter((item) => item.name.includes(query));
  return <List items={filtered} />;
}

// Good: useMemo if expensive
function FilteredList({ items, query }: Props) {
  let filtered = useMemo(
    () => items.filter((item) => item.name.includes(query)),
    [items, query],
  );
  return <List items={filtered} />;
}
```

#### Responding to Events

```tsx
// Bad: useEffect to handle form submission result
function Form() {
  let [submitted, setSubmitted] = useState(false);

  useEffect(() => {
    if (submitted) {
      showToast("Submitted!");
      navigate("/success");
    }
  }, [submitted]);

  return <form onSubmit={() => setSubmitted(true)}>...</form>;
}

// Good: handle in event handler
function Form() {
  function handleSubmit() {
    // Do it directly in the handler
    showToast("Submitted!");
    navigate("/success");
  }

  return <form onSubmit={handleSubmit}>...</form>;
}
```

#### Resetting State on Prop Change

```tsx
// Bad: useEffect to reset state
function UserProfile({ userId }: Props) {
  let [user, setUser] = useState(null);

  useEffect(() => {
    setUser(null); // Reset when userId changes
  }, [userId]);
}

// Good: use key to reset component
<UserProfile key={userId} userId={userId} />;
```

#### Transforming Data for Render

```tsx
// Bad: useEffect to transform
function Chart({ data }: Props) {
  let [chartData, setChartData] = useState([]);

  useEffect(() => {
    setChartData(data.map((d) => ({ x: d.date, y: d.value })));
  }, [data]);
}

// Good: transform during render
function Chart({ data }: Props) {
  let chartData = data.map((d) => ({ x: d.date, y: d.value }));
  return <LineChart data={chartData} />;
}
```

### When to Use useEffect

#### Synchronizing with External Systems

```tsx
// Good: subscribing to browser APIs
function useOnlineStatus() {
  let [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    function handleOnline() {
      setIsOnline(true);
    }
    function handleOffline() {
      setIsOnline(false);
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return () => {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  return isOnline;
}
```

#### Connecting to Third-Party Libraries

```tsx
// Good: integrating with non-React code
function Map({ center }: Props) {
  let mapRef = useRef<HTMLDivElement>(null);
  let mapInstance = useRef<MapLibrary | null>(null);

  useEffect(() => {
    if (!mapRef.current) return;

    mapInstance.current = new MapLibrary(mapRef.current, { center });

    return () => {
      mapInstance.current?.destroy();
    };
  }, []);

  // Update map when center changes
  useEffect(() => {
    mapInstance.current?.setCenter(center);
  }, [center]);

  return <div ref={mapRef} />;
}
```

#### Analytics/Logging (fire and forget)

```tsx
// Good: logging page views
useEffect(() => {
  analytics.logPageView(pathname);
}, [pathname]);
```

### Summary

| Scenario                     | Use Instead               |
| ---------------------------- | ------------------------- |
| Derive state from props      | Calculate during render   |
| Expensive calculation        | `useMemo`                 |
| Respond to user action       | Event handler             |
| Reset state on prop change   | `key` prop                |
| Transform data               | Calculate during render   |
| Subscribe to external system | `useEffect` (correct use) |
| Connect to third-party lib   | `useEffect` (correct use) |
