---
title: Avoid Over-Abstracting Component APIs
impact: HIGH
tags: [composition, api, components]
---

# Avoid Over-Abstracting Component APIs

Prefer composable, element-like APIs over rigid configuration objects.

## Why

- Large abstractions require more configuration surface
- Options objects hide native semantics and limit flexibility
- Children-based APIs are easier to compose and extend

## Pattern

```tsx
// Bad: data-driven options hide native capabilities
type SelectProps = {
  options: { label: string; value: string }[];
  value: string;
  onChange: (value: string) => void;
};

function Select({ options, value, onChange }: SelectProps) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}>
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// Good: composable API, close to native HTML
type SelectProps = React.ComponentPropsWithoutRef<"select">;
type OptionProps = React.ComponentPropsWithoutRef<"option">;

function Select({ children, ...props }: SelectProps) {
  return <select {...props}>{children}</select>;
}

function Option({ children, ...props }: OptionProps) {
  return <option {...props}>{children}</option>;
}

<Select value="abc" onChange={...}>
  <Option value="abc">ABC</Option>
  <Option value="xyz">XYZ</Option>
</Select>
```

## When You Need Shared Behavior

Use context for shared config instead of adding more props to every child:

```tsx
const SelectContext = React.createContext({ variant: "default" });

function Select({ variant = "default", children, ...props }: SelectProps & { variant?: string }) {
  return (
    <select {...props}>
      <SelectContext.Provider value={{ variant }}>
        {children}
      </SelectContext.Provider>
    </select>
  );
}

function Option({ children, ...props }: OptionProps) {
  let { variant } = React.useContext(SelectContext);
  return (
    <option {...props} data-variant={variant}>
      {children}
    </option>
  );
}
```

## Rules

1. Keep component abstractions small and focused
2. Prefer children-based APIs over configuration objects
3. Preserve native HTML behavior and props when possible
4. Use context for shared configuration instead of prop drilling
