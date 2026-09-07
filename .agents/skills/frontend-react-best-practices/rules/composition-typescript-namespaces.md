---
title: Use TypeScript Namespaces for Component Types
impact: LOW
tags: [typescript, components, patterns]
---

# Use TypeScript Namespaces for Component Types

Combine a component and its related types using TypeScript namespaces for cleaner imports.

## Why

- Single import gives you component + all its types
- Avoids naming conflicts (`ButtonProps` vs `Button.Props`)
- Groups related types together (`Button.Props`, `Button.Variant`, `Button.Size`)
- Cleaner API for consumers of the component

## Important: Types Only

**Namespaces should only contain type definitions, never runtime code.**

```tsx
// Good: namespace contains only types
export namespace Button {
  export type Props = { ... };
  export type Variant = "solid" | "ghost";
}

// Bad: namespace contains runtime code
export namespace Button {
  export const defaultVariant = "solid"; // Don't do this
  export function getClassName() { ... } // Don't do this
}
```

## Pattern

```tsx
// components/button.tsx

export namespace Button {
  export type Variant = "solid" | "ghost" | "outline";
  export type Size = "sm" | "md" | "lg";

  export interface Props {
    variant?: Variant;
    size?: Size;
    children: React.ReactNode;
    onClick?: () => void;
    disabled?: boolean;
  }
}

export function Button({
  variant = "solid",
  size = "md",
  children,
  onClick,
  disabled,
}: Button.Props) {
  return (
    <button
      className={getButtonClasses(variant, size)}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
```

## Usage

Consumers import once and get everything:

```tsx
import { Button } from "~/components/button";

// Use the component
<Button variant="ghost" size="lg">
  Click me
</Button>;

// Use the types
function CustomButton(props: Button.Props) {
  return <Button {...props} />;
}

// Use specific type
function getVariantColor(variant: Button.Variant): string {
  switch (variant) {
    case "solid":
      return "blue";
    case "ghost":
      return "transparent";
    case "outline":
      return "white";
  }
}
```

## Without Namespaces (Comparison)

```tsx
// Without namespace - multiple exports, potential naming conflicts
import {
  Button,
  ButtonProps,
  ButtonVariant,
  ButtonSize,
} from "~/components/button";

// Or with renaming
import { Button, type Props as ButtonProps } from "~/components/button";
```

## Extending Types

When creating a component that extends another:

```tsx
// components/icon-button.tsx
import { Button } from "./button";

export namespace IconButton {
  export interface Props extends Omit<Button.Props, "children"> {
    icon: React.ReactNode;
    label: string; // For accessibility
  }
}

export function IconButton({ icon, label, ...buttonProps }: IconButton.Props) {
  return (
    <Button {...buttonProps} aria-label={label}>
      {icon}
    </Button>
  );
}
```

## Complex Component Example

```tsx
// components/select.tsx

export namespace Select {
  export interface Option<T = string> {
    value: T;
    label: string;
    disabled?: boolean;
  }

  export interface Props<T = string> {
    options: Option<T>[];
    value: T;
    onChange: (value: T) => void;
    placeholder?: string;
    disabled?: boolean;
  }

  export type Size = "sm" | "md" | "lg";
}

export function Select<T extends string>({
  options,
  value,
  onChange,
  placeholder,
  disabled,
}: Select.Props<T>) {
  // Implementation
}
```

Usage:

```tsx
import { Select } from "~/components/select";

type Status = "active" | "inactive" | "pending";

let options: Select.Option<Status>[] = [
  { value: "active", label: "Active" },
  { value: "inactive", label: "Inactive" },
  { value: "pending", label: "Pending" },
];

<Select<Status> options={options} value={status} onChange={setStatus} />;
```

## When to Use

| Scenario                              | Recommendation                  |
| ------------------------------------- | ------------------------------- |
| Simple component with Props only      | Optional - either pattern works |
| Component with multiple related types | Use namespace                   |
| Type might conflict with other types  | Use namespace                   |
| Building a component library          | Use namespace                   |
| Internal utility component            | Optional                        |

## Rules

1. **Types only** - Never put runtime code (values, functions) in namespaces
2. Export both namespace and function with the same name
3. Use `Button.Props` instead of `ButtonProps` naming convention
4. Group all component-related types in the namespace
5. Prefer `interface` for Props, `type` for unions/aliases
