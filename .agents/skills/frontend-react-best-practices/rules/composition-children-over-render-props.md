---
title: Prefer Children Over Render Props
impact: MEDIUM
tags: [composition, render-props, children]
---

# Prefer Children Over Render Props

Use `children` for composition instead of `renderX` props. Children are more readable, compose naturally, and don't require understanding callback signatures.

## Why

- More readable and familiar JSX structure
- No callback signatures to understand
- Better composability with other patterns
- Cleaner API surface

## Bad: Render Props

```tsx
function Composer({
  renderHeader,
  renderFooter,
  renderActions,
}: {
  renderHeader?: () => React.ReactNode;
  renderFooter?: () => React.ReactNode;
  renderActions?: () => React.ReactNode;
}) {
  return (
    <form>
      {renderHeader?.()}
      <Input />
      {renderFooter ? renderFooter() : <DefaultFooter />}
      {renderActions?.()}
    </form>
  );
}

// Usage is awkward
<Composer
  renderHeader={() => <CustomHeader />}
  renderFooter={() => (
    <>
      <Formatting />
      <Emojis />
    </>
  )}
  renderActions={() => <SubmitButton />}
/>;
```

## Good: Compound Components with Children

```tsx
function ComposerFrame({ children }: { children: React.ReactNode }) {
  return <form className="composer">{children}</form>;
}

function ComposerFooter({ children }: { children: React.ReactNode }) {
  return <footer className="composer-footer">{children}</footer>;
}

// Usage is natural JSX
<Composer.Frame>
  <CustomHeader />
  <Composer.Input />
  <Composer.Footer>
    <Composer.Formatting />
    <Composer.Emojis />
    <SubmitButton />
  </Composer.Footer>
</Composer.Frame>;
```

## When Render Props Are Appropriate

Render props work well when the parent needs to **provide data** to the child:

```tsx
// GOOD: Render props when passing data back
<List
  data={items}
  renderItem={({ item, index }) => (
    <Item item={item} index={index} />
  )}
/>

// GOOD: Render props for slot patterns with context
<Combobox>
  {({ open, selected }) => (
    <>
      <Combobox.Button>{selected?.name}</Combobox.Button>
      {open && <Combobox.Options />}
    </>
  )}
</Combobox>
```

## Decision Guide

| Use Case                            | Pattern                 |
| ----------------------------------- | ----------------------- |
| Static structure composition        | `children`              |
| Need to pass data back to consumer  | Render props            |
| Multiple named slots                | Compound components     |
| Conditional children based on state | Render props with state |

## Examples

### Static Structure → Children

```tsx
// Children for layout
<Card>
  <Card.Header>Title</Card.Header>
  <Card.Body>Content</Card.Body>
  <Card.Footer>Actions</Card.Footer>
</Card>
```

### Data to Consumer → Render Props

```tsx
// Render props when providing data
<Autocomplete
  options={options}
  renderOption={(option, { selected, active }) => (
    <div className={cn(active && "bg-blue-100")}>
      {selected && <Check />}
      {option.label}
    </div>
  )}
/>
```

### Named Slots → Compound Components

```tsx
// Compound components for named slots
<Dialog>
  <Dialog.Title>Confirm</Dialog.Title>
  <Dialog.Description>Are you sure?</Dialog.Description>
  <Dialog.Actions>
    <Button>Cancel</Button>
    <Button>Confirm</Button>
  </Dialog.Actions>
</Dialog>
```

## Rules

1. Default to `children` for composition
2. Use compound components for multiple named slots
3. Use render props only when passing data back to consumer
4. Render props with no arguments should be `children` instead
5. Keep API surface minimal - don't add renderX props "just in case"
