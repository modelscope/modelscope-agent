---
title: Create Explicit Component Variants
impact: MEDIUM
tags: [composition, variants, architecture]
---

# Create Explicit Component Variants

Instead of one component with many boolean props, create explicit variant components. Each variant composes the pieces it needs.

## Why

- Self-documenting code - the variant name tells you what it does
- No hidden conditionals or impossible state combinations
- Each variant is explicit about its rendering and behavior
- Easier to test, maintain, and reason about

## Bad: One Component, Many Modes

```tsx
// What does this actually render?
<Composer
  isThread
  isEditing={false}
  channelId="abc"
  showAttachments
  showFormatting={false}
/>

// Inside Composer: nested conditionals everywhere
function Composer({ isThread, isEditing, channelId, ... }) {
  return (
    <form>
      {isThread && !isEditing && <ThreadHeader channelId={channelId} />}
      {isEditing && <EditHeader />}
      {!isThread && !isEditing && <DefaultHeader />}
      {/* ... more conditionals */}
    </form>
  );
}
```

## Good: Explicit Variants

```tsx
// Immediately clear what each renders
<ThreadComposer channelId="abc" />
<EditMessageComposer messageId="xyz" />
<ForwardMessageComposer />

// Each implementation is unique, explicit, and self-contained
function ThreadComposer({ channelId }: { channelId: string }) {
  return (
    <ThreadProvider channelId={channelId}>
      <Composer.Frame>
        <Composer.Input />
        <AlsoSendToChannelField channelId={channelId} />
        <Composer.Footer>
          <Composer.Formatting />
          <Composer.Emojis />
          <Composer.Submit />
        </Composer.Footer>
      </Composer.Frame>
    </ThreadProvider>
  );
}

function EditMessageComposer({ messageId }: { messageId: string }) {
  return (
    <EditMessageProvider messageId={messageId}>
      <Composer.Frame>
        <Composer.Input />
        <Composer.Footer>
          <Composer.Formatting />
          <Composer.CancelEdit />
          <Composer.SaveEdit />
        </Composer.Footer>
      </Composer.Frame>
    </EditMessageProvider>
  );
}

function ForwardMessageComposer() {
  return (
    <Composer.Frame>
      <Composer.Input placeholder="Add a message, if you'd like." />
      <Composer.Footer>
        <Composer.Formatting />
        <Composer.Emojis />
      </Composer.Footer>
    </Composer.Frame>
  );
}
```

## Each Variant Is Explicit About

1. **What provider/state it uses** - ThreadProvider, EditMessageProvider, etc.
2. **What UI elements it includes** - which subcomponents are rendered
3. **What actions are available** - Submit vs SaveEdit vs Forward
4. **What props it needs** - channelId, messageId, etc.

## Sharing Logic Between Variants

Variants can share:

- **Compound components** - Composer.Input, Composer.Footer
- **Hooks** - useComposerState, useSubmit
- **Utilities** - formatting functions, validation

But NOT conditional rendering logic.

```tsx
// Shared compound components
function ThreadComposer({ channelId }) {
  return (
    <Composer.Frame>
      <Composer.Input /> {/* Shared */}
      <AlsoSendToChannel /> {/* Thread-specific */}
      <Composer.Footer>
        {" "}
        {/* Shared */}
        <Composer.Submit /> {/* Shared */}
      </Composer.Footer>
    </Composer.Frame>
  );
}

function DMComposer({ dmId }) {
  return (
    <Composer.Frame>
      <Composer.Input /> {/* Shared */}
      <AlsoSendToDM /> {/* DM-specific */}
      <Composer.Footer>
        {" "}
        {/* Shared */}
        <Composer.Submit /> {/* Shared */}
      </Composer.Footer>
    </Composer.Frame>
  );
}
```

## Rules

1. If a component has mode-switching logic, split into variants
2. Name variants descriptively: ThreadComposer, EditComposer, ForwardComposer
3. Each variant should be obvious about what it renders (no hidden behavior)
4. Share internals (compound components, hooks) but not conditional logic
5. No boolean prop combinations to reason about, no impossible states
