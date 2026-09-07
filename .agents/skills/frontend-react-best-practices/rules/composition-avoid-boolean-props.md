---
title: Avoid Boolean Prop Proliferation
impact: HIGH
tags: [composition, props, architecture]
---

# Avoid Boolean Prop Proliferation

Don't add boolean props like `isThread`, `isEditing`, `isDMThread` to customize component behavior. Use composition instead.

## Why

- Each boolean doubles possible states (2^n complexity)
- Creates unmaintainable conditional logic inside components
- Hard to reason about all possible combinations
- Changes require modifying the monolithic component

## Bad: Boolean Props

```tsx
function Composer({
  onSubmit,
  isThread,
  channelId,
  isDMThread,
  dmId,
  isEditing,
  isForwarding,
}: Props) {
  return (
    <form>
      <Header />
      <Input />
      {isDMThread ? (
        <AlsoSendToDMField id={dmId} />
      ) : isThread ? (
        <AlsoSendToChannelField id={channelId} />
      ) : null}
      {isEditing ? (
        <EditActions />
      ) : isForwarding ? (
        <ForwardActions />
      ) : (
        <DefaultActions />
      )}
      <Footer onSubmit={onSubmit} />
    </form>
  );
}

// Usage: what does this actually render?
<Composer
  isThread
  isEditing={false}
  channelId="abc"
  showAttachments
  showFormatting={false}
/>;
```

## Good: Explicit Variants via Composition

```tsx
// Each variant is explicit about what it renders
function ChannelComposer() {
  return (
    <Composer.Frame>
      <Composer.Header />
      <Composer.Input />
      <Composer.Footer>
        <Composer.Attachments />
        <Composer.Formatting />
        <Composer.Emojis />
        <Composer.Submit />
      </Composer.Footer>
    </Composer.Frame>
  );
}

function ThreadComposer({ channelId }: { channelId: string }) {
  return (
    <Composer.Frame>
      <Composer.Header />
      <Composer.Input />
      <AlsoSendToChannelField id={channelId} />
      <Composer.Footer>
        <Composer.Formatting />
        <Composer.Emojis />
        <Composer.Submit />
      </Composer.Footer>
    </Composer.Frame>
  );
}

function EditComposer() {
  return (
    <Composer.Frame>
      <Composer.Input />
      <Composer.Footer>
        <Composer.Formatting />
        <Composer.CancelEdit />
        <Composer.SaveEdit />
      </Composer.Footer>
    </Composer.Frame>
  );
}

// Usage: immediately clear what this renders
<ThreadComposer channelId="abc" />
<EditComposer />
```

## When Boolean Props Are OK

Simple, non-combinatorial toggles:

```tsx
// OK: single boolean for a specific feature
<Button isDisabled>Submit</Button>
<Input isReadOnly />
<Modal isOpen={showModal} />

// NOT OK: multiple booleans that change structure
<Form isEditing isThread showAdvanced hideFooter />
```

## Rules

1. If you have 2+ boolean props that affect rendering, use composition
2. Create explicit variant components instead of prop combinations
3. Use compound components to share internals without sharing conditionals
4. Each variant should be self-documenting about what it renders
5. Single boolean props for simple toggles (disabled, loading) are fine
