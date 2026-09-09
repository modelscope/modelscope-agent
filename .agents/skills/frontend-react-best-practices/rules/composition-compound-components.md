---
title: Use Compound Components
impact: HIGH
tags: [composition, compound-components, architecture]
---

# Use Compound Components

Structure complex components as compound components with shared context. Each subcomponent accesses shared state via context, not props.

## Why

- Consumers compose exactly what they need
- No hidden conditionals or prop drilling
- Subcomponents can be rearranged freely
- State is shared without passing through every component

## Bad: Monolithic with Render Props

```tsx
function Composer({
  renderHeader,
  renderFooter,
  renderActions,
  showAttachments,
  showFormatting,
  showEmojis,
}: Props) {
  return (
    <form>
      {renderHeader?.()}
      <Input />
      {showAttachments && <Attachments />}
      {renderFooter ? (
        renderFooter()
      ) : (
        <Footer>
          {showFormatting && <Formatting />}
          {showEmojis && <Emojis />}
          {renderActions?.()}
        </Footer>
      )}
    </form>
  );
}
```

## Good: Compound Components

```tsx
// Define shared context
interface ComposerContextValue {
  state: ComposerState;
  actions: ComposerActions;
  meta: ComposerMeta;
}

const ComposerContext = createContext<ComposerContextValue | null>(null);

function useComposer() {
  let context = useContext(ComposerContext);
  if (!context) throw new Error("Must be used within Composer.Provider");
  return context;
}

// Provider component
function ComposerProvider({ children, state, actions, meta }: ProviderProps) {
  return (
    <ComposerContext.Provider value={{ state, actions, meta }}>
      {children}
    </ComposerContext.Provider>
  );
}

// Compound components access context
function ComposerFrame({ children }: { children: React.ReactNode }) {
  return <form className="composer">{children}</form>;
}

function ComposerInput() {
  const { state, actions, meta } = useComposer();
  return (
    <textarea
      ref={meta.inputRef}
      value={state.input}
      onChange={(e) => actions.update((s) => ({ ...s, input: e.target.value }))}
    />
  );
}

function ComposerSubmit() {
  const { actions } = useComposer();
  return <Button onPress={actions.submit}>Send</Button>;
}

// Export as namespace
const Composer = {
  Provider: ComposerProvider,
  Frame: ComposerFrame,
  Input: ComposerInput,
  Submit: ComposerSubmit,
  Header: ComposerHeader,
  Footer: ComposerFooter,
  Attachments: ComposerAttachments,
  Formatting: ComposerFormatting,
  Emojis: ComposerEmojis,
};

export { Composer };
```

## Usage

```tsx
// Consumers compose exactly what they need
<Composer.Provider state={state} actions={actions} meta={meta}>
  <Composer.Frame>
    <Composer.Header />
    <Composer.Input />
    <Composer.Footer>
      <Composer.Formatting />
      <Composer.Submit />
    </Composer.Footer>
  </Composer.Frame>
</Composer.Provider>
```

## Pattern: Components Outside the Frame

Components that need state don't have to be visually inside the frame:

```tsx
function ForwardMessageDialog() {
  return (
    <ForwardMessageProvider>
      <Dialog>
        {/* The composer UI */}
        <Composer.Frame>
          <Composer.Input placeholder="Add a message..." />
        </Composer.Frame>

        {/* Preview lives OUTSIDE Composer.Frame but can read state */}
        <MessagePreview />

        {/* Submit button OUTSIDE Composer.Frame but can submit */}
        <DialogActions>
          <CancelButton />
          <ForwardButton />
        </DialogActions>
      </Dialog>
    </ForwardMessageProvider>
  );
}

// These work because they're inside the Provider
function ForwardButton() {
  const { actions } = useComposer();
  return <Button onPress={actions.submit}>Forward</Button>;
}

function MessagePreview() {
  const { state } = useComposer();
  return <Preview message={state.input} />;
}
```

## Rules

1. Define a context for shared state and actions
2. Create small, focused subcomponents that consume context
3. Export as namespace object (Composer.Input, Composer.Submit)
4. Provider boundary determines access, not visual nesting
5. Prefer children over render props for composition
