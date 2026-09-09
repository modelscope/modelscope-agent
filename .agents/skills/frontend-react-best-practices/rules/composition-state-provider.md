---
title: Lift State into Provider Components
impact: HIGH
tags: [composition, state, provider, architecture]
---

# Lift State into Provider Components

Move state management into dedicated provider components. This allows sibling components outside the main UI to access and modify state.

## Why

- Components outside the main UI can access state
- State implementation is decoupled from UI
- Same UI works with different state sources
- No prop drilling or awkward refs

## Bad: State Trapped Inside Component

```tsx
function ForwardMessageComposer() {
  let [state, setState] = useState(initialState);
  let forwardMessage = useForwardMessage();

  return (
    <Composer.Frame>
      <Composer.Input />
      <Composer.Footer />
    </Composer.Frame>
  );
}

// Problem: How does ForwardButton access composer state?
function ForwardMessageDialog() {
  return (
    <Dialog>
      <ForwardMessageComposer />
      <MessagePreview /> {/* Needs composer state - can't access it */}
      <DialogActions>
        <CancelButton />
        <ForwardButton /> {/* Needs to call submit - can't access it */}
      </DialogActions>
    </Dialog>
  );
}
```

## Bad: useEffect to Sync State Up

```tsx
function ForwardMessageDialog() {
  const [input, setInput] = useState("");
  return (
    <Dialog>
      <ForwardMessageComposer onInputChange={setInput} />
      <MessagePreview input={input} />
    </Dialog>
  );
}

function ForwardMessageComposer({ onInputChange }) {
  const [state, setState] = useState(initialState);

  // Syncing state on every change is messy
  useEffect(() => {
    onInputChange(state.input);
  }, [state.input, onInputChange]);
}
```

## Good: State Lifted to Provider

```tsx
function ForwardMessageProvider({ children }: { children: React.ReactNode }) {
  let [state, setState] = useState(initialState);
  let forwardMessage = useForwardMessage();
  let inputRef = useRef(null);

  return (
    <Composer.Provider
      state={state}
      actions={{ update: setState, submit: forwardMessage }}
      meta={{ inputRef }}
    >
      {children}
    </Composer.Provider>
  );
}

function ForwardMessageDialog() {
  return (
    <ForwardMessageProvider>
      <Dialog>
        <ForwardMessageComposer />
        <MessagePreview /> {/* Can access state via context */}
        <DialogActions>
          <CancelButton />
          <ForwardButton /> {/* Can access submit via context */}
        </DialogActions>
      </Dialog>
    </ForwardMessageProvider>
  );
}

function ForwardButton() {
  const { actions } = useComposer();
  return <Button onPress={actions.submit}>Forward</Button>;
}

function MessagePreview() {
  const { state } = useComposer();
  return <Preview message={state.input} attachments={state.attachments} />;
}
```

## Different Providers, Same UI

The same UI components work with different state implementations:

```tsx
// Local state for ephemeral forms
function ForwardMessageProvider({ children }) {
  let [state, setState] = useState(initialState);
  let forwardMessage = useForwardMessage();

  return (
    <Composer.Provider
      state={state}
      actions={{ update: setState, submit: forwardMessage }}
    >
      {children}
    </Composer.Provider>
  );
}

// Global synced state for channels
function ChannelProvider({ channelId, children }) {
  const { state, update, submit } = useGlobalChannel(channelId);

  return (
    <Composer.Provider state={state} actions={{ update, submit }}>
      {children}
    </Composer.Provider>
  );
}

// Same Composer.Input works with both!
<ForwardMessageProvider>
  <Composer.Input /> {/* Uses local state */}
</ForwardMessageProvider>

<ChannelProvider channelId="abc">
  <Composer.Input /> {/* Uses global synced state */}
</ChannelProvider>
```

## Define Generic Context Interface

```tsx
interface ComposerState {
  input: string;
  attachments: Attachment[];
  isSubmitting: boolean;
}

interface ComposerActions {
  update: (updater: (state: ComposerState) => ComposerState) => void;
  submit: () => void;
}

interface ComposerMeta {
  inputRef: React.RefObject<HTMLTextAreaElement>;
}

interface ComposerContextValue {
  state: ComposerState;
  actions: ComposerActions;
  meta: ComposerMeta;
}
```

Any provider that implements this interface works with the UI components.

## Rules

1. State management lives in provider components, not UI components
2. UI components only know about the context interface
3. Different providers can implement the same interface differently
4. Provider boundary is what matters, not visual nesting
5. Components outside the "main" UI can still access state if inside provider
