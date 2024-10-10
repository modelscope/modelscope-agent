from typing import List, Optional


class BaseCallback:

    def __init__(self) -> None:
        pass

    def on_llm_start(self, *args, **kwargs):
        pass

    def on_llm_end(self, *args, **kwargs):
        pass

    def on_llm_new_token(self, *args, **kwargs):
        pass

    def on_rag_start(self, *args, **kwargs):
        pass

    def on_rag_end(self, *args, **kwargs):
        pass

    def on_tool_start(self, *args, **kwargs):
        pass

    def on_tool_end(self, *args, **kwargs):
        pass

    def on_run_start(self, *args, **kwargs):
        pass

    def on_run_end(self, *args, **kwargs):
        pass

    def on_step_start(self, *args, **kwargs):
        pass

    def on_step_end(self, *args, **kwargs):
        pass


class CallbackManager(BaseCallback):

    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self.callbacks = callbacks

    def call_event(self, event, *args, **kwargs):
        if not self.callbacks:
            return
        for callback in self.callbacks:
            func = getattr(callback, event)
            func(*args, **kwargs)

    def on_llm_start(self, *args, **kwargs):
        self.call_event('on_llm_start', *args, **kwargs)

    def on_llm_end(self, *args, **kwargs):
        self.call_event('on_llm_end', *args, **kwargs)

    def on_llm_new_token(self, *args, **kwargs):
        self.call_event('on_llm_new_token', *args, **kwargs)

    def on_rag_start(self, *args, **kwargs):
        self.call_event('on_rag_start', *args, **kwargs)

    def on_rag_end(self, *args, **kwargs):
        self.call_event('on_rag_end', *args, **kwargs)

    def on_tool_start(self, *args, **kwargs):
        self.call_event('on_tool_start', *args, **kwargs)

    def on_tool_end(self, *args, **kwargs):
        self.call_event('on_tool_end', *args, **kwargs)

    def on_run_start(self, *args, **kwargs):
        self.call_event('on_run_start', *args, **kwargs)

    def on_run_end(self, *args, **kwargs):
        self.call_event('on_run_end', *args, **kwargs)

    def on_step_start(self, *args, **kwargs):
        self.call_event('on_step_start', *args, **kwargs)

    def on_step_end(self, *args, **kwargs):
        self.call_event('on_step_end', *args, **kwargs)
