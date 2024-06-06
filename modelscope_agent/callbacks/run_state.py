import time
from typing import Dict, List, Union

from pydantic import BaseModel, Field

from .base import BaseCallback


class RunState(BaseModel):
    type: str
    name: str
    content: Union[str, Dict]
    create_time: int = Field(default_factory=lambda: int(time.time()))


class RunStateList(BaseModel):
    run_states: List[RunState]


class RunStateCallback(BaseCallback):

    def __init__(self):
        super().__init__()

        self._run_states = {}
        self.step = 0

    def on_step_start(self, *args, **kwargs):
        self.step += 1
        self._run_states[self.step] = []

    def on_llm_end(self, name, messages, **kwargs):
        stream = kwargs.get('stream', True)
        if stream:
            response = self._run_states[self.step][-1].content
        else:
            response = messages
        self._run_states[self.step].append(
            RunState(name=name, type='llm', content=response))

    def on_llm_new_token(self, name, chunk, **kwargs):
        if len(self._run_states[self.step]) > 0 and self._run_states[
                self.step][-1].type == 'llm_chunk':
            last_chunk = self._run_states[self.step][-1].content
        else:
            last_chunk = ''
        chunk = last_chunk + chunk
        self._run_states[self.step].append(
            RunState(name=name, type='llm_chunk', content=chunk))

    def on_tool_start(self, func_name, params={}):
        self._run_states[self.step].append(
            RunState(name=func_name, type='tool_input', content=params))

    def on_tool_end(self, func_name, exec_result):
        self._run_states[self.step].append(
            RunState(name=func_name, type='tool_output', content=exec_result))

    @property
    def run_states(self):
        return self._run_states