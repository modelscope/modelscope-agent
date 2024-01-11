import os
from abc import abstractmethod
from typing import Iterable, List, Union

import json
from modelscope_agent.schemas import AgentHolder, Message
from pydantic import ConfigDict


class BaseMemory(AgentHolder):

    model_config = ConfigDict(extra='allow')

    @abstractmethod
    def dump(self, data):
        pass

    @abstractmethod
    def read(self):
        pass

    def serialize(self, data: dict):
        self.dump(data)

    def deserialize(self):
        data = self.read()
        if data:
            print('Data deserialized.')
            return data
        else:
            print('Unable to deserialize data.')
            return None

    def get_history(self) -> List[Message]:
        return [message.model_dump() for message in self.history]

    def update_history(self, message: Union[Message, Iterable[Message]]):
        if isinstance(message, list):
            self.history.extend(message)
        else:
            self.history.append(message)

    def pop_history(self):
        return self.history.pop()

    def clear_history(self):
        self.history = []


class FileStorageMemory(BaseMemory):
    path: str

    def dump(self, data):
        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(self.path, 'w') as f:
            json.dump(data, f)

    def read(self):
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print('File not found.')
            return None
