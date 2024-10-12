import os
from functools import wraps
from pathlib import Path
from typing import Dict, Iterable, List, Union

import json
from modelscope_agent.schemas import AgentAttr, Message
from modelscope_agent.utils.tokenization_utils import count_tokens
from pydantic import ConfigDict


def enable_rag_callback(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        callbacks = self.callback_manager
        if callbacks.callbacks:
            callbacks.on_rag_start(*args, **kwargs)
        response = func(self, *args, **kwargs)
        if callbacks:
            callbacks.on_rag_end('retrieval', response)
        return response

    return wrapper


class Memory(AgentAttr):
    path: Union[str, Path]
    model_config = ConfigDict(extra='allow')

    def save_history(self):
        """
        save history memory to path
        Args:
            history: List of Message

        Returns: None

        """
        if self.history is None or len(self.history) == 0:
            return

        directory = os.path.dirname(self.path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(self.path, 'w', encoding='utf-8') as file:
            # 使用 Pydantic 的 dict() 方法将模型列表转换为字典列表
            messages_dict_list = [
                message.model_dump() for message in self.history
            ]
            # 使用 json.dump 将字典列表写入文件
            json.dump(messages_dict_list, file, ensure_ascii=False, indent=2)

    def load_history(self) -> List[Message]:
        """
        Load memory from path
        Returns: list of Message

        """
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                # 使用 json.load 读取文件中的字典列表
                messages_dict_list = json.load(file)
                # 使用 Pydantic 的 parse_obj_list 方法将字典列表转换为模型列表
                messages_list = [
                    Message.model_validate(message_dict)
                    for message_dict in messages_dict_list
                ]
                self.history = messages_list
                return messages_list
        except FileNotFoundError:
            print('File not found.')
            return []

    def get_history(self) -> List[Message]:
        return [message.model_dump() for message in self.history]

    def update_history(self, message: Union[Message, Iterable[Message]]):
        if isinstance(message, list):
            self.history.extend(message)
        else:
            self.history.append(message)

    def get_history_token_count(self) -> int:
        history_token_count = sum(
            count_tokens(message.content) for message in self.history)
        return history_token_count

    def pop_history(self):
        return self.history.pop()

    def clear_history(self):
        self.history = []
