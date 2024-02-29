import queue
from typing import List, Union

from modelscope_agent.constants import DEFAULT_SEND_TO
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger


class Environment:
    turn: str = '0'
    raw_history: str = ''
    state: Union[str,
                 dict] = ''  # sort of transition state? shall we maintain it?
    messages_list_map: dict[str, list] = {}
    messages_queue_map: dict[str, object] = {}
    message_history: list = []
    roles: list = []

    def __init__(self, roles: List = [], **kwargs):
        self.remote = kwargs.get('remote', True)
        self.register_roles(roles)

    def register_roles(self, roles: List[str]):
        self.roles.extend(roles)
        for role in roles:
            if not isinstance(role, str):
                raise ValueError(
                    f'The type of role should be str, but get {type(role)}')
            if self.remote:
                from ray.util.queue import Queue
            else:
                from queue import Queue
            self.messages_queue_map[role] = Queue()
            self.messages_list_map[role] = []

    def get_message_list(self, role: str):
        return self.messages_list_map[role]

    def store_message_from_role(self, role: str, message: Message):
        """
        Store message from role to the environment
        Args:
            role: the role who send the messages
            message: the detail message information

        Returns:

        """
        self._check_role_in_env(role)
        self.raw_history += f'{role}: {message.content}/n'
        recipients = message.send_to
        if isinstance(recipients, str):
            recipients = [recipients]
        if DEFAULT_SEND_TO in recipients:
            recipients = self.roles

        # add the message to system
        self.message_history.append(message)
        for recipient in recipients:
            if role != recipient:
                logger.info(
                    f'{role} send message: {message.content} to {recipient}')
                message = Message(
                    content=message.content,
                    send_to=recipient,
                    sent_from=message.sent_from)
                self.messages_queue_map[recipient].put(message)
                self.messages_list_map[recipient].append(message)

    def extract_message_by_role(self, role: str):
        """
        extract all messages that left to the role from others
        Args:
            role: the role

        Returns:

        """
        self._check_role_in_env(role)
        messages_to_role = []
        while self.messages_queue_map[role]:
            if self.remote:
                item = self.messages_queue_map[role].get()
            else:
                try:
                    item = self.messages_queue_map[role].get_nowait()
                except queue.Empty:
                    break
            messages_to_role.append(item)
        logger.info(f'{role} extract data: {messages_to_role}')

        return messages_to_role

    def extract_all_history_message(self, limit: int = None):
        if limit and limit > 0:
            return self.message_history[-limit:]
        else:
            return self.message_history

    def get_notified_roles(self):
        notified_roles = []
        for role in self.messages_queue_map.keys():
            if hasattr(self.messages_queue_map[role], 'size'):
                if self.messages_queue_map[role].size() > 0:
                    notified_roles.append(role)
            else:
                if self.messages_queue_map[role].qsize() > 0:
                    notified_roles.append(role)
        return notified_roles

    def get_all_roles(self):
        return self.roles

    def _check_role_in_env(self, role: str):
        role_set = set(self.roles)
        if role not in role_set and role != 'human':
            raise ValueError(
                f'Role {role} is not in the environment scope, please register it to env'
            )
