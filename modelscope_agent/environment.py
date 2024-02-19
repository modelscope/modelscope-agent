import logging
import time
from typing import List, Union

import ray
from modelscope_agent.schemas import Message
from ray.util.client.common import ClientActorHandle, ClientObjectRef
from ray.util.queue import Queue


class Environment:
    turn: str = '0'
    raw_history: str = ''
    message_queue_persist: dict[str, Queue] = {}
    messages_queue_map: dict[str, Queue] = {}
    state: Union[str,
                 dict] = ''  # sort of transition state? shall we maintain it?
    messages_list_map: dict[str, list] = {}
    roles: list = []

    def __init__(self, roles: List = [], **kwargs):
        self.registry_role(roles)

    def registry_role(self, roles: List[str]):
        self.roles.extend(roles)
        for role in roles:
            if not isinstance(role, str):
                raise ValueError(
                    f'The type of role should be str, but get {type(role)}')
            self.messages_queue_map[role] = Queue()
            self.message_queue_persist[role] = Queue()
            self.messages_list_map[role] = []

    def get_message_queue_persist(self, role: str):
        return self.message_queue_persist[role]

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
        self.raw_history += f'state at {self.state}, {role}: {message.content}/n'
        recipiants = message.send_to
        if 'all' in recipiants:
            recipiants = self.roles
        for recipiant in recipiants:
            if role != recipiant:
                logging.warning(
                    msg=f'time:{time.time()} {role} put data: {message}')

                self.messages_queue_map[recipiant].put(
                    Message(
                        content=message.content,
                        sent_to=recipiant,
                        sent_from=message.sent_from))
                self.message_queue_persist[recipiant].put(
                    Message(
                        content=message.content,
                        sent_to=recipiant,
                        sent_from=message.sent_from))
                self.messages_list_map[recipiant].append(
                    Message(
                        content=message.content,
                        sent_to=recipiant,
                        sent_from=message.sent_from))

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
            messages_to_role.append(self.messages_queue_map[role].get())
        logging.warning(
            msg=f'time:{time.time()} {role} extract data: {messages_to_role}')
        logging.warning(
            msg=
            f'time:{time.time()} {role} get data: {self.messages_list_map[role]}'
        )

        return messages_to_role

    def get_notified_roles(self):
        notified_roles = []
        for role in self.messages_queue_map.keys():
            if self.messages_queue_map[role].size() > 0:
                notified_roles.append(role)
        return notified_roles

    def _check_role_in_env(self, role: str):
        role_set = set(self.roles)
        if role not in role_set and role != 'human':
            raise ValueError(
                f'Role {role} is not in the environment scope, please register it to env'
            )

    #
    # @staticmethod
    # def create_remote(cls,
    #                   roles=[],
    #                   state='',
    #                   max_concurrency=10,
    #                   *args,
    #                   **kwargs) -> ClientActorHandle:
    #     return ray.remote(
    #         name='env', max_concurrency=max_concurrency)(cls).remote(
    #             roles=roles, state=state, *args, **kwargs)
    #
    # @staticmethod
    # def create_local(cls, roles=[], state='', *args, **kwargs):
    #     return cls(roles=roles, state=state, *args, **kwargs)
