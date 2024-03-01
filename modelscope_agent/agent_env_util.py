import os
import time
from pathlib import Path
from typing import Callable, List, Union

import ray
from modelscope_agent.constants import DEFAULT_AGENT_ROOT, DEFAULT_SEND_TO
from modelscope_agent.environment import Environment
from modelscope_agent.memory import MemoryWithRetrievalKnowledge
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger
from ray._raylet import ObjectRefGenerator
from ray.util.client.common import ClientActorHandle, ClientObjectRef


class AgentEnvMixin:

    def __init__(self,
                 role: str = 'default_role',
                 env: Union[Environment, ClientActorHandle] = None,
                 storage_path: Union[str, Path] = DEFAULT_AGENT_ROOT,
                 is_watcher: bool = False,
                 use_history: bool = True,
                 parse_env_prompt_function: Callable = None,
                 remote=True,
                 **kwargs):
        """
        Agent environment context mixin class to allow the agent to communicate with other agent, in the
        form of multi-agent
        Args:
            role: the name of role
            env: the environment instance, where the message come from
            storage_path: the local history story path
            is_watcher: if the agent is a watcher, who view all information and leave no message
            use_history：some roles need history, while some not
            parse_env_prompt_function: The function convert the env message into current prompt,
            this function receive message and convert it into prompt
            **kwargs:
        """
        self._role = role
        self.env_context = env
        self.cur_step_env_prompt = ''
        self.is_watcher = is_watcher
        self.use_history = use_history
        self.remote = remote
        if not parse_env_prompt_function:
            self.parse_env_prompt_function = self.convert_to_string
        else:
            self.parse_env_prompt_function = parse_env_prompt_function
        assert isinstance(self.parse_env_prompt_function, Callable)

        knowledge_path = os.path.join(storage_path, role, 'knowledge')
        memory_path = os.path.join(storage_path, role, 'memory')

        self.memory = MemoryWithRetrievalKnowledge(
            storage_path=knowledge_path,
            name=role + '_memory',
            memory_path=memory_path,
            use_cache=False,
        )

    def set_env_context(self, env_context):
        if env_context:
            self.env_context = env_context

    def set_remote(self, remote):
        self.remote = remote

    def role(self):
        """Get the name of the agent."""
        return self._role

    def step(self,
             messages: Union[str, dict, ObjectRefGenerator] = None,
             send_to: Union[str, list] = DEFAULT_SEND_TO,
             **kwargs):
        """
        step function for agent to interact with env and other agents
        Args:
            messages:
            send_to: the message that allows to send to other agents
            kwargs: additional keywords, such as runtime llm setting

        Returns: ObjectRefGenerator that could be used to get result from other agent or env

        """
        # check if env is ready
        if not self._check_env_ready():
            raise ValueError(
                'Environment context is not set, please set environment first')

        if isinstance(send_to, str):
            send_to = [send_to]

        # get message from other agent or env by generator
        prompt = ''
        if isinstance(messages, ObjectRefGenerator):
            remote_input = {}
            try:
                ref = next(messages)
                input_frame = AgentEnvMixin.extract_frame(ray.get(ref))
                if input_frame['agent'] not in remote_input:
                    remote_input[input_frame['agent']] = input_frame['content']
                else:
                    remote_input[
                        input_frame['agent']] += input_frame['content']
            except StopIteration:
                pass
            prompt = remote_input[input_frame['agent']]
        elif isinstance(messages, dict):
            prompt = messages['content']
        elif messages is None:
            prompt = ''
        else:
            prompt = messages

        cur_step_env_info = self.pull()
        prompt += cur_step_env_info

        # run agent core loop to get action or reslt
        result = ''
        logger.info(f'{self._role} cur prompt is: {prompt}')

        # get history
        history = []
        if self.use_history:
            history = self.memory.get_history()

        # run generation
        for frame in self.run(
                prompt,
                history=history,
                **kwargs,
        ):
            cur_frame = frame
            result += cur_frame
            yield AgentEnvMixin.frame_wrapper(self._role, cur_frame)

        # update memory
        if self.use_history:
            self.memory.update_history([
                Message(
                    role='user',
                    content=prompt,
                    send_to=send_to,
                    sent_from=self._role,
                ),
                Message(
                    role='assistant',
                    content=result,
                    send_to=send_to,
                    sent_from=self._role,
                )
            ])

        # publish result to env if not only observe
        if not self.is_watcher:
            self.publish(result, send_to)

    @staticmethod
    def frame_wrapper(agent_name, frame: str) -> str:
        """
        wrap frame with agent name
        Args:
            agent_name: current agent name
            frame: content

        Returns: <agent1>: content

        """
        return f'<{agent_name}>: {frame}'

    @staticmethod
    def extract_frame(frame: str) -> dict:
        """
        extract frame from agent name and frame format
        Args:
            frame:  <agent1>: raw content

        Returns: {'agent': 'agent1', 'content': 'raw content'}

        """
        agent, content = frame.split(': ', 1)
        agent = agent.strip('<>')
        result_dict = {'agent': agent, 'content': content}
        return result_dict

    def publish(self, result, send_to: list = []):
        # parse current state and message from
        # state, message, send_to_by_model = self._parse_message_attribute_from_llm(llm_result)

        # if no specific send to then, send to all
        # todo: should add parse from llm to decide send to which role
        agents_to_send = send_to

        message = Message(
            content=result, send_to=agents_to_send, sent_from=self._role)

        logger.info(
            f'Ready for send message from: {self._role}, to {agents_to_send}')
        if self.remote:
            self.env_context.store_message_from_role.remote(
                self._role, message)
        else:
            self.env_context.store_message_from_role(self._role, message)

    def pull(self):
        """
        extract message from environment by role name
        Returns: prompt

        """
        if not self.is_watcher:
            if self.remote:
                received_messages = self.env_context.extract_message_by_role.remote(
                    self._role)
                received_messages = ray.get(received_messages)
            else:
                received_messages = self.env_context.extract_message_by_role(
                    self._role)
            if received_messages and len(received_messages) > 0:
                cur_step_env_prompt = self.parse_env_prompt_function(
                    received_messages)
                return cur_step_env_prompt
            else:
                return ''
        else:
            if self.remote:
                received_messages = self.env_context.extract_all_history_message.remote(
                )
                received_messages = ray.get(received_messages)
            else:
                received_messages = self.env_context.extract_all_history_message(
                )
            if received_messages and len(received_messages) > 0:
                conversation_history = ''
                for item in received_messages:
                    conversation_history += f'{item.sent_from}\n{item.content}\n'
                return conversation_history
            else:
                return ''

    def convert_to_string(self, messages: List[Message]):
        prompt_template = """
        . In last round, you get the following information from environment:
        {conversation_history}
        """
        conversation_history = ''
        for item in messages:
            conversation_history += f'{item.sent_from}: {item.content}\n'
        return prompt_template.format(
            conversation_history=conversation_history)

    def _check_env_ready(self):
        if self.env_context:
            return True
        else:
            return False
