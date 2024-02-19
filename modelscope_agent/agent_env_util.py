import logging
import os
import time
from pathlib import Path
from typing import List, Union

import ray
from modelscope_agent.agent import Agent
from modelscope_agent.constants import DEFAULT_AGENT_ROOT, DEFAULT_SEND_TO
from modelscope_agent.environment import Environment
from modelscope_agent.memory import MemoryWithRetrievalKnowledge
from modelscope_agent.schemas import Message
from ray._raylet import ObjectRefGenerator
from ray.util.client.common import ClientActorHandle, ClientObjectRef


class AgentEnvMixin:

    def __init__(self,
                 role: str,
                 env: Union[Environment, ClientActorHandle] = None,
                 storage_path: Union[str, Path] = DEFAULT_AGENT_ROOT,
                 **kwargs):
        self._role = role
        self.env_context = env
        self.cur_step_env_prompt = ''
        knowledge_path = os.path.join(storage_path, role, 'knowledge')
        memory_path = os.path.join(storage_path, role, 'memory')

        self.memory = MemoryWithRetrievalKnowledge(
            storage_path=knowledge_path,
            name=role + '_memory',
            memory_path=memory_path,
            use_cache=False,
        )

    @staticmethod
    def create_remote(cls, role: str, function_list: list, llm, env, *args,
                      **kwargs) -> ClientActorHandle:
        max_concurrency = kwargs.get('max_concurrency', 1)
        return ray.remote(
            name=role, max_concurrency=max_concurrency)(cls).remote(
                role=role,
                function_list=function_list,
                llm=llm,
                env=env,
                *args,
                **kwargs)

    @staticmethod
    def create_local(cls, role: str, function_list: list, llm, env, *args,
                     **kwargs) -> Agent:
        return cls(
            role=role,
            function_list=function_list,
            llm=llm,
            env=env,
            *args,
            **kwargs)

    def set_env_context(self, env_context):
        if env_context:
            self.env_context = env_context

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
            send_to: the message allow to send to other agent
            kwargs: other keywords

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
        logging.warning(
            msg=f'time:{time.time()} {self._role} cur prompt is: {prompt}')
        logging.warning(
            msg=
            f'time:{time.time()} {self._role} cur history is: {self.memory.get_history()}'
        )

        for frame in self.run(
                prompt,
                history=self.memory.get_history(),
        ):
            cur_frame = frame
            result += cur_frame
            yield AgentEnvMixin.frame_wrapper(self._role, cur_frame)

        # update memory
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

        # publish result to env
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

        logging.warning(
            msg=
            f'time:{time.time()} ready for send message from: {self._role}, to {agents_to_send}'
        )

        self.env_context.store_message_from_role.remote(self._role, message)

    def pull(self):
        """
        extract message from environment by role name
        Returns: prompt

        """
        recieved_messages = self.env_context.extract_message_by_role.remote(
            self._role)
        recieved_messages = ray.get(recieved_messages)
        if recieved_messages and len(recieved_messages) > 0:
            cur_step_env_prompt = self.convert_to_string(recieved_messages)
            return cur_step_env_prompt
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
