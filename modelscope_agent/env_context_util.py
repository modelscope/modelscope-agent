import logging
import os
import time
from pathlib import Path
from typing import List, Union

import ray
from modelscope_agent.agent import Agent
from modelscope_agent.constants import DEFAULT_AGENT_ROOT
from modelscope_agent.environment import Environment
from modelscope_agent.memory import MemoryWithRetrievalKnowledge
from modelscope_agent.schemas import Message
from ray._raylet import ObjectRefGenerator
from ray.util.client.common import ClientActorHandle, ClientObjectRef


class AgentEnvContextMixin:

    def __init__(
        self,
        role: str,
        env: Union[Environment, ClientActorHandle] = {},
        storage_path: Union[str, Path] = DEFAULT_AGENT_ROOT,
    ):
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

    def role(self):
        """Get the name of the agent."""
        return self._role

    def agent_step(self,
                   messages: Union[str, dict, ObjectRefGenerator] = [],
                   send_to: list = [],
                   chat_mode=True):
        """
        step function for agent to interact with env and other agents
        Args:
            messages:
            send_to: the message allow to send to other agent
            chat_mode:

        Returns: ObjectRefGenerator that could be used to get result from other agent or env

        """
        # get message from other agent or env by generator
        prompt = ''
        if isinstance(messages, ObjectRefGenerator):
            remote_input = {}
            try:
                ref = next(messages)
                input_frame = AgentEnvContextMixin.extract_frame(ray.get(ref))
                if input_frame['agent'] not in remote_input:
                    remote_input[input_frame['agent']] = input_frame['content']
                else:
                    remote_input[
                        input_frame['agent']] += input_frame['content']
            except StopIteration:
                pass
            messages = remote_input
            prompt = remote_input[input_frame['agent']]
        elif isinstance(messages, dict):
            prompt = messages['content']
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
            yield AgentEnvContextMixin.frame_wrapper(self._role, cur_frame)

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

    def publish(self, result, send_to=[]):
        # parse current state and message from
        # state, message, send_to_by_model = self._parse_message_attribute_from_llm(llm_result)
        # env_state = self.env_context.get_state()
        # make sure state maintain the same
        env_state = True
        state = env_state
        agents_to_send = {'all'}
        if env_state == state:
            if len(send_to) > 0:
                # user defined logic is in the primary
                agents_to_send = send_to
            # elif len(send_to_by_model) > 0 and use_rule_from_llm:
            #     # if user use parse from model
            #     agents_to_send = check_valid(send_to_by_model)
            # else:
            #     # rule based
            #     agents = self.env_context.get_agents()
            #     agents_to_send = self.rules.send_to_group(self.role, agents, state)
        else:
            # mismatched state should send no message
            agents_to_send = [], message = None

        # agents_to_send = self.remove_self(agents_to_send)
        message = Message(
            content=result, send_to=agents_to_send, sent_from=self._role)

        logging.warning(msg=f'time:{time.time()} name: {self._role}')

        self.env_context.store_message_from_role.remote(self._role, message)

    #
    # def subscribe(self, role: str):
    #     recieved_message = self.env_context.produce_message(self, self.role)
    #     self.cur_step_env_prompt = convert_to_string(recieved_message)
    #
    # register to a callback runnable later
    def pull(self):
        """
        extract
        Returns:

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

    #
    #
    # def _parse_message_attribute_from_llm(llm_result):
    #     # override parse logic here for different Agent
    #     return state, message，send_to
