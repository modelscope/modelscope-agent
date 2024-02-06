from typing import Union

import ray
from modelscope_agent.agent import Agent
from modelscope_agent.environment import Environment
from modelscope_agent.schemas import Message
from ray._raylet import ObjectRefGenerator
from ray.util.client.common import ClientActorHandle, ClientObjectRef


class AgentEnvContextMixin:

    def __init__(
        self,
        role: str,
        env: Union[Environment, ClientActorHandle] = {},
    ):
        self._role = role
        self.env_context = env
        self.cur_step_env_prompt = ''

    @staticmethod
    def create_remote(cls, role: str, function_list: list, llm, env, *args,
                      **kwargs) -> ClientActorHandle:
        return ray.remote(
            name=role, max_concurrency=10)(cls).remote(
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

    def run_test(self):
        for i in range(5):
            yield str(i)

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

        # if chat_mode:
        #     self.pull()
        #
        # last_step_env_prompt = self.cur_step_env_prompt
        #
        # # get hint messages
        # cur_step_env_prompt = convert_to_string(messages)

        # # update cur_step_env_prompt
        # cur_step_env_prompt += last_step_env_prompt

        # run agent core loop to get action or reslt
        result = ''
        for frame in self.run(prompt):
            cur_frame = frame
            result += cur_frame
            yield AgentEnvContextMixin.frame_wrapper(self._role, cur_frame)

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
        import logging
        import time
        logging.warning(msg=f'time:{time.time()} name: {self._role}')

        self.env_context.consume_message.remote(self._role, message)

    #
    # def subscribe(self, role: str):
    #     recieved_message = self.env_context.produce_message(self, self.role)
    #     self.cur_step_env_prompt = convert_to_string(recieved_message)
    #
    # # register to a callback runnable later
    # def pull(self):
    #     recieved_message = self.env_context.produce_message(self, self.role)
    #     self.cur_step_env_prompt = convert_to_string(recieved_message)
    #
    #
    #
    # def _parse_message_attribute_from_llm(llm_result):
    #     # override parse logic here for different Agent
    #     return state, message，send_to
