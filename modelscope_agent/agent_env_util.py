import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from modelscope_agent.constants import DEFAULT_AGENT_ROOT, DEFAULT_SEND_TO
from modelscope_agent.environment import Environment
from modelscope_agent.memory import Memory
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger


class AgentEnvMixin:

    def __init__(self,
                 role: str = 'default_role',
                 env: Union[Environment] = None,
                 storage_path: Union[str, Path] = DEFAULT_AGENT_ROOT,
                 is_watcher: bool = False,
                 use_history: bool = True,
                 human_input_mode: Optional[str] = 'CLOSE',
                 parse_env_prompt_function: Callable = None,
                 remote=False,
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
            human_input_mode: human input mode, which is used to control the human input mode,
                including: CLOSE, ON, TERMINAL
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
        self.human_input_mode = human_input_mode

        if not parse_env_prompt_function:
            self.parse_env_prompt_function = self.convert_to_string
        else:
            self.parse_env_prompt_function = parse_env_prompt_function
        assert isinstance(self.parse_env_prompt_function, Callable)

        memory_path = os.path.join(storage_path, role, 'memory')

        self.memory = Memory(path=memory_path, )
        if self.remote:
            from modelscope_agent.multi_agents_utils.executors.ray import RayTaskExecutor
            self.executor_cls = RayTaskExecutor
        else:
            from modelscope_agent.multi_agents_utils.executors.local import LocalTaskExecutor
            self.executor_cls = LocalTaskExecutor

    def set_env_context(self, env_context):
        if env_context:
            self.env_context = env_context

    def update_memory(self, messages: List[Message]):
        """
        update memory with messages
        Args:
            messages: list of messages

        Returns: None

        """
        if self.use_history:
            self.memory.update_history(messages)

    def set_remote(self, remote):
        self.remote = remote

    def set_human_input_mode(self, human_input_mode):
        self.human_input_mode = human_input_mode

    def is_user_agent(self):
        return self.human_input_mode == 'ON' or self.human_input_mode == 'TERMINAL'

    def role(self):
        """Get the name of the agent."""
        return self._role

    def step(self,
             messages: Union[str, dict] = None,
             send_to: Union[str, list] = DEFAULT_SEND_TO,
             user_response: str = None,
             **kwargs):
        """
        step function for agent to interact with env and other agents
        Args:
            messages: the message that send to the current agent as input
            send_to: the message that allows to send to other agents
            user_response: the output from user, could be treated as LLM output's alternative
                sort of the step function's output if human input mode is on
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
        if isinstance(messages, dict):
            prompt = messages['content']
        elif messages is None:
            prompt = ''
        else:
            prompt = messages

        cur_step_env_info = self.pull()
        prompt += cur_step_env_info

        # run agent core loop to get action or result
        result = ''
        logger.info(f'{self._role}\'s current prompt is: {prompt}')

        user_not_response = True
        # In some case user might run the agent in terminal mode without remote, then use this
        if self.human_input_mode == 'TERMINAL' and not self.remote:
            result = input(
                f'You are {self.role()}. Press enter to skip and use auto-reply, '
                f'or input any information to talk with other roles: ')
            user_not_response = True if not result else False
            if not user_not_response:
                yield AgentEnvMixin.frame_wrapper(self._role, result)

        # In the most cases, user input will come from task center, then use this
        if self.human_input_mode == 'ON' or (self.human_input_mode
                                             == 'TERMINAL' and self.remote):
            result = user_response
            user_not_response = True if not result else False
            if not user_not_response:
                self.publish(result, send_to)
                # user response is a response from user input, don't yield it to system as response again.
                return

        # If human input mode is close, or human input is empty, then run the generation,
        if self.human_input_mode == 'CLOSE' or not user_not_response:
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

    def publish(self, result, send_to: list = [DEFAULT_SEND_TO]):
        # parse current state and message from llm
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
            # received_messages = self.executor_cls.extract_message_by_role_from_env(
            #    self.env_context, self._role)
            received_messages = self.executor_cls.extract_all_message_from_env(
                self.env_context)
            if received_messages and len(received_messages) > 0:
                cur_step_env_prompt = self.parse_env_prompt_function(
                    received_messages)
                return cur_step_env_prompt
            else:
                return ''
        else:
            # watcher could see all message
            received_messages = self.executor_cls.extract_all_message_from_env(
                self.env_context)

            if received_messages and len(received_messages) > 0:
                conversation_history = self.parse_env_prompt_function(
                    received_messages)
                return conversation_history.strip()
            else:
                return ''

    def convert_to_string(self, messages: List[Message], max_turn=15):
        prompt_template = """{conversation_history}"""
        conversation_history = ''
        for item in messages[-1 * max_turn:]:
            conversation_history += f'{item.sent_from}: {item.content}\n'
        return prompt_template.format(
            conversation_history=conversation_history.strip())

    def _check_env_ready(self):
        if self.env_context:
            return True
        else:
            return False
