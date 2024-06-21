import logging
from typing import Union

from modelscope_agent.agents_registry import AgentRegistry
from modelscope_agent.constants import USER_REQUIREMENT
from modelscope_agent.environment.environment import Environment
from modelscope_agent.schemas import Message
from ray._raylet import ObjectRefGenerator

try:
    import ray
except ImportError:
    logging.error(
        'Ray is not installed, please install ray first by running `pip install ray>=2.9.4`'
    )


class RayTaskExecutor:

    @staticmethod
    def init_ray():

        if ray.is_initialized:
            ray.shutdown()
        ray.init(logging_level=logging.ERROR)

    @staticmethod
    def shutdown_ray():
        ray.shutdown()

    @staticmethod
    def get_agents_by_role_names(agent_registry: AgentRegistry,
                                 role_names: list) -> list:
        """
        get agents by role names
        Args:
            agent_registry: the agent_resgistry instance
            role_names: list of role names in string

        Returns: list of agent instance

        """
        agents = ray.get(agent_registry.get_agents_by_role.remote(role_names))
        return agents

    @staticmethod
    def register_agents_and_roles(env: Environment,
                                  agent_registry: AgentRegistry, agents: list,
                                  roles: list):
        """
        register agent information to env and agent_registry
        Args:
            env: the environment instance
            agent_registry: the agent_registry instance
            agents: list of agents instance
            roles: list of roles names

        Returns: None

        """
        ray.get(env.register_roles.remote(roles))
        ray.get(agent_registry.register_agents.remote(agents, env))

    @staticmethod
    def get_notified_roles(env) -> list:
        """
        get notified role from env
        Args:
            env: the environment instance
        Returns: role name list

        """
        notified_roles = ray.get(env.get_notified_roles.remote())
        return notified_roles

    @staticmethod
    def get_user_roles(agent_registry: AgentRegistry) -> list:
        """
        get user role from agent registry
         Args:
            agent_registry: the agent_registr instance
        Returns: user role name list

        """
        user_roles = ray.get(agent_registry.get_user_agents_role_name.remote())
        return user_roles

    @staticmethod
    def store_message_from_role(env: Environment,
                                message: Message,
                                send_from: str = USER_REQUIREMENT):
        """
        store message from roles
        Args:
            env: the environment instance
            message: the message
            send_from: role name of the message sender

        Returns: None

        """
        ray.get(env.store_message_from_role.remote(send_from, message))

    @staticmethod
    def reset_queue(env: Environment):
        """
        reset the queues in env
        Args:
            env: the environment instance

        Returns: None

        """
        ray.get(env.reset_queue.remote())

    @staticmethod
    def get_generator_result(generator: ObjectRefGenerator):
        """
        get the result from a generator
        Args:
            generator:

        Returns: the next string result from generator

        """
        return ray.get(next(generator))

    @staticmethod
    def get_agent_step_future(agent,
                              messages: Union[str, dict],
                              send_to: Union[str, list],
                              user_response: str = None,
                              **kwargs):
        """
        get the future from agent step, the method referred to AgentEnvMixin.step.
        Args:
            agent: an agent instance
            messages: the message that send to the current agent as input
            send_to: the message that allows to send to other agents
            user_response: the output from user, could be treated as LLM output's alternative
                sort of the step function's output if human input mode is on
            kwargs: additional keywords, such as runtime llm setting


        Returns: the next string result from agent step

        """

        return agent.step.remote(messages, send_to, user_response, **kwargs)

    @staticmethod
    def get_agent_role(agent) -> str:
        """
        used to get role name from agent
        Args:
            agent: an agent instance

        Returns: role name of the agent

        """
        return ray.get(agent.role.remote())

    @staticmethod
    def is_user_agent(agent) -> bool:
        """
        To decide if is the agent a user
        Args:
            agent: an agent instance

        Returns: if is the agent a user return True, else False

        """
        return ray.get(agent.is_user_agent.remote())

    @staticmethod
    def update_agent_memory(agent, messages: list):
        """
        update agent memory
        Args:
            agent:  the agent instance
            messages: the message history

        Returns:

        """
        ray.get(agent.update_memory.remote(messages))

    @staticmethod
    def set_agent_human_input_mode(agent, human_input_mode: str):
        """

        Args:
            agent: the agent instance
            human_input_mode: ON, CLOSE or TERMINAL

        Returns: None

        """
        ray.get(agent.set_human_input_mode.remote(human_input_mode))

    @staticmethod
    def set_agent_env(agent, env_context: Environment):
        """
        set env context in agent
        Args:
            agent: the agent instance
            env_context: the env instance

        Returns: None

        """
        ray.get(agent.set_env_context.remote(env_context))

    @staticmethod
    def extract_message_by_role_from_env(env_context: Environment,
                                         role: str) -> list:
        """
        extract message by role from env
        Args:
            env_context: env context
            role: role name in string

        Returns: The messages that role has seen

        """
        received_messages = ray.get(
            env_context.extract_message_by_role.remote(role))
        return received_messages

    @staticmethod
    def extract_all_message_from_env(env_context: Environment) -> list:
        """
        extract all message from env
        Args:
            env_context: env context

        Returns: All messages in last round

        """
        received_messages = ray.get(
            env_context.extract_all_history_message.remote())
        return received_messages
