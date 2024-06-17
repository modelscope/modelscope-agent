from typing import Union

from modelscope_agent.agents_registry import AgentRegistry
from modelscope_agent.constants import USER_REQUIREMENT
from modelscope_agent.environment.environment import Environment
from modelscope_agent.schemas import Message


class LocalTaskExecutor:
    """
    This class used to

    """

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
        agents = agent_registry.get_agents_by_role(role_names)
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
        env.register_roles(roles)
        agent_registry.register_agents(agents, env)

    @staticmethod
    def get_notified_roles(env) -> list:
        """
        get notified role from env
        Args:
            env: the environment instance
        Returns: role name list

        """
        notified_roles = env.get_notified_roles()
        return notified_roles

    @staticmethod
    def get_user_roles(agent_registry: AgentRegistry) -> list:
        """
        get user role from agent registry
         Args:
            agent_registry: the agent_registr instance
        Returns: user role name list

        """
        user_roles = agent_registry.get_user_agents_role_name()
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
        env.store_message_from_role(send_from, message)

    @staticmethod
    def reset_queue(env: Environment):
        """
        reset the queues in env
        Args:
            env: the environment instance

        Returns: None

        """
        env.reset_env_queues()

    @staticmethod
    def get_generator_result(generator) -> str:
        """
        get the result from a generator
        Args:
            generator:

        Returns: the next string result from generator

        """
        return next(generator)

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

        return agent.step(messages, send_to, user_response, **kwargs)

    @staticmethod
    def get_agent_role(agent):
        """
        used to get role name from agent
        Args:
            agent: an agent instance

        Returns: role name of the agent

        """
        return agent.role()

    @staticmethod
    def is_user_agent(agent) -> bool:
        """
        To decide if is the agent a user
        Args:
            agent: an agent instance

        Returns: if is the agent a user return True, else False

        """
        return agent.is_user_agent.remote()

    @staticmethod
    def set_agent_human_input_mode(agent, human_input_mode: str):
        """

        Args:
            agent: the agent instance
            human_input_mode: ON, CLOSE or TERMINAL

        Returns: None

        """
        agent.set_human_input_mode(human_input_mode)

    @staticmethod
    def update_agent_memory(agent, messages: list):
        """
        update agent memory
        Args:
            agent:  the agent instance
            messages: the message history

        Returns:

        """
        agent.update_memory(messages)

    @staticmethod
    def set_agent_env(agent, env_context: Environment):
        """
        set env context in agent
        Args:
            agent: the agent instance
            env_context: the env instance

        Returns: None

        """
        agent.set_env_context(env_context)

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
        received_messages = env_context.extract_message_by_role(role)
        return received_messages

    @staticmethod
    def extract_all_message_from_env(env_context: Environment) -> list:
        """
        extract all message from env
        Args:
            env_context: env context

        Returns: All messages in last round

        """
        received_messages = env_context.extract_all_history_message()
        return received_messages
