import logging
from typing import List

import ray
from modelscope_agent.schemas import Message
from ray._raylet import ObjectRefGenerator


class RayTaskExecutor:

    def __init__(self, env=None, agent_registry=None):
        if ray.is_initialized:
            ray.shutdown()
        ray.init(logging_level=logging.ERROR)
        self.env = env
        self.agent_registry = agent_registry

    def set_env(self, env):
        self.env = env

    def set_agent_registry(self, agent_registry):
        self.agent_registry = agent_registry

    def shutdown(self):
        ray.shutdown()

    def get_agent_role(self, agent) -> str:
        """
        used to get role name from agent
        Args:
            agent: an agent instance

        Returns: role name of the agent

        """
        return ray.get(agent.role.remote())

    def get_agents_by_role_names(self, role_names) -> list:
        """
        get agents by role names
        Args:
            role_names: list of role names in string

        Returns: list of agent instance

        """
        agents = ray.get(
            self.agent_registry.get_agents_by_role.remote(role_names))
        return agents

    def register_agents_and_roles(self, agents: list, roles: list):
        """
        register agent information to env and agent_registry with ray get
        Args:
            agents: list of agents instance
            roles: list of roles names

        Returns: None

        """
        ray.get(self.env.register_roles.remote(roles))
        ray.get(self.agent_registry.register_agents.remote(agents, self.env))

    def get_notified_roles(self):
        """
        get notified role from env
        Returns: role name list

        """
        notified_roles = ray.get(self.env.get_notified_roles.remote())
        return notified_roles

    def get_user_roles(self):
        """
        get user role from agent registry
        Returns: user role name list

        """
        user_roles = ray.get(
            self.agent_registry.get_user_agents_role_name.remote())
        return user_roles

    def store_message_from_role(self,
                                message: Message,
                                send_from: str = 'human'):
        """
        store message from roles
        Args:
            message: the message
            send_from: role name of the message sender

        Returns: None

        """
        ray.get(self.env.store_message_from_role.remote(send_from, message))

    def reset_queue(self):
        """
        reset the queues in env
        Returns: None

        """
        ray.get(self.env.reset_queue.remote())

    def get_generator(self, generator: ObjectRefGenerator):
        """
        get the result from a generator
        Args:
            generator:

        Returns: the next string result from generator

        """
        return ray.get(next(generator))
