from modelscope_agent.schemas import Message


class LocalTaskExecutor:
    """
    This class used to

    """

    def __init__(self, env=None, agent_registry=None):
        self.env = env
        self.agent_registry = agent_registry

    def set_env(self, env):
        self.env = env

    def set_agent_registry(self, agent_registry):
        self.agent_registry = agent_registry

    def shutdown(self):
        pass

    def get_agent_role(self, agent):
        """
        used to get role name from agent
        Args:
            agent: an agent instance

        Returns: role name of the agent

        """
        return agent.role()

    def get_agents_by_role_names(self, role_names) -> list:
        """
        get agents by role names
        Args:
            role_names: list of role names in string

        Returns: list of agent instance

        """
        agents = self.agent_registry.get_agents_by_role(role_names)
        return agents

    def register_agents_and_roles(self, agents, roles):
        """
        register agent information to env and agent_registry
        Args:
            agents: list of agents instance
            roles: list of roles names

        Returns: None

        """
        self.env.register_roles(roles)
        self.agent_registry.register_agents(agents, self.env)

    def get_notified_roles(self) -> list:
        """
        get notified role from env
        Returns: role name list

        """
        notified_roles = self.env.get_notified_roles()
        return notified_roles

    def get_user_roles(self) -> list:
        """
        get user role from agent registry
        Returns: user role name list

        """
        user_roles = self.agent_registry.get_user_agents_role_name()
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
        self.env.store_message_from_role(send_from, message)

    def reset_queue(self):
        """
        reset the queues in env
        Returns: None

        """
        self.env.reset_env_queues()

    def get_generator_result(self, generator) -> str:
        """
        get the result from a generator
        Args:
            generator:

        Returns: the next string result from generator

        """
        return next(generator)
