from typing import List

from modelscope_agent.agent import Agent
from modelscope_agent.environment import Environment


class AgentRegistry:

    def __init__(self, remote=False, **kwargs):
        self._agents = {}
        self._agents_state = {}
        self.remote = remote
        if self.remote:
            from modelscope_agent.multi_agents_utils.executors.ray import RayTaskExecutor
            self.executor_cls = RayTaskExecutor
        else:
            from modelscope_agent.multi_agents_utils.executors.local import LocalTaskExecutor
            self.executor_cls = LocalTaskExecutor

    def register_agent(self, agent: Agent, env_context: Environment = None):
        """
        Add an agent to the register center
        Args:
            agent: Agent object
            env_context: Env context that need to pass to agent
        Returns: None

        """

        role = self.executor_cls.get_agent_role(agent)
        if role in self._agents:
            pass
            # todo: not raise error for now, need to handle this case later
            # raise ValueError(f'Role {role} already registered')
        else:
            self._agents[role] = agent
        self._agents_state[role] = True

        # set up the env_context
        if env_context:
            self.executor_cls.set_agent_env(agent, env_context)

    def get_agents_by_role(self, roles: list) -> List:
        agents = {}
        for role in roles:
            agents[role] = self.get_agent_by_role(role)
        return agents

    def get_agent_by_role(self, role: str) -> Agent:
        return self._agents.get(role)

    def get_all_role(self):
        return self._agents

    def get_available_role_name(self):

        return [role for role, state in self._agents_state.items() if state]

    def get_user_agents_role_name(self, agents: List[Agent] = None):
        if not agents:
            agents = self._agents.values()
        return [
            self.executor_cls.get_agent_role(agent) for agent in agents
            if self.executor_cls.is_user_agent(agent)
        ]

    def set_user_agent(self, role: str, human_input_mode: str = 'ON'):
        agent = self._agents.get(role)
        if agent:
            self.executor_cls.set_agent_human_input_mode(
                agent, human_input_mode)

    def unset_user_agent(self, role: str):
        agent = self._agents.get(role)
        if agent:
            self.executor_cls.set_agent_human_input_mode(agent, 'CLOSE')

    def register_agents(self,
                        agents: List[Agent],
                        env_context: Environment = None):
        """
        Add a list of agents to the environment
        Args:
            agents: list of agent object
            env_context: environment context that need to pass to agent

        Returns: None

        """
        if len(agents) == 0:
            return
        for item in agents:
            self.register_agent(item, env_context)
