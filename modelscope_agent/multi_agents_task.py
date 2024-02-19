import logging
from typing import List, Union

import ray
from modelscope_agent import create_component
from modelscope_agent.agent import Agent
from modelscope_agent.agents_registry import AgentRegistry
from modelscope_agent.constants import DEFAULT_SEND_TO
from modelscope_agent.environment import Environment
from modelscope_agent.schemas import Message


class TaskCenter:

    def __init__(self, remote=False):
        if remote:
            if ray.is_initialized:
                ray.shutdown()
            ray.init(logging_level=logging.ERROR)
        self.env = create_component(Environment, remote)
        self.agent_registry = create_component(AgentRegistry, remote)
        self.remote = remote

    def add_agents(self, agents: List[Agent]):
        """
        add agents to the task scope
        Args:
            agents: should be either local agent or remote agent

        Returns:

        """
        if self.remote:
            ray.get(self.agent_registry.register_agents(agents, self.env))
        else:
            self.agent_registry.register_agents(agents, self.env)

    def disable_agent(self, agent):
        pass

    def start_task(self,
                   task,
                   send_to: Union[str, list] = DEFAULT_SEND_TO,
                   send_from: str = 'human'):
        """
        Start the task by send the first message to the environment
        Args:
            task: the task from user
            send_to: send to the message to whom
            send_from: the message might from other than human

        Returns:

        """

        if isinstance(send_to, str):
            send_to = [send_to]

        message = Message(
            role=send_from,
            content=task,
            send_to=send_to,
            sent_from=send_from,
        )
        self.env.store_message_from_role(send_from, message)

    def step(self, task=None, round: int = 1, send_to: Union[str, list] = []):
        """
        Core step to make sure
        Args:
            task: additional task in current step
            round: current step might have multi round
            send_to: manually define the message send to which role

        Returns:
            ray's object ref generator
        """
        # convert single role to list
        if isinstance(send_to, str):
            send_to = [send_to]

        # get current steps' agent
        roles = ray.get(self.env.get_notified_roles.remote())
        if len(roles) == 0:
            return
        agents = ray.get(self.agent_registry.get_agents_by_role.remote(roles))

        for _ in range(round):
            # create a list to hold the futures of all notified agents
            futures = [
                agent.step.remote(task, send_to) for agent in agents.values()
            ]

            # wait for the agents to finish
            finish_flag = {}
            while True:
                for future in futures:
                    try:
                        # try to get the next result from the agent
                        result = ray.get(next(future))
                        yield result
                    except StopIteration:
                        # if the agent has no more results, break
                        finish_flag[future] = True

                #  the number of finish flag equals to the num of agents
                if len(finish_flag.keys()) == len(futures):
                    break
