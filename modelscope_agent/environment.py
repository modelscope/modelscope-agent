import logging
import time
from queue import Queue
from typing import List, Union

import ray
from modelscope_agent.agent import Agent
from modelscope_agent.schemas import Message
from ray.util.client.common import ClientActorHandle, ClientObjectRef


@ray.remote
def run_test():
    for i in range(5):
        yield str(i)


class Environment:
    turn: str = '0'
    raw_history: str = ''
    agents: dict[str, Agent] = {}
    agents_state: dict = {}
    message_queue_persist: dict[str, Queue] = {}
    messages_queue_map: dict[str, Queue] = {}
    state: Union[str,
                 dict] = ''  # sort of transition state? shall we maintain it?
    messages_list_map: dict[str, list] = {}

    def __init__(self, agents=[], state=''):
        self.add_agents(agents)

        self.state = state
        for item in agents:
            self.messages_queue_map[item.role] = Queue()
            self.message_queue_persist[item.role] = Queue()
            self.messages_list_map[item.role] = []

    def get_message_queue_persist(self, role: str):
        return self.message_queue_persist[role]

    def get_message_list(self, role: str):
        return self.messages_list_map[role]

    def add_agent(self, agent: Union[Agent, ClientActorHandle]):
        """
        Add an agent to the environment
        Args:
            agent: Agent object

        Returns: None

        """
        if isinstance(agent, Agent):
            role = agent.role()
        else:
            role = ray.get(agent.role.remote())
        self.agents[role] = agent
        self.agents_state[role] = True
        self.messages_queue_map[role] = Queue()
        self.message_queue_persist[role] = Queue()
        self.messages_list_map[role] = []

    def add_agents(self, agents: List[Agent]):
        """
        Add a list of agents to the environment
        Args:
            agents: list of agent object

        Returns: None

        """
        if len(agents) == 0:
            return
        for item in agents:
            self.add_agent(item)

    def consume_message(self, role: str, message: Message):
        self.raw_history += f'state at {self.state}, {role}: {message.content}/n'
        recipiants = message.send_to
        if 'all' in recipiants:
            recipiants = self.agents.keys()
        for recipiant in recipiants:
            self.messages_queue_map[recipiant].put(
                Message(
                    content=message.content,
                    sent_to=recipiant,
                    sent_from=message.sent_from))
            self.message_queue_persist[recipiant].put(
                Message(
                    content=message.content,
                    sent_to=recipiant,
                    sent_from=message.sent_from))
            self.messages_list_map[recipiant].append(
                Message(
                    content=message.content,
                    sent_to=recipiant,
                    sent_from=message.sent_from))

    def step(self, task, round=1):
        for _ in range(round):
            # create a list to hold the futures of all the agents
            futures = [
                agent.agent_step.remote(task)
                for agent in self.agents.values()
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

    # def produce_message(self, role: str)
    #     messages_to_role = []
    #     while messages_queue_map[role]:
    #         messages_to_role.append(messages_queue_map[role].pop())
    #     return messages_to_role
    #
    # def produce_message_from_role(self, role: str)
    #     messages_to_role = []
    #     while messages_queue_map[role]:
    #         messages_to_role.append(messages_queue_map[role].pop())
    #     return messages_to_role
    #
    # def get_state(self):
    #     with lock:
    #         return self.state
    #
    # def update_state(self, state):
    #     with lock:
    #         self.state = state
    #
    # def get_agents(self):
    #     with lock:
    #         return self.agents
    #
    # def update_agents(self, agents):
    #     with lock:
    #         new_agents = check_duplicate(agents)
    #         self.agents_state = {agent.role: True for agent in new_agents}
    #         self.agents.append(agents)
    #
    # def disable_agent(self, agent):
    #     with lock:
    #         self.agent_state[agent.role] = False
    #

    @staticmethod
    def create_remote(cls,
                      agents=[],
                      state='',
                      *args,
                      **kwargs) -> ClientActorHandle:
        return ray.remote(
            name='env', max_concurrency=10)(cls).remote(
                agents=agents, state=state, *args, **kwargs)

    @staticmethod
    def create_local(cls, agents=[], state='', *args, **kwargs):
        return cls(agents=agents, state=state, *args, **kwargs)
