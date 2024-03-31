from typing import List, Union

from modelscope_agent import create_component
from modelscope_agent.agent import Agent
from modelscope_agent.agents_registry import AgentRegistry
from modelscope_agent.constants import DEFAULT_SEND_TO
from modelscope_agent.environment import Environment
from modelscope_agent.schemas import Message
from modelscope_agent.utils.logger import agent_logger as logger


class TaskCenter:

    def __init__(self, remote=False):
        if remote:
            from modelscope_agent.multi_agents_tasks.executors.ray import RayTaskExecutor
            self.task_executor = RayTaskExecutor
            self.task_executor.init_ray()
        else:
            from modelscope_agent.multi_agents_tasks.executors.local import LocalTaskExecutor
            self.task_executor = LocalTaskExecutor
        self.env = create_component(Environment, 'env', remote)
        self.agent_registry = create_component(AgentRegistry, 'agent_center',
                                               remote)
        self.remote = remote

    def __del__(self):
        if self.remote:
            self.task_executor.shutdown_ray()

    def add_agents(self, agents: List[Agent]):
        """
        add agents to the task scope
        Args:
            agents: should be either local agent or remote agent

        Returns:

        """
        roles = []
        for agent in agents:
            agent_role = self.task_executor.get_agent_role(agent)
            logger.info(f'Adding agent to task center: {agent_role}')
            roles.append(agent_role)
        self.task_executor.register_agents_and_roles(self.env,
                                                     self.agent_registry,
                                                     agents, roles)

    def disable_agent(self, agent):
        pass

    def is_user_agent_present(self, roles: List[str] = []):
        if len(roles) == 0:
            roles = self.task_executor.get_notified_roles(self.env)
        user_roles = self.task_executor.get_user_roles(self.agent_registry)
        notified_user_roles = list(set(roles) & set(user_roles))

        return notified_user_roles

    def send_task_request(self,
                          task,
                          send_to: Union[str, list] = DEFAULT_SEND_TO,
                          send_from: str = 'human'):
        """
        Send the task request by send the message to the environment
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
        self.task_executor.store_message_from_role(self.env, message,
                                                   send_from)
        logger.info(f'Send init task, {task} to {send_to}')

    def reset_env(self):
        self.task_executor.reset_queue(self.env)

    def step(self,
             task=None,
             round: int = 1,
             send_to: Union[str, list] = DEFAULT_SEND_TO,
             allowed_roles: list = [],
             user_response: str = None,
             **kwargs):
        """
        Core step to make sure
        Args:
            task: additional task in current step
            round: current step might have multi round
            send_to: manually define the message send to which role
            allowed_roles: make sure only the notified role can be step
            user_response: using the user response to replace the llm output from user_agent,
                if user_agent is in this step
            kwargs: additional keywords, such as runtime llm setting

        Returns:
            ray's object ref generator
        """
        # convert single role to list
        if isinstance(send_to, str):
            send_to = [send_to]

        # get current steps' agent from env or from input
        if len(allowed_roles) == 0:
            roles = self.task_executor.get_notified_roles(self.env)
        else:
            roles = allowed_roles

        if len(roles) == 0:
            return

        agents = self.task_executor.get_agents_by_role_names(
            self.agent_registry, roles)

        for _ in range(round):
            # create a list to hold the futures of all notified agents
            futures = [
                agent.step.remote(task, send_to, user_response, **kwargs)
                for agent in agents.values()
            ]

            # wait for the agents to finish
            finish_flag = {}
            while True:
                for future in futures:
                    try:
                        # try to get the next result from the agent
                        result = self.task_executor.get_generator_result(
                            future)
                        yield result
                    except StopIteration:
                        # if the agent has no more results, break
                        finish_flag[future] = True

                #  the number of finish flag equals to the num of agents
                if len(finish_flag.keys()) == len(futures):
                    break


class LocalTaskExecutor:

    def __init__(self):
        self.task_center = TaskCenter(remote=False)
