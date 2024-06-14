import os
import re
from typing import Dict, List, Optional, Tuple, Union

import json5
from modelscope_agent import Agent
from modelscope_agent.agent_env_util import AgentEnvMixin
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.schemas import Plan, Task
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.tokenization_utils import count_tokens
from modelscope_agent.utils.utils import (check_and_limit_input_length,
                                          parse_code)

PLAN_TEMPLATE = """
# User Request:
{context}
# Available Task Types:
code - write code to solve the problem
# Task:
Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan \
consists of one to four tasks.
If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole\
 plan unless instructed to modify only one task of the plan.
If you encounter errors on the current task, revise and output the current single task only.
Output a list of jsons following the format:
```json
[
    {{
        "task_id": str = "unique identifier for a task in plan, can be an ordinal",
        "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
        "instruction": "what you should do in this task, one short phrase or sentence",
        "task_type": "type of this task, should be one of Available Task Types",
    }},
    ...
]
```
"""
CODE_TEMPLATE = """
You are a python code helper, you need to generate python code based on the instruction: {instruction}, the code \
format is as follows:
```python
# the code you need to write
```
"""


class PlanAndActAgent(RolePlay):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 **kwargs):
        super().__init__(
            function_list=function_list,
            llm=llm,
            storage_path=storage_path,
            name=name,
            description=description,
            instruction=instruction,
            **kwargs)

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             image_url: Optional[List[Union[str, Dict]]] = None,
             lang: str = 'zh',
             **kwargs):
        resp = super()._run(
            PLAN_TEMPLATE.format(
                context='User Request: ' + user_request + '\n', ),
            history=history,
            ref_doc=ref_doc,
            image_url=image_url,
            lang=lang,
            **kwargs)
        tasks = ''
        for r in resp:
            tasks += r
        logger.info(f'tasks: {tasks}')
        tasks = parse_code(text=tasks, block=None, lang='json')
        tasks = json5.loads(tasks)
        tasks = [Task(**task) for task in tasks]
        plan = Plan(goal=user_request)
        plan.add_tasks(tasks=tasks)
        logger.info(f'plan: {plan}')
        while plan.current_task_id:
            task = plan.task_map.get(plan.current_task_id)
            logger.info(f'task: {str(task)}')
            # TODO: support tool use for plan and act agent.
            # TODO: support plan update according to the task result.
            plan.finish_current_task()
