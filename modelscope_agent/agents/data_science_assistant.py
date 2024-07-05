# Implementation inspired by the paper "DATA INTERPRETER: AN LLM AGENT FOR DATA SCIENCE"

import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import json
import json5
import nbformat
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.schemas import Plan, Task
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.tools.code_interpreter import CodeInterpreter
from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.utils import parse_code

DATA_SCIENTIST_TEMPLATE = """As a data scientist, you need to help user to achieve their goal step by step in a \
continuous Jupyter notebook.Since it is a notebook environment, don\'t use asyncio.run. Instead, use await if you
need to call an async function."""
PLAN_TEMPLATE = """
# Context:
{context}
# Available Task Types:
code - write code to solve the problem

# Task:
Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan \
consists of one to four tasks.
If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. \
Give the whole plan unless instructed to modify only one task of the plan.
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
you are a code fixer, you need to fix python a code block in jupyter notebook to achieve the\
current task:
{instruction}

current task is part of the whole plan to achieve the user request:
{user_request}

the code format is as follows:
```python
# the code you need to write
```
previous code are as follows, you need to generate python code that follows previous code, no need to repeat previous \
code:
{previous_code}
Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse\
the code correctly,the code format is as follows:
```python
# the code you need to write
```
"""
CODE_REFLECT_TEMPLATE = """

you are a code fixer, you need to fix python a code block in jupyter notebook to achieve the\
current task:
{instruction}

current task is part of the whole plan to achieve the user request:
{user_request}

the code format is as follows:
```python
# the code you need to write
```

the code you need to fix is as follows:
```python
{code}
```

but the code is not correct, and caused the following error:
{error}
please correct the code and try again

previous code are as follows and have been executed successfully in the previous jupyter notebook code blocks, \
which means you can use the variables defined in the previous code blocks.
the code you need to fix should follow previous code, no need to repeat
previous code:
{previous_code}

Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse the code \
correctly,the code format is as follows:
```python
# the code you need to write
```
"""
ERROR_KEYWORDS = ['Error', 'error', 'ERROR', 'Traceback', 'exception']


class DataScienceAssistant(RolePlay):

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
        self.code_interpreter = CodeInterpreter()

    def _update_plan(self, user_request: str, curr_plan: Plan = None) -> Plan:
        resp = self._call_llm(
            prompt=PLAN_TEMPLATE.format(
                context='User Request: ' + user_request + '\n', ),
            messages=None,
            stop=None)
        tasks_text = ''
        for r in resp:
            tasks_text += r
        tasks_text = parse_code(text=tasks_text, lang='json')
        logger.info(f'tasks: {tasks_text}')
        tasks = json5.loads(tasks_text)
        tasks = [Task(**task) for task in tasks]
        if curr_plan is None:
            new_plan = Plan(goal=user_request)
            new_plan.add_tasks(tasks=tasks)
            return new_plan
        else:
            if len(tasks) == 1 or tasks[0].dependent_task_ids:
                if tasks[0].dependent_task_ids and len(tasks) > 1:
                    logger.warning(
                        'Current plan will take only the first generated task if the generated tasks are not a '
                        'complete plan')

                if curr_plan.has_task_id(tasks[0].task_id):

                    curr_plan.replace_task(tasks[0])
                else:

                    curr_plan.append_task(tasks[0])
            else:

                curr_plan.add_tasks(tasks)
            return curr_plan

    @staticmethod
    def _save(nb: nbformat.NotebookNode):
        if not os.path.exists('data'):
            os.makedirs('data')
        file_name = 'data/' + str(
            datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.ipynb'
        with open(file_name, 'w', encoding='utf-8') as file:
            nbformat.write(nb, file)

    def _run(self, user_request, save: bool = True, **kwargs):
        try:
            plan = self._update_plan(user_request=user_request)
            logger.info(f'plan: {plan}')

            while plan.current_task_id:
                task = plan.task_map.get(plan.current_task_id)
                logger.info(f'task: {task}')
                previous_code = ''
                for cell in self.code_interpreter.nb.cells:
                    error = False
                    if cell.cell_type == 'code':
                        for out_put in cell.outputs:
                            if out_put.output_type == 'error':
                                error = True
                        if not error:
                            previous_code += cell.source
                success = False
                counter = 0
                resp = ''
                code = ''

                while not success and counter < 5:

                    if counter == 0:
                        # first time to generate code
                        prompt = CODE_TEMPLATE.format(
                            instruction=task.instruction,
                            previous_code=previous_code,
                            user_request=user_request)
                    else:
                        # reflect the error and ask user to fix the code
                        prompt = CODE_REFLECT_TEMPLATE.format(
                            instruction=task.instruction,
                            previous_code=previous_code,
                            code=code,
                            user_request=user_request,
                            error=resp[:10000])
                    resp = self._call_llm(
                        prompt=prompt,
                        messages=None,
                        stop=None,
                    )

                    code = ''
                    for chunk in resp:
                        code += chunk
                    code = parse_code(text=code, lang='python')
                    try:
                        # call code interpreter to execute the code
                        resp = self.code_interpreter.call(
                            params=json.dumps({'code': code}), nb_mode=True)
                        success = not any(keyword in resp
                                          for keyword in ERROR_KEYWORDS)
                    except Exception as e:
                        success = False
                        resp = 'Error: ' + str(e)

                    counter += 1
                if success:
                    plan.finish_current_task()
                else:
                    plan = self._update_plan(
                        user_request=user_request, curr_plan=plan)
            if save:
                self._save(self.code_interpreter.nb)
        except Exception as e:
            logger.error(f'error: {e}')
            raise e
