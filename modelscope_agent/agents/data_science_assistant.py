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
CODE_TEMPLATE = """you are a code interpreter, you need to generate a new jupyter notebook code block based on the \
instruction: {instruction}, the code format is as follows:
```python
# the code you need to write
```
previous code are as follows, you need to generate python code that follows previous code, no need to repeat previous \
code:
{previous_code} Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse
the code correctly,the code format is as follows: ```python # the code you need to write ```"""
CODE_REFLECT_TEMPLATE = """you are a code fixer, you need to fix python code block in jupyter notebook to achieve the \
goal: {instruction}, the code format is as follows: ```python # the code you need to write ```

the code you need to fix is:
```python
{code}
```

but the code is not correct, the code caused the following error:
{error}
please correct the code and try again

previous code blocks are as follows, you need to generate python code that follows previous code, no need to repeat
previous code: {previous_code}

Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse the code
correctly,the code format is as follows: ```python # the code you need to write ```"""


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

    def _run(self,
             user_request,
             history: Optional[List[Dict]] = None,
             ref_doc: str = None,
             image_url: Optional[List[Union[str, Dict]]] = None,
             lang: str = 'zh',
             **kwargs):
        try:
            resp = self._call_llm(
                prompt=PLAN_TEMPLATE.format(
                    context='User Request: ' + user_request + '\n', ),
                messages=None,
                stop=None,
                **kwargs)
            tasks_text = ''
            for r in resp:
                tasks_text += r
            logger.info(f'tasks: {tasks_text}')
            tasks_text = parse_code(text=tasks_text, lang='json')
            tasks = json5.loads(tasks_text)
            tasks = [Task(**task) for task in tasks]
            plan = Plan(goal=user_request)
            plan.add_tasks(tasks=tasks)
            logger.info(f'plan: {plan}')
            code_interpreter = CodeInterpreter()
            while plan.current_task_id:
                task = plan.task_map.get(plan.current_task_id)
                logger.info(f'task: {task}')
                previous_code = ''
                for cell in code_interpreter.nb.cells:
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
                print('previous_code:\n', previous_code)
                while not success and counter < 5:
                    if counter == 0:
                        resp = self._call_llm(
                            prompt=CODE_TEMPLATE.format(
                                instruction=task.instruction,
                                previous_code=previous_code),
                            messages=None,
                            stop=None,
                            **kwargs)
                    else:
                        resp = self._call_llm(
                            prompt=CODE_REFLECT_TEMPLATE.format(
                                instruction=task.instruction,
                                previous_code=previous_code,
                                code=code,
                                error=resp[:10000]),
                            messages=None,
                            stop=None,
                            **kwargs)

                    for chunk in resp:
                        code += chunk
                    code = parse_code(text=code, lang='python')

                    kwargs = {'code': code}
                    try:
                        resp = code_interpreter.call(params=json.dumps(kwargs))
                        if 'Error: ' in resp:
                            success = False
                        else:
                            success = True
                    except Exception as e:
                        print(f'error: {e}')
                    print('code interpreter response:\n', resp)
                    counter += 1
                if success:
                    plan.finish_current_task()
                # todo: add update task logic
            # if data dir does not exist, create it
            if not os.path.exists('data'):
                os.makedirs('data')
            file_name = 'data/' + str(
                datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '.ipynb'
            with open(file_name, 'w', encoding='utf-8') as file:
                nbformat.write(code_interpreter.nb, file)
        except Exception as e:
            logger.error(f'error: {e}')
            raise e
