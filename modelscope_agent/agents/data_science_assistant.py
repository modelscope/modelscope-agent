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
you are a code generator, you need to generate a code python block in jupyter notebook to achieve the \
current task:
{instruction}

current task is part of the whole plan to achieve the user request:
{user_request}

previous code blocks are as follows, you need to generate python code \
that follows previous code, no need to repeat previous
code:
{previous_code_blocks}

Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse\
the code correctly,the code format is as follows:
```python
# the code you need to write
```
"""
CODE_REFLECT_TEMPLATE = """
you are a code generator, you need to generate a new code block in a new jupyter notebook to achieve the \
current task. The code should be generated based on the previous code block and the current task instruction.
we have generated some code for current task but didn't execute successfully, you can use these code \
as a reference to generate the correct code.

[current task]
{instruction}

current task is part of the whole plan to achieve the user request:
[User Request]
{user_request}

previous code blocks are as follows and have been executed successfully in the previous jupyter notebook code blocks, \
which means you can use the variables defined in the previous code blocks.\
the code you need to generate should follow previous code, no need to repeat.
[previous code blocks]
{previous_code_blocks}

the code we have generated for current task is as follows, you can use it as a reference to generate the correct code:
{code_and_error}

Attention: the code format MUST be followed, otherwise the code interpreter will not be able to parse the code \
correctly,the code format is as follows:
```python
# the code you need to write
```
"""
JUDGE_TEMPLATE = """
take a deep breath and think step by step.
you are a code judge, you need to judge the code block in jupyter notebook to achieve the \
current task.
[current task]
{instruction}

this is the code block you need to judge, it contains code and execution result:
{code}

Even if the code has been executed successfully, doesn't mean it's totally correct. You need to carefully \
check the code logic to ensure the code can accomplish the task correctly. Ignore the warning messages\

these are the previous code blocks, which have been executed successfully in the previous jupyter notebook code blocks \
{previous_code_blocks}

Attention: your response should be one of the following:
- correct, [reason]
- incorrect, [reason and advice]
"""

ERROR_KEYWORDS = ['Error']


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
        self.plan = None

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
    def _save(nb: nbformat.NotebookNode, plan: Plan, **kwargs):
        if not os.path.exists('data'):
            os.makedirs('data')
        if kwargs.get('dir_name'):
            dir_name = 'data/' + kwargs.get('dir_name') + '/'
        dir_name = 'data/' + str(
            datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if (kwargs.get('name')):
            file_name = dir_name + kwargs.get('name') + '.ipynb'
        else:
            file_name = dir_name + 'result' + '.ipynb'
        # save ipynb
        with open(file_name, 'w', encoding='utf-8') as file:
            nbformat.write(nb, file)
        # save plan.tasks
        tasks_list = plan.tasks

        # 将Task列表转换为字典列表，以便于序列化
        tasks_dict_list = [task.model_dump() for task in tasks_list]

        # 将字典列表转换为JSON字符串
        tasks_json = json.dumps(tasks_dict_list, indent=4)

        # 将JSON字符串写入文件
        with open(dir_name + 'plan.json', 'w', encoding='utf-8') as file:
            file.write(tasks_json)

    def _generate_code(self, code_counter: int, task: Task,
                       previous_code_blocks: str, user_request: str,
                       code_and_error: str):
        if code_counter == 0:
            # first time to generate code
            prompt = CODE_TEMPLATE.format(
                instruction=task.instruction,
                previous_code_blocks=previous_code_blocks,
                user_request=user_request)
        else:
            # reflect the error and ask user to fix the code
            prompt = CODE_REFLECT_TEMPLATE.format(
                instruction=task.instruction,
                previous_code_blocks=previous_code_blocks,
                code_and_error=code_and_error,
                user_request=user_request)

            # prompt = CODE_REFLECT_TEMPLATE_NEW_NEW.format(
            #     instruction=task.instruction,
            #     previous_code=previous_code_and_results,
            #     user_request=user_request,
            #     code_and_error=code_and_error)
        logger.info('\n---------\ngenerate code prompt: \n' + prompt
                    + '\n--------\n')
        messages = [{'role': 'user', 'content': prompt}]
        resp = self._call_llm(
            prompt=None,
            messages=messages,
            stop=None,
        )

        llm_result = ''
        for s in resp:
            llm_result += s
        code = parse_code(text=llm_result, lang='python')
        return code

    def _get_previous_code_blocks(self):
        previous_code_blocks = ''
        # for cell in self.code_interpreter.nb.cells:
        #     error = False
        #     if cell.cell_type == 'code':
        #         for out_put in cell.outputs:
        #             if out_put.output_type == 'error':
        #                 error = True
        #             if "name" in out_put:
        #                 if out_put.name == 'stderr':
        #                     error = True
        #         if not error:
        #             previous_code_blocks += cell.source
        counter = 0
        for task in self.plan.tasks:
            if task.is_finished:
                counter += 1
                previous_code_blocks += (
                    f'\nCodeblock_{counter}:\n```python{task.code}\n```\n'
                    f'Codeblock_{counter} Output:\n{task.result}\n')
        return previous_code_blocks

    def _judge_code(self, task, cell, previous_code_blocks):
        success = True
        failed_reason = ''
        cell = str(cell)
        judge_prompt = JUDGE_TEMPLATE.format(
            instruction=task.instruction,
            previous_code_blocks=previous_code_blocks,
            code=cell)
        logger.info(f'\n---------\njudge_prompt: \n{judge_prompt}\n--------\n')
        messages = [{'role': 'user', 'content': judge_prompt}]
        judge_resp = self._call_llm(prompt=None, messages=messages, stop=None)
        judge_result = ''
        for s in judge_resp:
            judge_result += s
        logger.info(
            f'\n---------\n judge_result: \n{judge_result}\n--------\n')
        if 'incorrect' in judge_result:
            success = False
            failed_reason = 'The code logic is incorrect, here is the reason: ' + judge_result
        return success, failed_reason

    def _run(self, user_request, save: bool = True, **kwargs):
        try:
            self.plan = self._update_plan(user_request=user_request)
            jupyter_file_path = ''
            if save:
                dir_name = 'data/' + str(
                    datetime.now().strftime('%Y-%m-%d-%H-%M-%S')) + '/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                jupyter_file_path = dir_name + 'result' + '.ipynb'

            while self.plan.current_task_id:
                task = self.plan.task_map.get(self.plan.current_task_id)

                logger.info(
                    f'new task starts: task_{task.task_id} , instruction: {task.instruction}'
                )
                previous_code_blocks = self._get_previous_code_blocks()
                code_execute_success = False
                code_counter = 0
                code_and_error = ''
                max_try = kwargs.get('max_try', 10)
                while not code_execute_success and code_counter < max_try:
                    # generate code
                    code = self._generate_code(code_counter, task,
                                               previous_code_blocks,
                                               user_request, code_and_error)

                    # execute code
                    failed_reason = ''
                    code_interpreter_resp = ''
                    try:
                        # call code interpreter to execute the code
                        code_interpreter_resp = self.code_interpreter.call(
                            params=json.dumps({'code': code}), nb_mode=True)
                        logger.info(
                            f'code_interpreter_resp task_{task.task_id} '
                            f'counter {code_counter}: \n{code_interpreter_resp}'
                        )
                        code_execute_success = not any(
                            keyword in code_interpreter_resp
                            for keyword in ERROR_KEYWORDS)
                        if not code_execute_success:
                            failed_reason = (
                                'The code execution caused error: \n'
                                + code_interpreter_resp[:5000])

                    except Exception as e:
                        code_execute_success = False
                        logger.info(
                            f'task_{task.task_id} code execution failed, counter: {code_counter}'
                        )
                        failed_reason = 'The code execution caused error: \n' + str(
                            e)[:5000]

                    if code_execute_success:
                        logger.info(
                            f'task_{task.task_id} code successfully executed , counter: {code_counter}'
                        )
                        code_execute_success, failed_reason = self._judge_code(
                            task=task,
                            previous_code_blocks=previous_code_blocks,
                            cell=self.code_interpreter.nb.cells[-1])
                        if not code_execute_success:
                            code_and_error += 'code_' + str(
                                code_counter
                                + 1) + ':\n```python\n' + code + '\n```\n'
                            code_and_error += 'code_' + str(
                                code_counter + 1
                            ) + ' error message: \n' + failed_reason + '\n'
                    else:
                        code_and_error += 'code_' + str(
                            code_counter
                            + 1) + ':\n```python\n' + code + '\n```\n'
                        code_and_error += 'code_' + str(
                            code_counter
                            + 1) + ' error message: \n' + failed_reason + '\n'

                    if not code_execute_success:
                        # delete the last cell if the code execution failed
                        del self.code_interpreter.nb.cells[-1]
                    else:
                        task.code = code
                        task.result = code_interpreter_resp
                    code_counter += 1

                    # save the successful code in jupyter notebook
                if code_execute_success:
                    self.plan.finish_current_task()
                    if save:
                        with open(
                                jupyter_file_path, 'w',
                                encoding='utf-8') as file:
                            nbformat.write(self.code_interpreter.nb, file)
                else:
                    self.plan = self._update_plan(
                        user_request=user_request, curr_plan=self.plan)
                    self.code_interpreter.nb.cells.clear()

        except Exception as e:
            logger.error(f'error: {e}')
            raise e
