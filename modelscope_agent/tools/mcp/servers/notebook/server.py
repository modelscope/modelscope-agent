from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import json
from fastmcp import FastMCP

mcp = FastMCP('notebook')


@dataclass
class Task:

    name: str = ''

    system: str = ''

    result: str = ''

    _done: bool = False

    sub_tasks: List['Task'] = field(default_factory=list)

    @staticmethod
    def parse_tasks(plans: List[Union[str, Dict[str, Any]]]) -> List:
        if not plans:
            return []
        sub_tasks = []
        for plan in plans:
            if isinstance(plan, str):
                sub_tasks.append(Task(name=plan))
            else:
                step = plan['step']
                system = plan.get('system')
                substeps = plan['substeps']
                sub_tasks.append(
                    Task(name=step, system=system, sub_tasks=substeps))
        return sub_tasks

    def __post_init__(self):
        self.sub_tasks = self.parse_tasks(self.sub_tasks)

    def set_done(self):
        assert not self.sub_tasks
        self._done = True

    def get_done(self):
        if self.sub_tasks:
            self._done = all([task.get_done() for task in self.sub_tasks])
        return self._done

    @staticmethod
    def format_tasks(next_task, tasks, indent=0):
        result = ''
        for task in tasks:
            prefix = '  ' * indent

            # Determine the status symbol
            if task.get_done():
                result += f'{prefix}✓ {task.name}\n'
                if task.result:
                    result += f'{prefix}  Result: {task.result}\nResult end.\n\n'
            elif task == next_task:
                result += f'{prefix}🔄 {task.name} (CURRENT)\n'
            else:
                result += f'{prefix}• {task.name}\n'

            # Process subtasks if any
            if task.sub_tasks:
                result += Task.format_tasks(next_task, task.sub_tasks,
                                            indent + 1)

        return result


@dataclass
class Notebook:

    query: str = ''

    analysis: str = ''

    sub_tasks: List[Task] = field(default_factory=list)

    first_push: bool = True

    def override_tasks(self, plans: List[Union[str, Dict[str, Any]]]):
        self.remove_undone()
        self.sub_tasks.extend(Task.parse_tasks(plans))

    def remove_undone(self):

        def filter_tasks(tasks):
            filtered = []
            for task in tasks:
                if task.sub_tasks:
                    task.sub_tasks = filter_tasks(task.sub_tasks)
                    if task.sub_tasks or task.get_done():
                        filtered.append(task)
                elif task.get_done():
                    filtered.append(task)
            return filtered

        self.sub_tasks = filter_tasks(self.sub_tasks)

    def get_first_task(self) -> Task:
        if not self.sub_tasks:
            return None

        def find_first_undone(tasks):
            for task in tasks:
                if not task.get_done():
                    if task.sub_tasks:
                        first_undone = find_first_undone(task.sub_tasks)
                        if first_undone:
                            return first_undone
                        else:
                            raise 'Should not happen'
                    else:
                        return task
            return None

        return find_first_undone(self.sub_tasks)

    def find_main_task(self, cur_task: Task):
        for main_task in self.sub_tasks:
            if self.recursive_find_main(cur_task, main_task):
                return main_task

    @staticmethod
    def recursive_find_main(cur_task: Task, search_task: Task):
        if cur_task is search_task:
            return True
        if search_task.sub_tasks:
            for sub_task in search_task.sub_tasks:
                find = Notebook.recursive_find_main(cur_task, sub_task)
                if find:
                    return True
        return False

    def main_task_finished(self, cur_task: Task):
        main_task = self.find_main_task(cur_task)
        return main_task.get_done()

    def task_switching(self, cur_task: Task):
        main_task = self.find_main_task(cur_task)
        idx = self.sub_tasks.index(main_task)
        return main_task.get_done() and idx < len(self.sub_tasks) - 1


notebook = None


@mcp.tool(
    description=
    'Documents the user\'s original request and task requirements. Use this at the beginning of '
    'a complex task to record both the user\'s query and the specific success criteria. The '
    '\'conditions_and_todo_list\' parameter should contain a structured breakdown of completion '
    'conditions and high-level steps needed. This tool initializes the planning system and clears '
    'any existing plans.')
def initialize_task(user_query: str, conditions_and_todo_list) -> str:
    global notebook
    notebook = Notebook()
    notebook.query = user_query
    notebook.analysis = conditions_and_todo_list
    return (
        'Task initialized successfully. Now you should create a detailed step-by-step plan '
        'to address the user\'s request. Break down the task into specific, actionable steps and save '
        'them using the `create_execution_plan` tool. Support for hierarchical plans is available - '
        'you can create nested plans with main steps and sub-steps for better organization.'
    )


@mcp.tool(
    description=
    'Creates or updates your execution plan with specific actionable steps. The \'plans\' should be '
    'either a list of clear, concrete instructions or a hierarchical structure with main steps '
    'and sub-steps (using dictionaries with "step" and "substeps"). '
    'Plans are executed in the order provided. This tool can be called multiple times as needed '
    'to modify future plans, but each call must include the complete future plan. '
    'After creating a plan, use `advance_to_next_step` to start executing steps sequentially. '
    'Example format: [{"step": "Main step 1", "substeps": ["Sub-step 1.1", "Sub-step 1.2"]}, '
    '"Simple step without substeps", {"step": "Main step 3", "substeps": ["Sub-step 3.1"]}]'
)
def create_execution_plan(plans: List[Union[str, Dict[str, Any]]]) -> str:
    try:
        global notebook
        notebook.override_tasks(plans)

        return (
            'Execution plan successfully created. Now call `advance_to_next_step` '
            'with summary_and_result=null to retrieve your first action item and begin execution. '
            'Remember to call `advance_to_next_step` again with summary_and_result info '
            'after completing each step to progress through your plan.\n\n'
            'Note that main steps who have sub steps will not be executed directly - instead, '
            'you\'ll execute their substeps, and the main step will be '
            'marked as complete once all sub steps are done.')
    except Exception as e:
        return f'Error creating execution plan: {str(e)}. Please check your input format and try again.'


@mcp.tool(
    description=
    'Retrieves the next action from your execution plan and marks the current step as complete,'
    'or before the first step. '
    'Call this after finishing your current task to move to the next step. '
    'IMPORTANT: Include all essential information in the summary_and_result parameter - '
    'only this summary and the schedule will be retained between main steps. All other information '
    'from previous main steps will be lost, so ensure your summary contains everything needed '
    'to successfully complete the user\'s request. '
    'Main steps are automatically marked complete when all their sub-steps are completed.'
)
def advance_to_next_step(summary_and_result: str = '') -> str:
    global notebook
    if summary_and_result:
        notebook.first_push = False
    current_task = notebook.get_first_task()

    if current_task and summary_and_result:
        current_task.result = summary_and_result

    if current_task and not notebook.first_push:
        current_task.set_done()

    next_task = notebook.get_first_task()
    main_task = notebook.find_main_task(next_task)

    tasks_display = Task.format_tasks(next_task, notebook.sub_tasks)
    tasks_display = tasks_display.strip(
    ) if tasks_display else 'No tasks found'

    content = '📋 PLAN STATUS:\n\n'

    if notebook.query:
        content += f'📝 ORIGINAL USER QUERY:\n"{notebook.query}"\n\n'
    if notebook.analysis:
        content += f'🎯 TASK REQUIREMENTS:\n{notebook.analysis}\n\n'

    content += f'📋 TASK LIST:\n{tasks_display}\n\n'

    if next_task:
        content += f'🔄 CURRENT STEP TO EXECUTE:\n"{next_task.name}"\n\n'

        if not notebook.first_push and notebook.task_switching(current_task):
            content += (
                '⚠️ NOTE: Previous main task done, will move to the next main step.\n\n'
            )

        content += (
            'OPTIONS:\n'
            '1️⃣ Execute this step now and provide the results that need to be preserved.\n'
            '2️⃣ If this step is too complex, break it down by using `create_execution_plan` '
            'with new detailed sub-steps.\n'
            '3️⃣ If you need to revise your entire plan, use `create_execution_plan` with a new complete plan.\n\n'
            'After completing this step, call `advance_to_next_step` with a summary of your results.\n\n'
            'Please proceed with your chosen option:')
    else:
        content += (
            '⚠️ NO CURRENT STEP AVAILABLE. You have either:\n'
            '• Completed all planned steps - use `verify_task_completion` to verify completion\n'
            '• Not yet created a plan - use `create_execution_plan` to create one\n'
        )
    notebook.first_push = False
    next_step_system = None
    if main_task:
        next_step_system = main_task.system
    return json.dumps([content, next_step_system], ensure_ascii=False)


@mcp.tool(
    description=
    'Validates your completed work against the original requirements. Call this tool when you believe '
    'you\'ve finished all planned tasks. It will display the original query, success criteria, and '
    'any remaining plans for verification. Use this final check to ensure all requirements have been '
    'met before delivering your response to the user.')
def verify_task_completion() -> str:
    global notebook

    # Get tasks status
    next_task = notebook.get_first_task()
    tasks_display = Task.format_tasks(next_task, notebook.sub_tasks)
    tasks_display = tasks_display.strip() if tasks_display else 'None'

    # Check for unfinished tasks
    unfinished_warning = ''
    if next_task:
        unfinished_warning = (
            '⚠️ WARNING: You have unfinished task(s). Please complete all tasks before '
            'delivering your final response to the user.\n\n')

    content = (
        f'🔍 FINAL VERIFICATION:\n\n'
        f'📝 ORIGINAL USER QUERY:\n"{notebook.query}"\n\n'
        f'🎯 TASK REQUIREMENTS:\n{notebook.analysis}\n\n'
        f'📋 TASK STATUS:\n{tasks_display}\n\n'
        f'{unfinished_warning}'
        f'📊 COMPLETION CHECKLIST:\n'
        f'1. Have all required conditions been satisfied? (Review task requirements)\n'
        f'2. Is your answer directly responsive to the user\'s query?\n'
        f'3. Have you provided all requested information/deliverables?\n'
        f'4. Is your answer factually accurate with no contradictions?\n\n'
        f'If all requirements are satisfied, include "<task_done>" in your next response.\n'
        f'If requirements are not fully met, either:\n'
        f'• Call `advance_to_next_step` to continue executing your current plan\n'
        f'• Use `create_execution_plan` to create additional tasks if needed')

    return content


if __name__ == '__main__':
    mcp.run(transport='stdio')
