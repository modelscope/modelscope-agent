from copy import deepcopy

from modelscope_agent.agents.codexgraph_agent.cypher_agent import \
    CODE_SEARCH_FORMAT
from modelscope_agent.agents.codexgraph_agent.task.code_general import \
    CodexGraphAgentGeneral
from modelscope_agent.agents.codexgraph_agent.utils.code_utils import \
    extract_text_between_markers
from modelscope_agent.agents.codexgraph_agent.utils.prompt_utils import (
    load_prompt_template, replace_system_prompt)
from modelscope_agent.environment.graph_database import GraphDatabaseHandler

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
You ultimate goal is to write a patch that resolves this issue.
"""

BUG_LOCALIZATION_FORMAT = """[start_of_bug_locations]
### Text Description 1 for Bug Location
- Concise text description for the bug location: <text_description_of_the_buggy_location>
- Why: <the_reason_why_it_is_buggy>

### Text Description 2 for Bug Location
- Concise text description for the bug location: <text_description_of_the_buggy_location>
- Why: <the_reason_why_it_is_buggy>
...
### Text Description n for Bug Location
- Concise text description for the bug location: <text_description_of_the_buggy_location>
- Why: <the_reason_why_it_is_buggy>
[end_of_bug_locations]
"""


def response_to_msg(extraced_analysis, extraced_code_search,
                    extraced_bug_location):
    msg = ''
    if extraced_analysis:
        msg += f'## analysis\n\n{extraced_analysis}\n\n'
    if extraced_code_search:
        msg += f'## code_search\n\n{extraced_code_search}\n\n'
    if extraced_bug_location:
        msg += f'## bug_location\n\n{extraced_bug_location}\n\n'
    return msg


def markdown_answer(answer):
    # <file>...</file>
    # <original>...</original>
    # <patched>...</patched>
    answer = answer.replace('```', '')
    replace_dict = {
        '<file>': '\n```text\n',
        '</file>': '\n```\n',
        '<original>': '\n## Original: \n```python\n',
        '</original>': '\n```\n',
        '<patched>': '\n## Patched: \n```python\n',
        '</patched>': '\n```\n',
    }
    for key, value in replace_dict.items():
        answer = answer.replace(key, value)
    return answer


class CodexGraphAgentDebugger(CodexGraphAgentGeneral):

    def __init__(self,
                 llm,
                 prompt_path: str,
                 schema_path: str,
                 task_id: str,
                 graph_db: GraphDatabaseHandler,
                 max_iterations=5,
                 max_iterations_cypher=5,
                 language: str = 'python',
                 message_callback=None):
        super().__init__(
            llm=llm,
            prompt_path=prompt_path,
            schema_path=schema_path,
            task_id=task_id,
            graph_db=graph_db,
            max_iterations=max_iterations,
            max_iterations_cypher=max_iterations_cypher,
            language=language,
            message_callback=message_callback)

        self.cypher_queries_buggy_template = load_prompt_template(
            prompt_path, 'start_prompt_cypher_buggy_loc.txt')

    def set_action_type_and_message(self):
        pass

    def _run(self, user_query: str, file_path: str = '', **kwargs) -> str:

        self.chat_history = []

        primary_user_prompt = self.primary_user_prompt_template.substitute()

        if file_path:
            file_path = f'`file_path`: `{file_path}`\n'

        user_query_issue = f'<issue>\n{file_path}{user_query}\n<\\issue>\n'

        messages = [
            {
                'role': 'system',
                'content': self.system_prompts
            },
            {
                'role': 'user',
                'content': user_query_issue
            },
            {
                'role': 'user',
                'content': primary_user_prompt
            },
        ]

        self.chat_history.append(('system', self.system_prompts))

        for iter in range(self.max_iterations):
            response_text = self.llm_call(messages)
            messages.append({'role': 'assistant', 'content': response_text})

            extracted_analysis, _ = extract_text_between_markers(
                response_text, '[start_of_analysis]', '[end_of_analysis]')
            extracted_code_search, _ = extract_text_between_markers(
                response_text, '[start_of_code_search]',
                '[end_of_code_search]')
            extracted_bug_location, _ = extract_text_between_markers(
                response_text, '[start_of_bug_locations]',
                '[end_of_bug_locations]')

            self.update_agent_message(
                response_to_msg(extracted_analysis, extracted_code_search,
                                extracted_bug_location))

            if not extracted_code_search and not extracted_bug_location:
                msg = (
                    'The text between the markers [start_of_code_search] and [end_of_code_search], '
                    'as well as the text between the markers [start_of_bug_locations] '
                    'and [end_of_bug_locations], is empty.')
                messages.append({'role': 'user', 'content': msg})
                continue
            elif extracted_code_search:
                cypher_queries = self.cypher_queries_template.substitute(
                    text_queries=extracted_code_search)
                user_response = self.cypher_agent.run(
                    cypher_queries, retries=self.max_iterations_cypher)

                if not user_response:
                    msg = (
                        'Cypher Code Assistant encountered issues while processing Cypher queries. '
                        'Please try writing simpler and clearer text queries, and ensure '
                        'that the corresponding parameters are correct.')
                    messages.append({'role': 'user', 'content': msg})
                else:
                    messages.append({'role': 'user', 'content': user_response})
                    self.update_user_message(user_response)

            elif extracted_bug_location:
                cypher_queries = self.cypher_queries_buggy_template.substitute(
                    text_queries=extracted_code_search)
                user_response = self.cypher_agent.run(
                    cypher_queries, retries=self.max_iterations_cypher)

                if not user_response:
                    msg = (
                        'Cypher Code Assistant encountered issues while processing Cypher queries. '
                        'Please try writing simpler and clearer text queries, and ensure that the '
                        'corresponding parameters are correct.')
                    messages.append({'role': 'user', 'content': msg})
                    continue

                collated_tool_response = f'Here is the code in buggy locations:\n\n{user_response}'

                if 'Node' not in user_response:
                    collated_tool_response += (
                        '\n\nIt seems that buggy locations are missing. '
                        'Please try again.')
                    messages.append({
                        'role': 'user',
                        'content': collated_tool_response
                    })
                    continue

                messages.append({
                    'role': 'user',
                    'content': collated_tool_response
                })
                self.update_user_message(collated_tool_response)
                break

            msg = "Let's analyze collected context first."
            messages.append({'role': 'user', 'content': msg})
            self.update_user_message(msg)

            response_text = self.llm_call(messages)
            messages.append({'role': 'assistant', 'content': response_text})
            self.update_agent_message(response_text)

            if iter < self.max_iterations - 1:
                msg = (
                    'Summarize your analysis first, and tell whether the current context is sufficient, '
                    'write your summarization here: \n#### Concise Summarization:\n...\n'
                    "\nThen if it's sufficient, please continue answering in the following format:\n"
                    f'{CODE_SEARCH_FORMAT}'
                    "\nIf it's not sufficient, please continue answering in the following format:\n"
                    f'{BUG_LOCALIZATION_FORMAT}'
                    '\n\nNOTE:'
                    '\n- If you have already identified the bug locations, do not write any search text queries.'
                    "\n- If you haven't yet reviewed the specific code related to the bug or "
                    'pinpointed the exact location of the bug '
                    '(such as which module, class, method, or function), it is not recommended '
                    'to provide answers that directly specify '
                    'the bug locations.')
                messages.append({'role': 'user', 'content': msg})

        generate_queries = self.generate_queries_template.substitute()
        messages.append({'role': 'user', 'content': generate_queries})

        answer = self.generate(messages)
        answer = markdown_answer(answer)
        self.update_agent_message(answer)
        return answer

    def generate(self, messages):
        messages = deepcopy(messages)
        messages = replace_system_prompt(messages, SYSTEM_PROMPT)
        response_text = self.llm_call(messages)
        return response_text
