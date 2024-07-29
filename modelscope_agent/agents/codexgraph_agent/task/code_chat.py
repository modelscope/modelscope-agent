from copy import deepcopy

from modelscope_agent.agents.codexgraph_agent.cypher_agent import \
    CODE_SEARCH_FORMAT
from modelscope_agent.agents.codexgraph_agent.task.code_general import \
    CodexGraphAgentGeneral
from modelscope_agent.agents.codexgraph_agent.utils.code_utils import \
    extract_text_between_markers
from modelscope_agent.agents.codexgraph_agent.utils.prompt_utils import \
    replace_system_prompt

SYSTEM_PROMPT = (
    'You are a software developer maintaining a large project.\n'
    'Your task is to answer various questions related to the code project raised by users, '
    'which may include asking questions, '
    'fixing bugs, adding function comments, adding new requirements, etc.\n'
    'The issue contains a description marked between <issue> and </issue>.\n')

ANSWER_FORMAT = """[start_of_answer]
### Answer
- Analysis: <analysis of this question>
- Conclusion: <conclusion of this question>
- Source code reference: <reference of Source code>
[end_of_answer]
"""


def response_to_msg(extraced_analysis, extraced_code_search, answer_question):
    msg = ''
    if extraced_analysis:
        msg += f'## analysis\n\n{extraced_analysis}\n\n'
    if extraced_code_search:
        msg += f'## code_search\n\n{extraced_code_search}\n\n'
    if answer_question:
        msg += f'## answer_question\n\n{answer_question}\n\n'
    return msg


def markdown_answer(answer):
    # <file>...</file>
    # <original>...</original>
    # <patched>...</patched>
    answer = answer.replace('```', '')
    replace_dict = {
        '<analysis>': '\n## analysis: \n',
        '</analysis>': '\n',
        '<answer>': '\n## answer: \n',
        '</answer>': '\n',
        '<reference>': '\n## reference: \n',
        '</reference>': '\n',
    }
    for key, value in replace_dict.items():
        answer = answer.replace(key, value)
    return answer


class CodexGraphAgentChat(CodexGraphAgentGeneral):

    def set_action_type_and_message(self):
        pass

    def _run(self, user_query: str, file_path: str = '', **kwargs) -> str:

        self.chat_history = []

        primary_user_prompt = self.primary_user_prompt_template.template

        user_query_issue = f'<questions>\n{user_query}\n<\\questions>\n'

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
            answer_question, _ = extract_text_between_markers(
                response_text, '[start_of_answer]', '[end_of_answer]')

            self.update_agent_message(
                response_to_msg(extracted_analysis, extracted_code_search,
                                answer_question))

            if not extracted_code_search and not answer_question:
                msg = (
                    'The text between the markers [start_of_code_search] and [end_of_code_search], '
                    'as well as the text between the markers [start_of_answer] and [end_of_answer], is empty.'
                )
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
                        'Please try writing simpler and clearer text queries, and ensure that '
                        'the corresponding parameters are correct.')
                    messages.append({'role': 'user', 'content': msg})
                else:
                    messages.append({'role': 'user', 'content': user_response})
                    self.update_user_message(user_response)

            elif answer_question:
                break

            if iter < self.max_iterations - 1:
                msg = (
                    'Summarize your analysis first, and tell whether the current context is sufficient, '
                    'write your summarization here: \n'
                    '#### Concise Summarization:\n...\n'
                    "\nThen if it's sufficient, please continue answering in the following format:"
                    f'{CODE_SEARCH_FORMAT}'
                    "\nif it's not sufficient, please continue answering in the following format:"
                    f'{ANSWER_FORMAT}'
                    '\n\nNOTE:'
                    "\n- If you have already answered the user's question, do not write any search text queries."
                )
                messages.append({'role': 'user', 'content': msg})

        generate_queries = self.generate_queries_template.substitute(
            message='You are ready to do answer question.',
            user_query=user_query)
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
