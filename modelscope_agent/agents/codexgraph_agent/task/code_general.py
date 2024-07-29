from abc import ABC, abstractmethod

from modelscope_agent import Agent
from modelscope_agent.agents.codexgraph_agent import CypherAgent
from modelscope_agent.agents.codexgraph_agent.prompt import JSON_PROMPT
from modelscope_agent.agents.codexgraph_agent.utils.code_utils import \
    extract_and_parse_json
from modelscope_agent.agents.codexgraph_agent.utils.prompt_utils import (
    build_system_prompt, load_prompt_template, response_to_msg)
from modelscope_agent.environment.graph_database import GraphDatabaseHandler


class CodexGraphAgentGeneral(Agent, ABC):

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
        super().__init__(llm=llm, name='CodexGraph agent')

        self.message_callback = message_callback

        system_prompts, cypher_system_prompts = build_system_prompt(
            prompt_path, schema_path, language=language)

        self.system_prompts = system_prompts
        self.primary_user_prompt_template = load_prompt_template(
            prompt_path, 'start_prompt_primary.txt', language=language)
        self.generate_queries_template = load_prompt_template(
            prompt_path, 'generate_prompt.txt', language=language)
        self.cypher_queries_template = load_prompt_template(
            prompt_path, 'start_prompt_cypher.txt', language=language)

        self.cypher_agent = CypherAgent(
            llm,
            graph_db=graph_db,
            system_prompts=cypher_system_prompts,
            task_id=task_id)

        self.max_iterations = max_iterations
        self.max_iterations_cypher = max_iterations_cypher
        self.input_token_num = 0
        self.output_token_num = 0
        self.max_tokens = 1024
        self.temperature = 1.0

        self.chat_history = []

        self.action_type = 'ACTIONS'
        self.generate_message = 'You are ready to do generate New Code.'

        self.set_action_type_and_message()

    @abstractmethod
    def set_action_type_and_message(self):
        pass

    def llm_call(self, msg):
        resp = self.llm.chat(
            messages=msg,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False)

        self.chat_history.append(('user', msg[-1]['content']))
        self.chat_history.append(('agent', resp))

        usage_info = self.llm.get_usage()

        self.input_token_num += usage_info.get('prompt_tokens', 0)
        self.output_token_num += usage_info.get('completion_tokens', 0)

        return resp

    def update_agent_message(self, msg):
        if self.message_callback:
            self.message_callback(msg, role='assistant', avatar='🤖')

    def update_user_message(self, msg):
        if self.message_callback:
            self.message_callback(msg, role='user', avatar='🧑‍💻')

    def _run(self, user_query: str, file_path: str = '', **kwargs) -> str:

        if file_path:
            file_path = f'# file path: {file_path}'

        self.chat_history = []
        primary_user_prompt = self.primary_user_prompt_template.substitute(
            file_path=file_path, user_query=user_query)
        messages = [
            {
                'role': 'system',
                'content': self.system_prompts
            },
            {
                'role': 'user',
                'content': primary_user_prompt
            },
        ]
        self.chat_history.append(('system', self.system_prompts))

        generate_msg = self.generate_message

        for iter in range(self.max_iterations):

            if iter == self.max_iterations - 1:
                generate_msg = 'You have exhausted all query opportunities.'

            response_text = self.llm_call(messages)
            messages.append({'role': 'assistant', 'content': response_text})
            parsed_response, error_msg = extract_and_parse_json(response_text)

            if error_msg:
                user_response = (
                    'Something wrong with the JSON format, please rewrite it '
                    'and follow the given format:\n'
                    f'{JSON_PROMPT}')
                messages.append({'role': 'user', 'content': user_response})
                continue

            thought, action, action_input = parsed_response.values()

            self.update_agent_message(
                response_to_msg(thought, action, action_input))

            if action == self.action_type:
                break

            elif action == 'TEXT_QUERIES':
                cypher_queries = self.cypher_queries_template.substitute(
                    text_queries=action_input)
                user_response = self.cypher_agent.run(
                    cypher_queries, retries=self.max_iterations_cypher)
            else:
                user_response = (f'Invalid action, the action should be '
                                 f'`{self.action_type}` or `TEXT_QUERIES`')

            messages.append({'role': 'user', 'content': user_response})
            self.update_user_message(user_response)

        generate_queries = self.generate_queries_template.substitute(
            message=generate_msg, file_path=file_path, user_query=user_query)

        messages.append({'role': 'user', 'content': generate_queries})
        answer = self.llm_call(messages)

        self.update_user_message(generate_queries)
        self.update_agent_message(answer)
        messages.append({'role': 'assistant', 'content': answer})

        return answer

    def get_chat_history(self):
        return self.chat_history


class CodexGraphAgentCommenter(CodexGraphAgentGeneral):

    def set_action_type_and_message(self):
        self.action_type = 'ADD_COMMENTS'
        self.generate_message = 'You are ready to add code comments.'


class CodexGraphAgentGenerator(CodexGraphAgentGeneral):

    def set_action_type_and_message(self):
        self.action_type = 'GENERATE_NEW_CODE'
        self.generate_message = 'You are ready to do generate New Code.'


class CodexGraphAgentUnitTester(CodexGraphAgentGeneral):

    def set_action_type_and_message(self):
        self.action_type = 'GENERATE_UNITTEST'
        self.generate_message = 'You are ready to do generate Professional Unittest.'
