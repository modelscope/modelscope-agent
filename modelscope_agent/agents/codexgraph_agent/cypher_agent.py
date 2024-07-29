from modelscope_agent import Agent
from modelscope_agent.agents.codexgraph_agent.prompt import (CYPHER_PROMPT,
                                                             JSON_PROMPT)
from modelscope_agent.agents.codexgraph_agent.utils.code_utils import \
    process_string
from modelscope_agent.agents.codexgraph_agent.utils.cypher_utils import (
    add_label_to_nodes, extract_cypher_queries)
from modelscope_agent.environment.graph_database import GraphDatabaseHandler

CODE_SEARCH_FORMAT = """[start_of_code_search]
### Text Query 1
<text_description_of_the_query>

### Text Query 2
<text_description_of_the_query>

...
### Text Query n
<text_description_of_the_query>
[end_of_code_search]
"""


class CypherAgent(Agent):

    def __init__(self,
                 llm,
                 graph_db: GraphDatabaseHandler,
                 system_prompts: str = '',
                 task_id: str = '',
                 message_callback=None):
        super().__init__(llm=llm, name='CodexGraph agent')

        self.graph_db = graph_db
        self.task_id = task_id
        self.message_callback = message_callback

        self.system_prompts = system_prompts

        self.input_token_num = 0
        self.output_token_num = 0
        self.max_tokens = 1024
        self.temperature = 1.0

        self.chat_history = []

    def llm_call(self, msg):
        resp = self.llm.chat(
            messages=msg,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False)
        usage_info = self.llm.get_usage()

        self.chat_history.append(('cypher_user', msg[-1]['content']))
        self.chat_history.append(('cypher_agent', resp))

        self.input_token_num += usage_info.get('prompt_tokens', 0)
        self.output_token_num += usage_info.get('completion_tokens', 0)

        return resp

    def _run(self, cypher_queries: str, retries: int = 5, **kwargs) -> str:
        """
        Executes a series of Cypher queries by interacting with a language model and a graph database.
        """
        cypher_messages = [
            {
                'role': 'system',
                'content': self.system_prompts
            },
            {
                'role': 'user',
                'content': cypher_queries
            },
        ]

        retry = 0
        # Cypher Agent Loop
        user_response = ''

        while True and retry <= retries:
            cypher_response = self.llm_call(cypher_messages)
            cyphers = extract_cypher_queries(cypher_response)
            user_response = ''
            tmp_flag = True
            for idx, cypher in enumerate(cyphers):
                user_response += (
                    f'### Extracted Cypher query {idx}:\n{cypher}\n')
                cypher = add_label_to_nodes(cypher,
                                            '`{}`'.format(self.task_id))
                cypher_response, flag = self.graph_db.execute_query_with_timeout(
                    cypher)

                if cypher_response != 'cypher too complex, out of memory':
                    cypher_response = [
                        process_string(str(record))
                        for record in cypher_response
                    ]
                    if cypher_response:
                        cypher_response = '\n\n'.join(cypher_response)
                    else:
                        cypher_response = 'Cypher query Return None'

                if not flag:
                    tmp_flag = False

                user_response += f'### Response for Cypher query {idx}:\n{str(cypher_response)}\n\n'

            if tmp_flag:
                break

            cypher_messages.append({
                'role': 'assistant',
                'content': cypher_response
            })
            cypher_messages.append({'role': 'user', 'content': user_response})
            cypher_messages.append({
                'role':
                'user',
                'content':
                ('Some Cypher statements may have syntax issues. Please correct them or break down '
                 'complex Cypher queries. '
                 'The answer should still follow the following formats:\n'
                 f'{CYPHER_PROMPT}'),
            })

            retry += 1
            if retry > retries:
                user_response = (
                    'Current text queries are too complicated, please generate code. '
                    'Answer in the following format:\n'
                    f'{JSON_PROMPT}')

        return user_response

    def get_chat_history(self):
        return self.chat_history
