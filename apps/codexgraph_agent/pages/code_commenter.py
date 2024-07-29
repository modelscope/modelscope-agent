import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
from apps.codexgraph_agent.components.page import PageBase, get_llm_config
from modelscope_agent.agents.codexgraph_agent import CodexGraphAgentCommenter


class CodeCommenterPage(PageBase):
    def __init__(self):
        super().__init__(task_name='code_commenter',
                         page_title="📝 Code Commenter",
                         output_path='logs\CC_conversation',
                         input_title="Code needing comments",
                         default_input_text="Please input the code that requires comments")
        self.agent = self.get_agent()

    def get_agent(self):
        graph_db = self.get_graph_db(st.session_state.shared['setting']['project_id'])

        if not graph_db:
            return None

        llm_config = get_llm_config(st.session_state.shared['setting']['llm_model_name'])

        max_iterations = int(st.session_state.shared['setting']['max_iterations'])

        prompt_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'code_commenter')
        schema_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'graph_database')

        try:
            agent = CodexGraphAgentCommenter(llm=llm_config,
                                             prompt_path=prompt_path,
                                             schema_path=schema_path,
                                             task_id=st.session_state.shared['setting']['project_id'],
                                             graph_db=graph_db,
                                             max_iterations=max_iterations,
                                             message_callback=self.create_update_message())
        except:
            agent = None
        return agent

def show():
    page = CodeCommenterPage()
    page.main()

if __name__ == '__main__':
    code_commenter_page = CodeCommenterPage()
    code_commenter_page.main()
