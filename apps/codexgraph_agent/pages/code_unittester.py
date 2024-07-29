import os

import streamlit as st
from apps.codexgraph_agent.pages.components.page import PageBase, get_llm_config
from modelscope_agent.agents.codexgraph_agent import CodexGraphAgentUnitTester


class CodeUnittesterPage(PageBase):
    def __init__(self):
        super().__init__(task_name='code_unittester',
                         page_title="🔍 Code Unit Tester",
                         output_path='logs\CU_conversation',
                         input_title="Code needing unittest",
                         default_input_text="Please input the code for which you need unit tests. Include any specific scenarios or edge cases you want to test.")
        self.agent = self.get_agent()

    def get_agent(self):
        graph_db = self.get_graph_db(st.session_state.shared['setting']['project_id'])

        if not graph_db:
            return None

        llm_config = get_llm_config(st.session_state.shared['setting']['llm_model_name'])

        max_iterations = int(st.session_state.shared['setting']['max_iterations'])

        prompt_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'code_unittester')
        schema_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'graph_database')

        try:
            agent = CodexGraphAgentUnitTester(llm=llm_config,
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
    page = CodeUnittesterPage()
    page.main()

if __name__ == '__main__':
    page = CodeUnittesterPage()
    page.main()
