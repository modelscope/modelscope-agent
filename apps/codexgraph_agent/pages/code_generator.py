import os

import streamlit as st
from apps.codexgraph_agent.pages.components.page import (PageBase,
                                                         get_llm_config)
from modelscope_agent.agents.codexgraph_agent import CodexGraphAgentGenerator


class CodeGeneratorPage(PageBase):

    def __init__(self):
        super().__init__(
            task_name='code_generator',
            page_title='🔧 Code Generator',
            output_path='logs\\CG_conversation',
            input_title='New Requirements',
            default_input_text=
            'Please input the requirements or specifications for the new feature or module you need.'
        )

        self.agent = self.get_agent()

    def get_agent(self):
        graph_db = self.get_graph_db(
            st.session_state.shared['setting']['project_id'])

        if not graph_db:
            return None

        llm_config = get_llm_config(
            st.session_state.shared['setting']['llm_model_name'])

        max_iterations = int(
            st.session_state.shared['setting']['max_iterations'])

        prompt_path = os.path.join(
            st.session_state.shared['setting']['prompt_path'],
            'code_generator')
        schema_path = os.path.join(
            st.session_state.shared['setting']['prompt_path'],
            'graph_database')

        try:
            agent = CodexGraphAgentGenerator(
                llm=llm_config,
                prompt_path=prompt_path,
                schema_path=schema_path,
                task_id=st.session_state.shared['setting']['project_id'],
                graph_db=graph_db,
                max_iterations=max_iterations,
                message_callback=self.create_update_message())
        except Exception:
            agent = None
        return agent


def show():
    page = CodeGeneratorPage()
    page.main()


if __name__ == '__main__':
    page = CodeGeneratorPage()
    page.main()
