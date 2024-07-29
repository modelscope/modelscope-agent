import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import streamlit as st
from datetime import datetime
from apps.codexgraph_agent.components.page import PageBase, get_llm_config
from modelscope_agent.agents.codexgraph_agent import CodexGraphAgentChat


class CodeChatPage(PageBase):
    def __init__(self):
        super().__init__(task_name='code_chat',
                         page_title="💬 Code Chat",
                         output_path='logs\CCH_conversation',
                         input_title="",
                         default_input_text="")
        self.agent = self.get_agent()
        self.chat_start = datetime.now().strftime("%d%H%M")

    def get_agent(self):
        graph_db = self.get_graph_db(st.session_state.shared['setting']['project_id'])

        if not graph_db:
            return None

        llm_config = get_llm_config(st.session_state.shared['setting']['llm_model_name'])

        max_iterations = int(st.session_state.shared['setting']['max_iterations'])

        prompt_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'code_chat')
        schema_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'graph_database')

        try:
            agent = CodexGraphAgentChat(llm=llm_config,
                                        prompt_path=prompt_path,
                                        schema_path=schema_path,
                                        task_id=st.session_state.shared['setting']['project_id'],
                                        graph_db=graph_db,
                                        max_iterations=max_iterations,
                                        message_callback=self.create_update_message())
        except:
            agent = None
        return agent

    def run_agent(self):
        user_input = st.session_state[self.page_name]['input_text']

        if not self.agent:
            self.agent = self.get_agent()

        if not self.agent:
            self.error('Failed to get agent, please try re-setting.')
            return

        st.session_state[self.page_name]['conversation_history'] = []
        st.session_state[self.page_name]['conversation_history'].append(
            {'message': user_input, 'role': "user", 'avatar': "🧑‍💻"}
        )
        st.session_state[self.page_name]['chat'].append({'message': user_input, 'role': "user", 'avatar': "🧑‍💻"})
        self.update_chat()

        start_time = datetime.now()

        answer = self.agent.run(user_input)
        # answer = agent_test_run(user_input, '', self.update_message)

        end_time = datetime.now()
        execution_time = end_time - start_time
        execution_time_seconds = execution_time.total_seconds()
        self.success(f'execution_time_seconds: {execution_time_seconds}')

        self.display_chat(answer, 'assistant', "🤖")

        st.session_state[self.page_name]['conversation_history'].append(
            {'message': answer, 'role': "assistant", 'avatar': "🤖"}
        )
        st.session_state[self.page_name]['chat'].append({'message': answer, 'role': "assistant", 'avatar': "🤖"})

        timestamp = datetime.now().strftime("%d%H%M")

        with open(os.path.join(self.output_path, f'conversation_history_{timestamp}.json'), 'w') as file:
            json.dump(st.session_state[self.page_name]['conversation_history'], file)
        with open(os.path.join(self.output_path, f'chat_history_{self.chat_start}.json'), 'w') as file:
            json.dump(st.session_state[self.page_name]['chat'], file)

    def body(self):
        col2, col3 = st.columns([2, 2])

        with col2:
            # st.header("Chat")
            st.session_state[self.page_name]['conversation_container_chat'] = st.container()

        with col3:
            st.session_state[self.page_name]['conversation_container'] = st.container()

        if st.session_state[self.page_name]['reload_button']:
            # st.success(f"File path set to: {st.session_state.history_path}")
            self.reload_history_message(st.session_state[self.page_name]['setting']['history_path'])

        if user_input := st.chat_input(placeholder="input any question..."):
            if user_input:
                st.session_state[self.page_name]['input_text'] = user_input
                self.run_agent()

        with st.session_state[self.page_name]['conversation_container_chat']:
            if len(st.session_state[self.page_name]['chat']) == 0:
                with st.chat_message('assistant', avatar="🤖"):
                    st.markdown('How can i help you?')

    def update_chat(self):
        with st.session_state[self.page_name]['conversation_container_chat']:
            for msg in st.session_state[self.page_name]['chat']:
                with st.chat_message(msg['role'], avatar=msg['avatar']):
                    st.markdown(msg['message'])

    def display_chat(self, message, role, avatar=None):
        with st.session_state[self.page_name]['conversation_container_chat']:
            with st.chat_message(role, avatar=avatar):
                st.markdown(message)

    def reload_history_message(self, history_path):
        with open(history_path, 'r') as file:
            history = json.load(file)
        if 'chat' in history_path:
            st.session_state[self.page_name]['chat'] = history
        else:
            st.session_state[self.page_name]['conversation_history'] = history

        self.update_chat()
        for data in st.session_state[self.page_name]['conversation_history']:
            self.show_message(data['message'], data['role'], data['avatar'])

    def show_message(self, message, role, avatar=None):
        with st.session_state[self.page_name]['conversation_container']:
            with st.chat_message(role, avatar=avatar):
                st.markdown(message)


def show():
    page = CodeChatPage()
    page.main()

if __name__ == '__main__':
    page = CodeChatPage()
    page.main()
