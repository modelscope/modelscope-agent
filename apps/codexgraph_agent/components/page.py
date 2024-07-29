import streamlit as st
import os
import time
import json
from datetime import datetime
from abc import ABC, abstractmethod
from apps.codexgraph_agent.components.setting import setting
from apps.codexgraph_agent.components.sidebar import sidebar
from apps.codexgraph_agent.components.states import initialize_page_state, get_json_files
from modelscope_agent.environment.graph_database import GraphDatabaseHandler, build_graph_database

def get_llm_config(llm_name):

    if llm_name == 'deepseek-coder':

        llm_config = {
            'model': 'deepseek-coder',
            'api_base': 'https://api.deepseek.com',
            'model_server': 'openai'
        }

    elif llm_name == 'gpt-4o':

        llm_config = {
            'model': 'gpt-4o-2024-05-13',
            'api_base': 'https://api.openai.com/v1',
            'model_server': 'openai'
        }

    else:
        return None

    return llm_config

def agent_test_run(user_query, file_path, call_back):
    for i in range(10):
        message = f"Processing {i + 1}/{10}..."
        call_back(message, "assistant", "🤖")

        time.sleep(1)  # Simulate work by sleeping for 1 second
    return 'test'

def update_progress_bar(page_name, progress):
    st.session_state[page_name]['progress_bar'].progress(int(progress * 100))



class PageBase(ABC):
    def __init__(self,
                 task_name='code_commenter',
                 page_title="📝 Code Commenter",
                 output_path='CC_conversation',
                 input_title="Code needing comments",
                 default_input_text="Please input the code that requires comments"):

        self.agent = None
        self.page_name = task_name
        self.page_title = page_title
        self.output_path = output_path
        self.input_title = input_title
        self.default_input_text = default_input_text

        initialize_page_state(self.page_name)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.prompt_path = os.path.join(st.session_state.shared['setting']['prompt_path'], task_name)
        self.schema_path = os.path.join(st.session_state.shared['setting']['prompt_path'], 'graph_database')

        st.session_state[self.page_name]['setting']['history_list'] = get_json_files(self.output_path)
        # st.set_page_config(layout="wide")

    def main(self):
        st.title(self.page_title)

        st.session_state[self.page_name]['conversation_history'] = []

        sidebar()

        setting(self.page_name, self.output_path)

        st.session_state[self.page_name]['error_place'] = st.empty()

        self.repo_db_test()

        openai_api_key = st.session_state.get("OPENAI_API_KEY")

        if not openai_api_key:
            self.warning(
                "Enter your OpenAI API key in the sidebar. You can get a key at"
                " https://platform.openai.com/account/api-keys."
            )

        self.body()

    def error(self, msg):
        if st.session_state[self.page_name]['error_place']:
            st.session_state[self.page_name]['error_place'].error(msg)

    def warning(self, msg):
        if st.session_state[self.page_name]['error_place']:
            st.session_state[self.page_name]['error_place'].warning(msg)

    def success(self, msg):
        if st.session_state[self.page_name]['error_place']:
            st.session_state[self.page_name]['error_place'].success(msg)

    def body(self):

        col1, col2 = st.columns([1, 2])

        with col2:
            # st.header("Conversation")
            st.session_state[self.page_name]['conversation_container'] = st.container()

        if st.session_state[self.page_name]['reload_button']:
            st.session_state[self.page_name]['conversation_history'] = []
            # st.success(f"File path set to: {st.session_state.history_path}")
            self.reload_history_message(st.session_state[self.page_name]['setting']['history_path'])

        with col1:
            st.header(self.input_title)
            st.session_state[self.page_name]['input_file_path'] = st.text_input("File Path (optional)",
                                                                           placeholder="Enter file path here")
            st.session_state[self.page_name]['input_text'] = st.text_area("Type your question here:", height=300,
                                                                     placeholder=self.default_input_text,
                                                                     label_visibility="collapsed",
                                                                     value=st.session_state[self.page_name]['input_text'])
            col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
            with col1_1:
                if st.button("Send"):
                    if st.session_state[self.page_name]['input_text']:

                        if not self.agent:
                            self.agent = self.get_agent()

                        if not self.agent:
                            self.error('Failed to get agent, please try re-setting.')
                            return

                        st.session_state[self.page_name]['conversation_history'] = []
                        st.session_state[self.page_name]['conversation_history'].append(
                            {'message': st.session_state[self.page_name]['input_text'], 'role': "user", 'avatar': "🧑‍💻"}
                        )

                        start_time = datetime.now()
                        answer = self.run_agent()
                        end_time = datetime.now()
                        execution_time = end_time - start_time
                        execution_time_seconds = execution_time.total_seconds()
                        self.success(f'execution_time_seconds: {execution_time_seconds}')

                        timestamp = datetime.now().strftime("%d%H%M")

                        with open(os.path.join(self.output_path, f'conversation_history_{timestamp}.json'), 'w') as file:
                            json.dump(st.session_state[self.page_name]['conversation_history'], file)

                        st.session_state[self.page_name]['final_result'] = answer

            with col1_2:
                if st.session_state[self.page_name]['conversation_history']:
                    if st.button("Clear Conversation"):
                        self.clear_conversation()

            if st.session_state[self.page_name]['final_result']:
                st.header("Final Result")
                st.write(st.session_state[self.page_name]['final_result'])

    def run_agent(self):
        answer = self.agent.run(user_query=st.session_state[self.page_name]['input_text'],
                                file_path=st.session_state[self.page_name]['input_file_path'])
        return answer

    def update_message(self, message, role, avatar=None):
        st.session_state[self.page_name]['conversation_history'].append({'message': message, 'role': role, 'avatar': avatar})
        with st.session_state[self.page_name]['conversation_container']:
            with st.chat_message(role, avatar=avatar):
                st.markdown(message)

    def create_update_message(self):
        page_name = self.page_name
        def update_message(message, role, avatar=None):
            st.session_state[page_name]['conversation_history'].append({'message': message, 'role': role, 'avatar': avatar})
            with st.session_state[page_name]['conversation_container']:
                with st.chat_message(role, avatar=avatar):
                    st.markdown(message)
        return update_message

    def reload_history_message(self, history_path):
        with open(history_path, 'r') as file:
            history = json.load(file)
        for data in history:
            self.update_message(data['message'], data['role'], data['avatar'])

        st.session_state[self.page_name]['final_result'] = history[-1]['message']

    def clear_conversation(self):
        """Clear conversation list"""
        st.session_state[self.page_name]['conversation_history'] = []


    def create_update_progress_bar(self):
        def update_progress_bar(progress):
            with st.session_state[self.page_name]['build_place']:
                st.progress(int(progress * 100))
        return update_progress_bar

    def repo_db_test(self):

        if st.session_state[self.page_name]['build_button']:

            if self.build_graph_db():
                st.session_state[self.page_name]['build_place'].success(
                    f"File path set to: {st.session_state.shared['setting']['repo_path']}")
            else:
                st.session_state[self.page_name]['build_place'].error("something error")

        if st.session_state[self.page_name]['test_connect_button']:
            if self.get_graph_db():
                st.session_state[self.page_name]['test_connect_place'].success(
                    f"Success connect to Neo4j: {st.session_state.shared['setting']['neo4j']['url']}")
            else:
                st.session_state[self.page_name]['test_connect_place'].error("Connect error")

        if not os.path.exists(st.session_state.shared['setting']['repo_path']):
            self.warning('Enter a correct repo path')

        if not st.session_state.shared['setting']['neo4j']['url'] or \
            not st.session_state.shared['setting']['neo4j']['user'] or \
            not st.session_state.shared['setting']['neo4j']['password'] or \
            not st.session_state.shared['setting']['neo4j']['database_name']:
            self.warning('Please setting Neo4j')

    def get_graph_db(self, task_id=''):
        try:
            graph_db = GraphDatabaseHandler(
                uri=st.session_state.shared['setting']['neo4j']['url'],
                user=st.session_state.shared['setting']['neo4j']['user'],
                password=st.session_state.shared['setting']['neo4j']['password'],
                database_name=st.session_state.shared['setting']['neo4j']['database_name'],
                task_id=task_id,
                use_lock=True,
            )
        except:
            graph_db = None

        return graph_db

    def build_graph_db(self):
        env_path_dict = {
            'env_path': st.session_state.shared['setting']['env_path_dict']['env_path'],
            'working_directory': st.session_state.shared['setting']['env_path_dict']['working_directory'],
            'url': st.session_state.shared['setting']['neo4j']['url'],
            'user': st.session_state.shared['setting']['neo4j']['user'],
            'password': st.session_state.shared['setting']['neo4j']['password'],
            'db_name': st.session_state.shared['setting']['neo4j']['database_name']
        }
        graph_db = self.get_graph_db(st.session_state.shared['setting']['project_id'])
        if graph_db:
            try:
                build_graph_database(graph_db, st.session_state.shared['setting']['repo_path'],
                                     task_id=st.session_state.shared['setting']['project_id'],
                                     is_clear=True, max_workers=None,
                                     env_path_dict=env_path_dict,
                                     update_progress_bar=self.create_update_progress_bar())
            except:
                graph_db = None

        return graph_db

    @abstractmethod
    def get_agent(self):
        pass


if __name__ == '__main__':
    page = PageBase()
    page.main()