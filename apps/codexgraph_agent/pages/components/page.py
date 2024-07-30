import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import json
import streamlit as st
from apps.codexgraph_agent.pages.components.setting import setting
from apps.codexgraph_agent.pages.components.sidebar import sidebar
from apps.codexgraph_agent.pages.components.states import (
    get_json_files, initialize_page_state, save_config)
from modelscope_agent.environment.graph_database import GraphDatabaseHandler
from modelscope_agent.environment.graph_database.build import \
    build_graph_database


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
        message = f'Processing {i + 1}/{10}...'
        call_back(message, 'assistant', '🤖')

        time.sleep(1)  # Simulate work by sleeping for 1 second
    return 'test'


def update_progress_bar(page_name, progress):
    st.session_state[page_name]['progress_bar'].progress(int(progress * 100))


class PageBase(ABC):
    """
    PageBase serves as an abstract base class for creating a structured and interactive Streamlit web page.
    It initializes the essential settings, manages the interface components, and handles the interaction
    flow for a specific task. The core functionality includes setting up the page layout, managing
    conversation history, handling user inputs, and interacting with an Agent to provide responses.

    Attributes:
        task_name (str): Name of the task associated with the page.
        page_title (str): Title of the page displayed on the interface.
        output_path (str): Path where conversation history is stored.
        input_title (str): Title for the input section of the page.
        default_input_text (str): Placeholder text for the input area.

    Methods:
        main(): Main entry point for setting up the page and handling user interactions.
        body(): Defines the body layout of the page and handles user inputs.
        repo_db_test(): Tests repository and database connectivity.
        reload_history_message(history_path): Reloads the conversation history from a specified file.
        update_message(message, role, avatar): Updates the conversation history with a new message.
        clear_conversation(): Clears the current conversation history.
        agent:
            run_agent(): Executes the AI agent with user-provided input.
            get_agent(): Abstract method to get the AI agent instance, to be implemented by subclasses.
        call_back:
            create_update_progress_bar(): Creates a function to update the progress bar dynamically.
            create_update_message(): Creates a callback function to update messages dynamically.
    """

    def __init__(
            self,
            task_name='code_commenter',
            page_title='📝 Code Commenter',
            output_path='logs/CC_conversation',
            input_title='Code needing comments',
            default_input_text='Please input the code that requires comments'):

        self.agent = None
        self.page_name = task_name
        self.page_title = page_title
        self.output_path = str(Path(output_path))
        self.input_title = input_title
        self.default_input_text = default_input_text

        initialize_page_state(self.page_name)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        self.prompt_path = str(
            Path(st.session_state.shared['setting']['prompt_path']).joinpath(
                task_name))
        self.schema_path = str(
            Path(st.session_state.shared['setting']['prompt_path']).joinpath(
                'graph_database'))

        st.session_state[
            self.page_name]['setting']['history_list'] = get_json_files(
                self.output_path)
        # st.set_page_config(layout="wide")

    def main(self):
        st.set_page_config(layout='wide')

        st.title(self.page_title)

        st.session_state[self.page_name]['conversation_history'] = []

        sidebar()

        setting(self.page_name, self.output_path)

        st.session_state[self.page_name]['error_place'] = st.empty()

        self.repo_db_test()

        openai_api_key = st.session_state.get('OPENAI_API_KEY')

        if not openai_api_key:
            self.warning(
                'Enter your OpenAI API key in the sidebar. You can get a key at'
                ' https://platform.openai.com/account/api-keys.')

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
            st.session_state[
                self.page_name]['conversation_container'] = st.container()

        if st.session_state[self.page_name]['reload_button']:
            st.session_state[self.page_name]['conversation_history'] = []
            # st.success(f"File path set to: {st.session_state.history_path}")
            self.reload_history_message(
                st.session_state[self.page_name]['setting']['history_path'])

        with col1:
            st.header(self.input_title)
            st.session_state[
                self.page_name]['input_file_path'] = st.text_input(
                    'File Path (optional)', placeholder='Enter file path here')
            st.session_state[self.page_name]['input_text'] = st.text_area(
                'Type your question here:',
                height=300,
                placeholder=self.default_input_text,
                label_visibility='collapsed',
                value=st.session_state[self.page_name]['input_text'])
            col1_1, col1_2, col1_3 = st.columns([1, 1, 1])
            with col1_1:
                if st.button('Send'):
                    if st.session_state[self.page_name]['input_text']:

                        if not self.agent:
                            self.agent = self.get_agent()

                        if not self.agent:
                            self.error(
                                'Failed to get agent, please try re-setting.')
                            return

                        st.session_state[
                            self.page_name]['conversation_history'] = []
                        st.session_state[
                            self.page_name]['conversation_history'].append({
                                'message':
                                st.session_state[self.page_name]['input_text'],
                                'role':
                                'user',
                                'avatar':
                                '🧑‍💻'
                            })

                        start_time = datetime.now()
                        answer = self.run_agent()
                        end_time = datetime.now()
                        execution_time = end_time - start_time
                        execution_time_seconds = execution_time.total_seconds()
                        self.success(
                            f'execution_time_seconds: {execution_time_seconds}'
                        )

                        timestamp = datetime.now().strftime('%d%H%M')

                        with open(
                                os.path.join(
                                    self.output_path,
                                    f'conversation_history_{timestamp}.json'),
                                'w') as file:
                            json.dump(
                                st.session_state[self.page_name]
                                ['conversation_history'], file)

                        st.session_state[
                            self.page_name]['final_result'] = answer

            with col1_2:
                if st.session_state[self.page_name]['conversation_history']:
                    if st.button('Clear Conversation'):
                        self.clear_conversation()

            if st.session_state[self.page_name]['final_result']:
                st.header('Final Result')
                st.write(st.session_state[self.page_name]['final_result'])

    def run_agent(self):
        answer = self.agent.run(
            user_query=st.session_state[self.page_name]['input_text'],
            file_path=st.session_state[self.page_name]['input_file_path'])
        return answer

    def update_message(self, message, role, avatar=None):
        # Append the new message to the conversation history
        st.session_state[self.page_name]['conversation_history'].append({
            'message':
            message,
            'role':
            role,
            'avatar':
            avatar
        })

        # Update the conversation container with the new message
        with st.session_state[self.page_name]['conversation_container']:
            with st.chat_message(role, avatar=avatar):
                st.markdown(message)

    def create_update_message(self):
        page_name = self.page_name

        def update_message(message, role, avatar=None):
            # Append the new message to the conversation history
            st.session_state[page_name]['conversation_history'].append({
                'message':
                message,
                'role':
                role,
                'avatar':
                avatar
            })

            # Update the conversation container with the new message
            with st.session_state[page_name]['conversation_container']:
                with st.chat_message(role, avatar=avatar):
                    st.markdown(message)

        return update_message

    def reload_history_message(self, history_path):
        with open(history_path, 'r') as file:
            history = json.load(file)
        for data in history:
            self.update_message(data['message'], data['role'], data['avatar'])

        st.session_state[
            self.page_name]['final_result'] = history[-1]['message']

    def clear_conversation(self):
        """Clear conversation list"""
        st.session_state[self.page_name]['conversation_history'] = []

    def create_update_progress_bar(self):

        def update_progress_bar(progress):
            with st.session_state[self.page_name]['build_place']:
                st.progress(int(progress * 100))

        return update_progress_bar

    def repo_db_test(self):
        page_state = st.session_state[self.page_name]
        setting = st.session_state.shared['setting']
        neo4j_setting = setting['neo4j']
        project_id = st.session_state.shared['setting']['project_id']
        project_list = st.session_state.shared['setting']['project_list']

        if page_state['build_button']:
            if project_id not in project_list:
                project_list.append(project_id)
                save_config(st.session_state.shared)
            if self.build_graph_db():
                page_state['build_place'].success(
                    f"File path set to: {setting['repo_path']}")

        if page_state['test_connect_button']:
            if self.get_graph_db():
                page_state['test_connect_place'].success(
                    f"Successfully connected to Neo4j: {neo4j_setting['url']}")
            else:
                page_state['test_connect_place'].error('Connection error.')

        if not os.path.exists(setting['repo_path']):
            self.warning('Enter a correct repo path.')

        if (not neo4j_setting['url'] or not neo4j_setting['user']
                or not neo4j_setting['password']
                or not neo4j_setting['database_name']):
            self.warning('Please set Neo4j settings.')

    def get_graph_db(self, task_id=''):
        try:
            neo4j_setting = st.session_state.shared['setting']['neo4j']
            graph_db = GraphDatabaseHandler(
                uri=neo4j_setting['url'],
                user=neo4j_setting['user'],
                password=neo4j_setting['password'],
                database_name=neo4j_setting['database_name'],
                task_id=task_id,
                use_lock=True,
            )
        except Exception:
            graph_db = None

        return graph_db

    def build_graph_db(self):
        setting = st.session_state.shared['setting']
        env_path_setting = setting['env_path_dict']
        neo4j_setting = setting['neo4j']
        page_state = st.session_state[self.page_name]

        env_path_dict = {
            'env_path': env_path_setting['env_path'],
            'working_directory': env_path_setting['working_directory'],
            'url': neo4j_setting['url'],
            'user': neo4j_setting['user'],
            'password': neo4j_setting['password'],
            'db_name': neo4j_setting['database_name']
        }

        graph_db = self.get_graph_db(setting['project_id'])
        msg = ''
        if graph_db:
            try:
                msg = build_graph_database(
                    graph_db,
                    setting['repo_path'],
                    task_id=setting['project_id'],
                    is_clear=True,
                    max_workers=None,
                    env_path_dict=env_path_dict,
                    update_progress_bar=self.create_update_progress_bar())
            except Exception as e:
                page_state['build_place'].error(
                    f'An error occurred while building the graph database: {str(e)}'
                )
                graph_db = None
        if msg:
            page_state['build_place'].error(msg)
            return None
        return graph_db

    @abstractmethod
    def get_agent(self):
        pass


if __name__ == '__main__':
    page = PageBase()
    page.main()
