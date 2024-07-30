import streamlit as st
from apps.codexgraph_agent.pages.components.states import get_json_files

# reload_button = None
# history_list = []


def update_config(key, value):
    st.session_state.shared['setting'][key] = value
    # save_config(st.session_state.shared)


def setting_neo4j(page_name):
    with st.container():
        st.subheader('Neo4j Settings')
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 1, 1])
        with col1:
            st.session_state.shared['setting']['neo4j'][
                'user'] = st.text_input(
                    'User',
                    value=st.session_state.shared['setting']['neo4j']['user'])

        with col2:
            st.session_state.shared['setting']['neo4j']['url'] = st.text_input(
                'URI',
                value=st.session_state.shared['setting']['neo4j']['url'])

        with col3:
            st.session_state.shared['setting']['neo4j'][
                'password'] = st.text_input(
                    'Password',
                    type='password',
                    value=st.session_state.shared['setting']['neo4j']
                    ['password'])

        with col4:
            st.session_state.shared['setting']['neo4j'][
                'database_name'] = st.text_input(
                    'Database Name',
                    value=st.session_state.shared['setting']['neo4j']
                    ['database_name'])
        with col5:
            st.write('\n\n')
            with st.form(key='test_connection', border=False):
                # 创建按钮
                st.session_state[page_name][
                    'test_connect_button'] = st.form_submit_button(
                        label='Test Connection', use_container_width=True)

    st.session_state[page_name]['test_connect_place'] = st.empty()


def setting_repo(page_name):
    with st.container():
        st.subheader('Code Repo Settings')
        col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])

        with col1:
            # Language selectbox
            st.session_state.shared['setting']['language'] = st.selectbox(
                'Language', ['Python'])

        with col2:
            repo_path = st.text_input(
                'Repo Path',
                placeholder='Enter repo path here',
                value=st.session_state.shared['setting']['repo_path'],
                key='repo_path_input')
            if repo_path != st.session_state.shared['setting']['repo_path']:
                update_config('repo_path', repo_path)
        with col3:
            project_options = st.session_state.shared['setting'][
                'project_list']
            project_id_cur = st.session_state.shared['setting']['project_id']
            if project_id_cur in project_options:
                cur_index = project_options.index(
                    st.session_state.shared['setting']['project_id'])
            else:
                cur_index = 0
            project_id = st.selectbox('Suggested Project ID', project_options,
                                      cur_index)

            if project_id != st.session_state.shared['setting']['project_id']:
                update_config('project_id', project_id)

        with col4:
            st.session_state.shared['setting']['project_id'] = st.text_input(
                'Enter Project ID',
                placeholder='Project id',
                value=st.session_state.shared['setting']['project_id'],
                key='repo_project_id')

        with col5:
            st.write('\n\n')
            with st.form(key='build', border=False):
                # 创建按钮
                st.session_state[page_name][
                    'build_button'] = st.form_submit_button(
                        label='Build', use_container_width=True)

    st.session_state[page_name]['build_place'] = st.empty()


def setting(page_name, path='CG_conversation'):
    # global history_list

    st.session_state[page_name]['setting']['history_list'] = get_json_files(
        path)
    with st.expander('Settings'):
        # Add your settings controls here, such as sliders, select boxes, etc.
        setting_neo4j(page_name)

        setting_repo(page_name)

        with st.container():
            st.subheader('Agent Settings')
            col1, col2 = st.columns([2, 1])
            with col1:
                # Max iterations slider
                max_iterations = st.slider(
                    'Max iterations', 0, 10,
                    st.session_state.shared['setting']['max_iterations'])

            with col2:
                # Model selectbox
                options = ['deepseek-coder', 'gpt-4o']
                llm_model_name = st.selectbox(
                    'Model', options,
                    options.index(
                        st.session_state.shared['setting']['llm_model_name']))
                if llm_model_name != st.session_state.shared['setting'][
                        'llm_model_name']:
                    update_config('llm_model_name', llm_model_name)

        file_col1, file_col2 = st.columns([2, 1])
        with file_col1:
            if st.session_state[page_name]['setting']['history_list']:
                st.session_state[page_name]['setting'][
                    'history_path'] = st.selectbox(
                        'Select History File',
                        options=st.session_state[page_name]['setting']
                        ['history_list'],
                        key='History file')
            else:
                st.write('No JSON files found in the selected directory.')

        with file_col2:
            st.write('\n\n')
            with st.form(key='history_form', border=False):
                # 创建按钮
                st.session_state[page_name][
                    'reload_button'] = st.form_submit_button(
                        label='Reload History')

        if max_iterations != st.session_state.shared['setting'][
                'max_iterations']:
            update_config('max_iterations', max_iterations)


def repo_db_test(page_name):
    if st.session_state[page_name]['build_button']:
        st.session_state[page_name]['build_place'].success(
            f"Success connect to Neo4j: {st.session_state.shared['setting']['neo4j']['url']}"
        )

    if st.session_state[page_name]['test_connect_button']:
        st.session_state[page_name]['test_connect_place'].success(
            f"File path set to: {st.session_state.shared['setting']['repo_path']}"
        )
