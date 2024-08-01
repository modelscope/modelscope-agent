import os
import sys

import streamlit as st
from apps.codexgraph_agent.pages.components.page import PageBase
from apps.codexgraph_agent.pages.components.setting import (setting_neo4j,
                                                            setting_repo)
from apps.codexgraph_agent.pages.components.sidebar import sidebar
from apps.codexgraph_agent.pages.components.states import (
    initialize_page_state, save_config)

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)


class Help(PageBase):

    def __init__(self):
        self.page_name = 'Help'

    def get_agent(self):
        pass

    def main(self):
        initialize_page_state(self.page_name)
        st.set_page_config(layout='wide')
        sidebar()
        st.title('Help')
        st.markdown("""
# 1. Connect to Neo4j\n

- Download and install [Neo4j Desktop](https://neo4j.com/download/)
- Set the password for the default user `neo4j`
- Create a new project and install a new database with database name `codexgraph`
- Get bolt port as url from the database settings, typically it is `bolt://localhost:7687`
- Use the `neo4j`, `bolt://localhost:7687`, `<YOUR-PASSWORD>` and `codexgraph` as input below
        """)
        setting_neo4j(self.page_name)
        st.markdown("""
# 2. Build a code repo
## 2.1 Build the graph database environment:

Create a separate `python (python<=3.9)` environment and install the required dependencies:
```bash
conda create --name index_build python=3.9

conda activate index_build

pip install -r build_requirements.txt
```
## 2.2. Get build `python env` (python<=3.9):
```bash
where python
```
On Unix-like systems (Linux, macOS):
```bash
which python
```
## 2.3 Setting the `python env` (python<=3.9) and `build index path`:
        """)
        env_path = st.text_input(
            'Python env path',
            placeholder=
            r'/Users/<YourUserName>/opt/miniconda3/envs/index_build39/bin/python',
            value=st.session_state.shared['setting']['env_path_dict']
            ['env_path'],
            key='env_path_input')
        if env_path != st.session_state.shared['setting']['env_path_dict'][
                'env_path']:
            st.session_state.shared['setting']['env_path_dict'][
                'env_path'] = env_path
            save_config(st.session_state.shared)
            st.success(f'set env path: {env_path}')

        if st.session_state.shared['setting']['env_path_dict'][
                'working_directory'] == '':
            working_directory_path = os.path.join(project_root,
                                                  'modelscope_agent',
                                                  'environment',
                                                  'graph_database', 'indexer')
            st.session_state.shared['setting']['env_path_dict'][
                'working_directory'] = working_directory_path
        working_directory = st.text_input(
            'Build index Path',
            placeholder='Enter path here',
            value=st.session_state.shared['setting']['env_path_dict']
            ['working_directory'],
            key='index_path_input')
        if working_directory != st.session_state.shared['setting'][
                'env_path_dict']['working_directory']:
            st.session_state.shared['setting']['env_path_dict'][
                'working_directory'] = os.path.join(working_directory)
            save_config(st.session_state.shared)
            st.success(f'set indexing directory: {working_directory}')

        st.markdown('# 3. Build\n')

        setting_repo(self.page_name)

        self.repo_db_test()


if __name__ == '__main__':
    page = Help()
    page.main()
