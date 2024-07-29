import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
from apps.codexgraph_agent.components.states import initialize_page_state, save_config
from apps.codexgraph_agent.components.setting import setting_neo4j, setting_repo
from apps.codexgraph_agent.components.page import PageBase

class Help(PageBase):
    def __init__(self):
        self.page_name = 'help'
    def get_agent(self):
        pass

    def main(self):
        initialize_page_state(self.page_name)
        st.set_page_config(layout="wide")
        st.title('Help')
        st.markdown('# 1. Connect to Neo4j\n')
        setting_neo4j(self.page_name)
        st.markdown("""
# 2. Build a code repo
## 2.1 Build the graph database environment: 

Create a separate `Python 3.7` environment and install the required dependencies:
```bash
conda create --name index_build python=3.7

conda activate myenv

pip install -r build_requirements.txt
```
## 2.2. Get build `python 3.7 env`:
```bash
where python
```
On Unix-like systems (Linux, macOS):
```bash
which python
```
## 2.3 Setting the `python3.7 env` and `build index path`:
        """)
        env_path = st.text_input("Python env path", placeholder=r"C:\Users\<YourUsername>\Anaconda3\envs\index_build\python.exe",
                                  value=st.session_state.shared['setting']['env_path_dict']['env_path'], key='env_path_input')
        if env_path != st.session_state.shared['setting']['env_path_dict']['env_path']:
            st.session_state.shared['setting']['env_path_dict']['env_path'] = env_path
            save_config(st.session_state.shared)
            st.success(f'set env path: {env_path}')

        if st.session_state.shared['setting']['env_path_dict']['working_directory'] == '':
            st.session_state.shared['setting']['env_path_dict']['working_directory'] = os.path.join(project_root, 'modelscope_agent\environment\graph_database\indexer')
        working_directory = st.text_input("Build index Path", placeholder="Enter path here",
                                  value=st.session_state.shared['setting']['env_path_dict']['working_directory'], key='index_path_input')
        if working_directory != st.session_state.shared['setting']['env_path_dict']['working_directory']:
            st.session_state.shared['setting']['env_path_dict']['working_directory'] = os.path.join(working_directory)
            save_config(st.session_state.shared)
            st.success(f'set indexing directory: {working_directory}')

        st.markdown('# 3. Build\n')

        setting_repo(self.page_name)

        self.repo_db_test()


if __name__ == '__main__':
    page = Help()
    page.main()