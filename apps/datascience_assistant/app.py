import os
import sys

import streamlit as st

os.environ['DASHSCOPE_API_KEY'] = 'YOUR_API_KEY'


def setup_project_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # noqa
    project_root_path = os.path.abspath(os.path.join(current_dir,
                                                     '../../'))  # noqa
    sys.path.append(project_root_path)  # noqa


if __name__ == '__main__':
    setup_project_paths()
    from modelscope_agent.agents.data_science_assistant import \
        DataScienceAssistant  # noqa
    from modelscope_agent.tools.metagpt_tools.tool_recommend import \
        TypeMatchToolRecommender  # noqa
    st.title('Data Science Assistant')
    st.write(
        'This is a data science assistant that can help you with your data science tasks.'
    )
    st.write(
        'Please input your request and upload files then click the submit button.'
    )

    files = st.file_uploader(
        'Please upload files that you need. ', accept_multiple_files=True)
    last_file_name = ''
    user_request = st.text_area('User Request')
    if st.button('submit'):
        llm_config = {
            'model': 'qwen2-72b-instruct',
            'model_server': 'dashscope',
        }
        data_science_assistant = DataScienceAssistant(
            llm=llm_config,
            tool_recommender=TypeMatchToolRecommender(tools=['<all>']))
        for file in files:
            with open(file.name, 'wb') as f:
                f.write(file.getbuffer())
        data_science_assistant.run(user_request=user_request, streamlit=True)
