import os

import streamlit as st
from modelscope_agent.agents.data_science_assistant import DataScienceAssistant
from modelscope_agent.tools.metagpt_tools.tool_recommend import \
    TypeMatchToolRecommender

llm_config = {
    'model': 'qwen2-72b-instruct',
    'model_server': 'dashscope',
}
os.environ['DASHSCOPE_API_KEY'] = input(
    'Please input your dashscope api key: ')
data_science_assistant = DataScienceAssistant(
    llm=llm_config, tool_recommender=TypeMatchToolRecommender(tools=['<all>']))
st.title('Data Science Assistant')
st.write(
    'This is a data science assistant that can help you with your data science tasks.'
)
st.write('Please input your request below and click the submit button.')
user_request = st.text_input('User Request')
if st.button('submit'):
    data_science_assistant.run(user_request=user_request, streamlit=True)
