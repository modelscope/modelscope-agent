import os

import streamlit as st


def sidebar():
    with st.sidebar:
        # global page
        st.image('apps/codexgraph_agent/codexgraph.png', width=100)
        st.title('CodexGraph Agent')
        st.markdown(
            '## How to use 💡\n'
            '1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) or [Deepseek API key](https://platform.deepseek.com/) below 🔑\n'  # noqa: E501
            '2. Upload an absolute path of local Code Repo 📄\n'
            '3. Choice a topic on top and ask a related question about the Code Repo 💬\n'
        )
        api_key_input = st.text_input(
            'OpenAI/Deepseek API Key',
            type='password',
            placeholder=
            'Paste and Enter your OpenAI/Deepseek API key here (sk-...)',
            help=
            'You can get your API key from https://platform.openai.com/account/api-keys or https://platform.deepseek.com',  # noqa: E501
            value=os.environ.get('OPENAI_API_KEY', None)
            or st.session_state.get('OPENAI_API_KEY', ''),
        )

        st.session_state['OPENAI_API_KEY'] = api_key_input
        os.environ['OPENAI_API_KEY'] = api_key_input
        # setting()

        st.markdown('---')
        st.markdown('# About')
        st.markdown("""
            **CodexGraph** Agent is an advanced multi-tasking agent that integrates
            a language model (LM) agent with a code graph database interface. By utilizing
            the structural characteristics of graph databases and the versatility of the Cypher
            query language, CodexGraph enables the LM agent to formulate and execute
            multi-step queries. This capability allows for precise context retrieval and code
            navigation that is aware of the code's structure.

            Currently, only gpt4o and deepseek-coder could generate reasonable results with limited tokens.
            """)

        st.markdown('---')
        st.markdown(
            '[![Powered by Modelscope-Agent](https://img.shields.io/badge/Powered_by-Modelscope_Agent-blue.svg)]'
            '(https://github.com/modelscope/modelscope-agent)')
