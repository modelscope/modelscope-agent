import os

import streamlit as st


def sidebar():
    with st.sidebar:
        # global page
        st.image('apps/codexgraph_agent/codexgraph.png', width=100)
        st.title('CodexGraph Agent')
        st.markdown(
            '## How to use 💡\n'
            '1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below 🔑\n'
            '2. Upload a Code Repo 📄\n'
            '3. Ask a question about the Code Repo 💬\n')
        api_key_input = st.text_input(
            'OpenAI API Key',
            type='password',
            placeholder='Paste your OpenAI API key here (sk-...)',
            help=
            'You can get your API key from https://platform.openai.com/account/api-keys.',  # noqa: E501
            value=os.environ.get('OPENAI_API_KEY', None)
            or st.session_state.get('OPENAI_API_KEY', ''),
        )

        st.session_state['OPENAI_API_KEY'] = api_key_input
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
            """)

        st.markdown('---')
        st.markdown(
            '[![Open in GitHub](https://img.shields.io/badge/Open%20in-GitHub-blue.svg)]'
            '(https://codespaces.new/streamlit/llm-examples?quickstart=1)')
