import os

import json
import streamlit as st

CONFIG_FILE = 'apps/codexgraph_agent/setting.json'


def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


# Load configuration
config = load_config()


def initialize_page_state(page_name):
    if 'shared' not in st.session_state:
        st.session_state.shared = config

    if page_name not in st.session_state:
        st.session_state[page_name] = {
            'conversation': [],
            'conversation_history': [],
            'chat': [],
            'final_result': '',
            'input_text': '',
            'input_file_path': '',
            'error_place': None,
            'reload_button': None,
            'test_connect_button': None,
            'build_button': None,
            'test_connect_place': None,
            'build_place': None,
            'progress_bar': None,
            'conversation_container': None,
            'conversation_container_chat': None,
            'setting': {
                'history_path': '',
                'history_list': [],
            },
        }


def get_json_files(path):
    try:
        return [
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith('.json')
        ]
    except FileNotFoundError:
        return []
