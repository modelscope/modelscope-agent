import os
import sys
import traceback

import json
import requests
import yaml

access_key_id = os.getenv("access_key_id").strip()
access_key_secret = os.getenv("access_key_secret").strip()


class AppConfig(object):
    ALREADY_CONVERTED_MARK = "<!-- ALREADY CONVERTED BY PARSER. -->"

    MODELS = {}
    ROLES = {}
    SHOW_CLASSIC = False
    SEARCH_INSTRUCTION = {}
    INSTALLED_PLUGINS = ['search_engine']
    QUERY_CLASSIFY_SERVICE_URL = ''
    REWRITE_SERVICE_URL = ''
    LEARN_TO_SEARCH_QUERY_URL = ''
    Learn2Search_URL = ''
    Learn2Search_TOKEN = ''
    OSS_ID, OSS_KEY, OSS_REGION_BUCKET = '', '', []
    OSS_BUCKET = None
    ES_USER, ES_PASS = '', ''
    ES_HOST = ''
    PLUGIN_INSTRUCTION = ''

    ROLE_PERSONALITY = {'默认': ''}
    MODELS_CONFIG = {}
    RETRIEVAL_CONFIG = {}

    instruction_search_part_max_length = 1500
    instruction_plugin_part_max_length = 1500
    instruction_max_length = 2800

    @classmethod
    def refresh(cls):
        try:
            print('| refresh config..')
            cls.OSS_ID = access_key_id  # config.get('OSS_ID')
            cls.OSS_KEY = access_key_secret  # config.get('OSS_KEY')
            # cls.OSS_REGION_BUCKET = [['cn-hangzhou', 'pretrain-lm'],
            #                          ['cn-zhangjiakou', 'xdp-expriment'],
            #                          [
            #                              'cn-beijing-internal',
            #                              'xdp-expriment-beijing'
            #                          ]]
        except Exception as e:
            traceback.print_exc()
            print(f'| using default config, exception: {e}')


AppConfig.refresh()
