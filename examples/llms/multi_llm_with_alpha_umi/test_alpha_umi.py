import os
import time

from modelscope_agent.agents.alpha_umi import AlphaUmi
from openai import OpenAI

llm_configs = {
    'planner_llm_config': {
        'model': 'iic/alpha-umi-planner-7b',
        'model_server': 'openai',
        'api_base': 'http://localhost:8090/v1',
        'is_chat': False
    },
    'caller_llm_config': {
        'model': 'iic/alpha-umi-caller-7b',
        'model_server': 'openai',
        'api_base': 'http://localhost:8091/v1',
        'is_chat': False
    },
    'summarizer_llm_config': {
        'model': 'iic/alpha-umi-summarizer-7b',
        'model_server': 'openai',
        'api_base': 'http://localhost:8092/v1',
        'is_chat': False
    },
}


def test_alpha_umi():
    function_list = [
        'get_data_fact_for_numbers', 'get_math_fact_for_numbers',
        'get_year_fact_for_numbers', 'listquotes_for_current_exchange',
        'exchange_for_current_exchange'
    ]

    bot = AlphaUmi(
        function_list=function_list,
        llm_planner=llm_configs['planner_llm_config'],
        llm_caller=llm_configs['caller_llm_config'],
        llm_summarizer=llm_configs['summarizer_llm_config'],
    )

    response = bot.run('how many CNY can I exchange for 1 US dollar? \
        also, give me a special property about the number of CNY after exchange'
                       )

    for chunk in response:
        print(chunk)


if __name__ == '__main__':

    test_alpha_umi()
