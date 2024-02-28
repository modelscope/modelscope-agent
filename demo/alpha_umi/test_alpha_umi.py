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

# llm_configs = {
#     'planner_llm_config': {'model': '/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-planner-7b', 'model_server': 'openai', 'api_base': 'http://localhost:8090/v1', 'is_chat': False},
#     'caller_llm_config': {'model': '/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-caller-7b', 'model_server': 'openai', 'api_base': 'http://localhost:8091/v1', 'is_chat': False},
#     'summarizer_llm_config': {'model': '/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-summarizer-7b', 'model_server': 'openai', 'api_base': 'http://localhost:8092/v1', 'is_chat': False},
# }


def deploy_model():
    # model_paths = [
    #     "/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-planner-7b",
    #     "/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-caller-7b",
    #     "/nas-wulanchabu/shenweizhou.swz/models/alpha-umi-local/online/alpha-umi-summarizer-7b"
    # ]
    model_paths = [
        "iic/alpha-umi-planner-7b", "iic/alpha-umi-caller-7b",
        "iic/alpha-umi-summarizer-7b"
    ]
    # os.system("VLLM_USE_MODELSCOPE=True")
    deploy_planner_cmd = f"CUDA_VISIBLE_DEVICES=0 VLLM_USE_MODELSCOPE=True python -m vllm.entrypoints.openai.api_server --model={model_paths[0]} --revision=v1.0.0 --trust-remote-code --port 8090 --dtype float16 --gpu-memory-utilization 0.9 > planner.log &"
    # deploy_caller_cmd = "python -m vllm.entrypoints.openai.api_server --model=iic/alpha-umi-caller-7b --revision=v1.0.0 --trust-remote-code --port 8091 --dtype float16 --gpu-memory-utilization 0.3 > caller.log &"
    # deploy_summarizer_cmd = "python -m vllm.entrypoints.openai.api_server --model=iic/alpha-umi-summarizer-7b --revision=v1.0.0 --trust-remote-code --port 8092 --dtype float16 --gpu-memory-utilization 0.3 > summarizer.log &"
    os.system(deploy_planner_cmd)
    # os.system(deploy_caller_cmd)
    # os.system(deploy_summarizer_cmd)


# def test_single_deploy(model_id: str, api_base: str):
#     print(f'Testing model api: {model_id}, url({api_base})')
#     openai_api_key = "EMPTY"
#     openai_api_base = api_base.strip()
#     client = OpenAI(
#         api_key=openai_api_key,
#         base_url=openai_api_base,
#     )
#     prompt = 'Introduce yourself.'
#     completion = client.completions.create(model=model_id,
#                                            max_tokens=2000,
#                                            prompt=prompt)
#     assert isinstance(completion.choices[0].text, str)
#     print(f'{model_id} is ready.')
#     print(completion)

# def test_deploys():
#     for _, cfg in llm_configs.items():
#         test_single_deploy(cfg['model'], cfg['api_base'])
#         # break


def test_alpha_umi():
    function_list = ["translate_for_google_translate"]

    bot = AlphaUmi(
        function_list=function_list,
        llm_planner=llm_configs['planner_llm_config'],
        llm_caller=llm_configs['caller_llm_config'],
        llm_summarizer=llm_configs['summarizer_llm_config'],
    )

    response = bot.run('translate "你早上吃饭了吗" into japanese and english')

    text = ''
    for chunk in response:
        print(chunk)


if __name__ == '__main__':
    # 需先在本地部署3个7b模型:调用deploy_model()即可. 需要A100 * 1, 需要手动释放
    # deploy_model()

    # test_deploys()
    # print('Model APIs are ready.')
    # print('-------------------------')
    # test_deploys()
    test_alpha_umi()
