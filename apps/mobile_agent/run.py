import argparse
import os

from modelscope_agent.agents.mobile_agent_v2 import MobileAgentV2
from modelscope_agent.environment.android_adb import ADBEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--adb_path', type=str, default='./adb/adb')
parser.add_argument(
    '--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY'))
parser.add_argument(
    '--dashscope_api_key', type=str, default=os.getenv('DASHSCOPE_API_KEY'))
parser.add_argument(
    '--instruction', type=str, default="Tell me today's weathers")

args = parser.parse_args()

adb_path = args.adb_path

os.environ['OPENAI_API_KEY'] = args.openai_api_key
# used to calling qwen-vl for description of icon during perception
os.environ['DASHSCOPE_API_KEY'] = args.dashscope_api_key

instruction = args.instruction

llm_config = {
    'model': 'gpt-4o',
    'model_server': 'openai',
}

env = ADBEnvironment(adb_path)

agent = MobileAgentV2(
    env=env,
    llm_decision=llm_config,
    llm_planner=llm_config,
    llm_reflect=llm_config)

agent.run(instruction)
