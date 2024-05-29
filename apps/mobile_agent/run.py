import argparse
import os

from modelscope_agent.agents.mobile_agent_v2 import MobileAgentV2
from modelscope_agent.environment import ADBEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--adb_path', type=str, default='./adb/adb')
parser.add_argument(
    '--openai_api_key', type=str, default=os.getenv('OPENAI_API_KEY'))
parser.add_argument(
    '--dashscope_api_key', type=str, default=os.getenv('DASHSCOPE_API_KEY'))
parser.add_argument('--groundingdino_dir', type=str, default='./groundingdino')
parser.add_argument(
    '--instruction', type=str, default="Tell me today's weathers")

args = parser.parse_args()
print(args)
# adb_path = "../../../../platform-tools/adb" # Your adb path
adb_path = args.adb_path

groundingdino_dir = [
    os.path.join(args.groundingdino_dir, 'config/GroundingDINO_SwinT_OGC.py'),
    os.path.join(args.groundingdino_dir, 'groundingdino_swint_ogc.pth')
]

# instruction = "Tell me today's weathers" # Your instruction

os.environ['OPENAI_API_KEY'] = args.openai_api_key
os.environ['DASHSCOPE_API_KEY'] = args.dashscope_api_key

instruction = args.instruction

llm_config = {
    'model': 'gpt-4o',
    'model_server': 'openai_proxy',
}

env = ADBEnvironment(adb_path, groundingdino_dir)

agent = MobileAgentV2(
    env=env,
    llm_decision=llm_config,
    llm_planner=llm_config,
    llm_reflect=llm_config)

agent.run(instruction)
