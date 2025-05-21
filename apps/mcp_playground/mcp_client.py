# flake8: noqa: F401
import os
import re
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List

import json
from exceptiongroup import ExceptionGroup
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from modelscope_agent.tools.mcp.mcp_client import MCPClient


def parse_mcp_config(mcp_config: dict, enabled_mcp_servers: list = None):
    mcp_servers = {}
    for server_name, server in mcp_config.get('mcpServers', {}).items():
        if server.get('type') == 'stdio' or not server.get('url') or (
                enabled_mcp_servers is not None
                and server_name not in enabled_mcp_servers):
            continue
        new_server = {**server}
        # new_server["transport"] = server.get("type", "sse")
        new_server['transport'] = 'sse'
        if hasattr(server, 'type'):
            del new_server['type']
        if server.get('env'):
            env = {'PYTHONUNBUFFERED': '1', 'PATH': os.environ.get('PATH', '')}
            env.update(server['env'])
            new_server['env'] = env
        mcp_servers[server_name] = new_server
    return mcp_servers


@asynccontextmanager
async def get_mcp_client_back(mcp_servers: dict):
    async with MultiServerMCPClient(mcp_servers) as client:
        yield client


@asynccontextmanager
async def get_mcp_client(mcp_servers: Dict[str, Any], api_config: Dict[str,
                                                                       Any]):
    async with MCPClient(mcp_servers, api_config) as client:
        yield client


async def get_mcp_prompts(mcp_config: dict, get_llm: Callable):
    try:
        mcp_servers = parse_mcp_config(mcp_config)
        if len(mcp_servers.keys()) == 0:
            return {}
        llm: BaseChatModel = get_llm()
        async with get_mcp_client_back(mcp_servers) as client:
            mcp_tool_descriptions = {}
            for mcp_name, server_tools in client.server_name_to_tools.items():
                mcp_tool_descriptions[mcp_name] = {}
                for tool in server_tools:
                    mcp_tool_descriptions[mcp_name][
                        tool.name] = tool.description
            prompt = f"""Based on the following MCP service tool descriptions, generate 2-4 example user queries for each service:

Input structure explanation:
- mcp_tool_descriptions is a nested dictionary
- The first level keys are MCP service names (e.g., "service1", "service2")
- The second level contains descriptions of tools available within each service

MCP Service Tool Descriptions: {json.dumps(mcp_tool_descriptions)}

Please provide 2-4 natural and specific example queries in Chinese that effectively demonstrate the capabilities of each service.

The response must be in strict JSON format as shown below, with MCP service names as keys:
```json
{{
    "mcp_name1": ["中文示例1", "中文示例2"],
    "mcp_name2": ["中文示例1", "中文示例2"]
}}
```

Ensure:
1. Each example is specific to the functionality of that particular MCP service
2. Example queries are in natural Chinese expressions
3. Strictly use the top-level MCP service names as JSON keys
4. The returned format must be valid JSON
5. Each service MUST have exactly 2-4 example queries - not fewer than 2 and not more than 4

Return only the JSON object without any additional explanation or text."""
            response = await llm.ainvoke(prompt)
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
            else:
                json_content = content
            raw_examples = json.loads(json_content)

            for mcp_name in mcp_tool_descriptions.keys():
                if mcp_name not in raw_examples:
                    raw_examples[mcp_name] = [
                        f'请使用 {mcp_name} 服务的功能帮我查询信息或解决问题',
                    ]
            return raw_examples
    except ExceptionGroup as eg:
        print('Prompt ExceptionGroup Error:', eg)
        return {
            mcp_name: [
                f'请使用 {mcp_name} 服务的功能帮我查询信息或解决问题',
            ]
            for mcp_name in mcp_servers.keys()
        }
    except Exception as e:
        print('Prompt Error:', e)
        return {
            mcp_name: [
                f'请使用 {mcp_name} 服务的功能帮我查询信息或解决问题',
            ]
            for mcp_name in mcp_servers.keys()
        }


def convert_mcp_name(tool_name: str, mcp_names: dict):
    if not tool_name:
        return tool_name
    separators = tool_name.split('__TOOL__')
    if len(separators) >= 2:
        mcp_name_idx, mcp_tool_name = separators[:2]
    else:
        mcp_name_idx = separators[0]
        mcp_tool_name = None
    mcp_name = mcp_names.get(mcp_name_idx)
    if not mcp_tool_name:
        return mcp_name or mcp_name_idx

    if not mcp_name:
        return mcp_tool_name
    return f'[{mcp_name}] {mcp_tool_name}'


async def generate_with_mcp(messages: List[dict], mcp_config: dict,
                            enabled_mcp_servers: list, sys_prompt: str,
                            get_llm: Callable):

    # todo: only for test
    print(
        f'>>messages: {messages}, mcp_config: {mcp_config}, enabled_mcp_servers: {enabled_mcp_servers}, sys_prompt: {sys_prompt}, get_llm: {get_llm}'
    )
    # [{'role': 'user', 'content': '北京今天天气怎么样'}], mcp_config: {'mcpServers': {'arxiv': {'type': 'sse', 'url': 'https://mcp-fb59e3e6-1e8e-461a.api-inference.modelscope.cn/sse'}, '高德地图': {'type': 'sse', 'url': 'https://mcp-948d4e04-78ae-4500.api-inference.modelscope.cn/sse'}, 'time': {'type': 'sse', 'url': 'https://mcp-80d0e36d-3045-4599.api-inference.modelscope.cn/sse'}, 'fetch': {'type': 'sse', 'url': 'https://mcp-cdb79f47-15a7-4a72.api-inference.modelscope.cn/sse'}}}, enabled_mcp_servers: ['高德地图', 'time', 'fetch'], sys_prompt: You are a helpful assistant., get_llm: <function submit.<locals>.<lambda> at 0x7f5346b68860>

    pass
