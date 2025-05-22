# flake8: noqa
import re

import json
from modelscope_agent.tools.mcp.mcp_client import MCPClient
from modelscope_agent.tools.mcp.utils import parse_mcp_config


async def get_mcp_prompts(mcp_config: dict, model: str,
                          openai_client: 'openai.AsyncOpenAI'):
    try:
        mcp_config = parse_mcp_config(mcp_config)
        if len(mcp_config['mcpServers'].keys()) == 0:
            return {}
        async with MCPClient(mcp_config) as client:
            tools = await client.get_tools()
            mcp_tool_descriptions = {}
            for mcp_name, server_tools in tools.items():
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

            response = await openai_client.chat.completions.create(
                model=model,
                stream=False,
                extra_body={'enable_thinking': False},
                messages=[{
                    'role': 'user',
                    'content': prompt
                }])
            print(f'response: {response}')
            content = response.choices[0].message.content
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
    except Exception as e:
        print('Prompt Error:', e)
        return {
            mcp_name: [
                f'请使用 {mcp_name} 服务的功能帮我查询信息或解决问题',
            ]
            for mcp_name in mcp_config['mcpServers'].keys()
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
