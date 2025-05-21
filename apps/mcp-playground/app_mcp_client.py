from typing import List, Callable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
# from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
import os
import re
from contextlib import asynccontextmanager

from modelscope_agent.agent import Agent


def parse_mcp_config(mcp_config: dict, enabled_mcp_servers: list = None):
    mcp_servers = {}
    for server_name, server in mcp_config.get("mcpServers", {}).items():
        if server.get("type", "") == "stdio" or (enabled_mcp_servers is not None
                                         and server_name
                                         not in enabled_mcp_servers):
            continue
        new_server = {**server}
        new_server["transport"] = server["type"]
        del new_server["type"]
        if server.get("env"):
            env = {'PYTHONUNBUFFERED': '1', 'PATH': os.environ.get('PATH', '')}
            env.update(server["env"])
            new_server["env"] = env
        mcp_servers[server_name] = new_server
    return mcp_servers


@asynccontextmanager
async def get_mcp_client(mcp_servers: dict):
    async with MultiServerMCPClient(mcp_servers) as client:
        yield client


async def get_mcp_prompts(mcp_config: dict, get_llm: Callable):
    try:
        mcp_servers = parse_mcp_config(mcp_config)
        if len(mcp_servers.keys()) == 0:
            return {}
        llm: BaseChatModel = get_llm()
        async with get_mcp_client(mcp_servers) as client:
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
                        f"请使用 {mcp_name} 服务的功能帮我查询信息或解决问题",
                    ]
            return raw_examples
    except Exception as e:
        print('Prompt Error:', e)
        return {
            mcp_name: [
                f"请使用 {mcp_name} 服务的功能帮我查询信息或解决问题",
            ]
            for mcp_name in mcp_servers.keys()
        }


def convert_mcp_name(tool_name: str, mcp_names: dict):
    if not tool_name:
        return tool_name
    separators = tool_name.split("__TOOL__")
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
    return f"[{mcp_name}] {mcp_tool_name}"


def generate_with_mcp(messages: List[dict], mcp_config: dict,
                            enabled_mcp_servers: list, sys_prompt: str,
                            get_llm: dict, chatbot):
    mcp_servers = {}
    mcp_servers["mcpServers"] = parse_mcp_config(mcp_config, enabled_mcp_servers)
    agent_executor = Agent(
        mcp=mcp_servers, llm=get_llm, instruction=sys_prompt)
    response = agent_executor.run("你好")
    for chunk in response:
        response += chunk
        chatbot[-1]["content"] = response
        yield chatbot


#     async with get_mcp_client(mcp_servers) as client:
#         tools = []
#         mcp_tools = []
#         mcp_names = {}
#         for i, server_name_to_tool in enumerate(
#                 client.server_name_to_tools.items()):
#             mcp_name, server_tools = server_name_to_tool
#             mcp_names[str(i)] = mcp_name
#             for tool in server_tools:
#                 new_tool = tool.model_copy()
#                 # tool match ^[a-zA-Z0-9_-]+$
#                 new_tool.name = f"{i}__TOOL__{tool.name}"
#                 mcp_tools.append(new_tool)
#         tools.extend(mcp_tools)
#         llm: BaseChatModel = get_llm()
#         tool_result_instruction = """When a tool returns responses containing URLs or links, please format them appropriately based on their CORRECT content type:

# For example:
# - Videos should use <video> tags
# - Audio should use <audio> tags
# - Images should use ![description](URL) or <img> tags
# - Documents and web links should use [description](URL) format

# Choose the appropriate display format based on the URL extension or content type information. This will provide the best user experience.

# Remember that properly formatted media will enhance the user experience, especially when content is directly relevant to answering the query.
# """
#         attachment_instruction = """
# The following instructions apply when user messages contain "Attachment links: [...]":

# These links are user-uploaded attachments that contain important information for this conversation. These are temporary, secure links to files the user has specifically provided for analysis.

# IMPORTANT INSTRUCTIONS:
# 1. These attachments should be your PRIMARY source of information when addressing the user's query.
# 2. Prioritize analyzing and referencing these documents BEFORE using any other knowledge.
# 3. If the content in these attachments is relevant to the user's request, base your response primarily on this information.
# 4. When you reference information from these attachments, clearly indicate which document it comes from.
# 5. If the attachments don't contain information needed to fully address the query, only then supplement with your general knowledge.
# 6. These links are temporary and secure, specifically provided for this conversation.
# 7. IMPORTANT: Do not use the presence of "Attachment links: [...]" as an indicator of the user's preferred language. This is an automatically added system text. Instead, determine the user's language from their actual query text.

# Begin your analysis by examining these attachments first, and structure your thinking to prioritize insights from these documents.
# """
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", tool_result_instruction),
#             ("system", sys_prompt),
#             ("system", attachment_instruction),
#             MessagesPlaceholder(variable_name="messages"),
#         ])

#         langchain_messages = []
#         for msg in messages:
#             if msg["role"] == "user":
#                 langchain_messages.append(HumanMessage(content=msg["content"]))
#             elif msg["role"] == "assistant":
#                 langchain_messages.append(AIMessage(content=msg["content"]))

#         agent_executor = create_react_agent(llm, tools, prompt=prompt)
#         use_tool = False
#         async for step in agent_executor.astream(
#             {"messages": langchain_messages},
#                 config={"recursion_limit": 50},
#                 stream_mode=["values", "messages"],
#         ):
#             if isinstance(step, tuple):
#                 if step[0] == "messages":
#                     message_chunk = step[1][0]
#                     if hasattr(message_chunk, "content"):
#                         if isinstance(message_chunk, ToolMessage):
#                             use_tool = False
#                             yield {
#                                 "type":
#                                 "tool",
#                                 "name":
#                                 convert_mcp_name(message_chunk.name,
#                                                  mcp_names),
#                                 "content":
#                                 message_chunk.content
#                             }
#                         elif hasattr(message_chunk,
#                                      'tool_call_chunks') and len(
#                                          message_chunk.tool_call_chunks) > 0:
#                             for tool_call_chunk in message_chunk.tool_call_chunks:
#                                 yield {
#                                     "type":
#                                     "tool_call_chunks",
#                                     "name":
#                                     convert_mcp_name(tool_call_chunk["name"],
#                                                      mcp_names),
#                                     "content":
#                                     tool_call_chunk["args"],
#                                     "next_tool":
#                                     bool(use_tool and tool_call_chunk["name"])
#                                 }
#                                 if tool_call_chunk["name"]:
#                                     use_tool = True
#                         elif message_chunk.content:
#                             yield {
#                                 "type": "content",
#                                 "content": message_chunk.content
#                             }
