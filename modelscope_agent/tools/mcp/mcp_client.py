# noqa
import inspect
import os
import re
import shutil
import time
import uuid
from contextlib import AsyncExitStack
from typing import Any, Dict, Literal

import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction)

EncodingErrorHandler = Literal['strict', 'ignore', 'replace']

DEFAULT_ENCODING = 'utf-8'
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = 'strict'

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5


class MCPClient:

    def __init__(self, mcp_config: Dict[str, Any], api_config: Dict[str, Any]):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.mcp = mcp_config
        self.current_server = None

        self.model = api_config['model']
        self.token = api_config['api_key']
        self.base_url = api_config['model_server']
        self.client = OpenAI(
            api_key=self.token,
            base_url=self.base_url,
        )

    @staticmethod
    def parse_config(mcp_servers: Dict[str, Any]) -> Dict[str, Any]:
        config_json = {}
        for mcp_server_name, mcp_content in mcp_servers.items():
            if 'command' in mcp_content:
                command = mcp_content['command']
                if 'fastmcp' in command:
                    command = shutil.which('fastmcp')
                    if not command:
                        raise FileNotFoundError(
                            'Cannot locate the fastmcp command file, please install fastmcp by `pip install fastmcp`'
                        )
                    mcp_content['command'] = command
                if 'uv' in command:
                    command = shutil.which('uv')
                    if not command:
                        raise FileNotFoundError(
                            'Cannot locate the uv command, please consider your installation of Python.'
                        )

                args = mcp_content['args']
                for idx in range(len(args)):
                    if '/path/to' in args[idx]:
                        # TODO: 对stdio的工具需要进一步整合
                        args[idx] = args[idx].replace('/path/to', os.getcwd())
            config_json[mcp_server_name] = mcp_content

        return config_json

    async def call_tool(self, server_name: str, tool_name: str,
                        tool_args: dict):
        response = await self.sessions[server_name].call_tool(
            tool_name, tool_args)
        texts = []
        for content in response.content:
            if content.type == 'text':
                texts.append(content.text)
        if texts:
            return '\n\n'.join(texts)
        else:
            return 'execute error'

    async def get_tools(self) -> Dict:
        tools = {}
        for key, session in self.sessions.items():
            tools[key] = []
            response = await session.list_tools()
            tools[key].extend(response.tools)
        return tools

    def generate_response(self,
                          messages,
                          model,
                          tools=None,
                          **kwargs) -> ChatCompletion:
        time.sleep(0.5)
        if tools:
            tools = [{
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': tool['input_schema']
                }
            } for tool in tools]

        _e = None
        completion = None
        parameters = inspect.signature(
            self.client.chat.completions.create).parameters
        kwargs = {
            key: value
            for key, value in kwargs.items() if key in parameters
        }
        for i in range(20):
            try:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    parallel_tool_calls=False,
                    **kwargs)
                _e = None
                break
            except Exception as e:
                print(str(e))
                _e = e
                time.sleep(20)
                continue
        if _e:
            raise _e

        if kwargs.get('stream', False):
            # 流式响应处理
            print_thinking = True
            final_content = ''
            final_func = []
            func_index = -1
            func_id = ''
            func_arguments = ''
            func_name = ''
            done_thinking = False
            done_print_id = False

            for chunk in completion:
                if not done_print_id:
                    print(f'request_id: {chunk.id}')
                    done_print_id = True

                # 处理内容部分
                thinking_chunk = chunk.choices[0].delta.reasoning_content
                answer_chunk = chunk.choices[0].delta.content

                # 添加内容到final_content
                if thinking_chunk != '' and print_thinking:
                    final_content += thinking_chunk
                elif answer_chunk != '':
                    if not done_thinking and print_thinking:
                        done_thinking = True
                    final_content += answer_chunk

                # 处理tool_calls部分
                if chunk.choices[0].delta.tool_calls:
                    func = chunk.choices[0].delta.tool_calls

                    if len(func) == 1 and func[0].index == -1:
                        continue

                    for tool_call in func:
                        if tool_call.index != func_index:
                            if func_index != -1:
                                toolcall = ChoiceDeltaToolCall(
                                    index=func_index,
                                    id=func_id,
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments=func_arguments,
                                        name=func_name),
                                    type='function')
                                final_func.append(toolcall)
                                func_id = ''
                                func_arguments = ''
                                func_name = ''

                            func_index = tool_call.index

                        if tool_call.id:
                            func_id = tool_call.id
                        if tool_call.function.arguments:
                            func_arguments += tool_call.function.arguments
                        if tool_call.function.name:
                            func_name = tool_call.function.name

            # 添加最后一个函数调用
            if func_index != -1:
                toolcall = ChoiceDeltaToolCall(
                    index=func_index,
                    id=func_id,
                    function=ChoiceDeltaToolCallFunction(
                        arguments=func_arguments, name=func_name),
                    type='function')
                final_func.append(toolcall)

            # 检查content中是否包含<tool_call>格式的内容
            if '<tool_call>' in final_content:

                # 提取JSON字符串并转换为函数调用
                tool_call_pattern = r'<tool_call>\n(.*?)</tool_call>'
                matches = re.findall(tool_call_pattern, final_content,
                                     re.DOTALL)

                if matches:
                    for json_str in matches:
                        try:
                            func_data = json.loads(json_str)
                            toolcall = ChoiceDeltaToolCall(
                                index=0,
                                id=str(uuid.uuid4()),  # 生成随机ID
                                function=ChoiceDeltaToolCallFunction(
                                    arguments=json.dumps(
                                        func_data.get('arguments', {})),
                                    name=func_data.get('name', '')),
                                type='function')
                            final_func.insert(0, toolcall)  # 插入到列表开头
                        except json.JSONDecodeError:
                            continue

                # 只保留<tool_call>之前的内容
                final_content = final_content.split('<tool_call>')[0]

            return final_content, final_func

        # 非流式响应处理
        message = completion.choices[0].message
        try:
            reasoning = message.model_extra['reasoning_content']
        except Exception as e:
            reasoning = ''
            print(e)
        content = reasoning + (message.content or '')

        return content, completion.choices[0].message.tool_calls

    @staticmethod
    def generate_config(mcp_servers: Dict[str, Any]) -> Dict[str, Any]:

        for mcp_server in mcp_servers:

            # Activate local planner `notebook`
            if mcp_server == 'notebook':
                mcp_path = os.path.dirname(os.path.abspath(__file__))
                notebook_mcp_path = os.path.join(mcp_path, 'servers',
                                                 mcp_server)
                notebook_mcp_config_path = os.path.join(
                    notebook_mcp_path, 'config.json')
                if os.path.exists(notebook_mcp_config_path):
                    print(
                        f'Got local planner `notebook`: {notebook_mcp_config_path}'
                    )
                    with open(notebook_mcp_config_path, 'r') as f:
                        content = json.load(f)
                        mcp_content = content[mcp_server]
                        command = mcp_content['command']
                        if 'fastmcp' in command:
                            command = shutil.which('fastmcp')
                            if not command:
                                raise FileNotFoundError(
                                    'Cannot locate the fastmcp command file, '
                                    'please install fastmcp by `pip install fastmcp`'
                                )
                            mcp_content['command'] = command
                        if 'uv' in command:
                            command = shutil.which('uv')
                            if not command:
                                raise FileNotFoundError(
                                    'Cannot locate the uv command, '
                                    'please consider your installation of Python.'
                                )

                        args = mcp_content['args']
                        for idx in range(len(args)):
                            if 'server.py' in args[idx]:
                                args[idx] = os.path.join(
                                    notebook_mcp_path, 'server.py')

                    mcp_servers[mcp_server] = mcp_content

        print(f'mcp_servers: {mcp_servers}')
        return mcp_servers

    async def connect_to_server(self, server_name: str, **kwargs):
        print('connect_to_server')
        command = kwargs.get('command')
        url = kwargs.get('url')
        session_kwargs = kwargs.get('session_kwargs')
        print(f'kwargs: {kwargs}')
        if url:
            # transport: 'sse'
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(
                    url, kwargs.get('headers'),
                    kwargs.get('timeout', DEFAULT_HTTP_TIMEOUT),
                    kwargs.get('sse_read_timeout', DEFAULT_SSE_READ_TIMEOUT)))
            read, write = sse_transport
            session_kwargs = session_kwargs or {}
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write, **session_kwargs))

            await session.initialize()
            # Store session
            self.sessions[server_name] = session

            # Set as current if it's the first one
            if self.current_server is None:
                self.current_server = server_name

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server '{server_name}' with tools:",
                  [tool.name for tool in tools])

        elif command:
            # transport: 'stdio'
            args = kwargs.get('args')
            if not args:
                raise ValueError(
                    "'args' parameter is required for stdio connection")
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=kwargs.get('env'),
                encoding=kwargs.get('encoding', DEFAULT_ENCODING),
                encoding_error_handler=kwargs.get(
                    'encoding_error_handler', DEFAULT_ENCODING_ERROR_HANDLER),
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params))

            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write))

            await session.initialize()

            # Store session
            self.sessions[server_name] = session

            # Set as current if it's the first one
            if self.current_server is None:
                self.current_server = server_name

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server '{server_name}' with tools:",
                  [tool.name for tool in tools])

        else:
            raise ValueError(
                "'url' or 'command' parameter is required for connection")

        return server_name

    async def switch_server(self, server_name: str):
        """Switch to a different connected server"""
        if server_name not in self.sessions:
            raise ValueError(
                f"Server '{server_name}' not connected. Available servers: {list(self.sessions.keys())}"
            )

        self.current_server = server_name
        print(f'Switched to server: {server_name}')

        # List available tools on current server
        response = await self.sessions[server_name].list_tools()
        tools = response.tools
        print('Available tools:', [tool.name for tool in tools])

    async def list_servers(self):
        """List all connected servers"""
        if not self.sessions:
            print('No servers connected')
            return

        print('\nConnected servers:')
        for name in self.sessions.keys():
            marker = '* ' if name == self.current_server else '  '
            print(f'{marker}{name}')

    def summary(self, query, content, **kwargs):
        prompt = """Based on the query: "{query}", filter this content to keep only the most relevant information.

Your task is to:
1. Mandatory: Retain ALL information directly relevant to the query
2. Mandatory: Keep URLs and links that provide useful resources related to the query
3. Mandatory: Filter out URLs and content that are not helpful for addressing the query
4. Mandatory: Preserve technical details, specifications, and instructions related to the query
5. Mandatory: Maintain the connection between relevant information and its corresponding URLs

Format your response as a JSON object without code block markers, containing:
- "title": A descriptive title reflecting the query focus (5-12 words)
- "summary": Filtered content with only query-relevant information and URLs
- "status": "success" if the content is relevant to the query, \
"error" if the content is irrelevant or contains incorrect information

Content to process:
{content}

Filtering guidelines:
- Keep URLs that provide resources, tools, downloads, or information directly related to the query
- Remove URLs to general pages, social media, promotional content, or unrelated material
- Keep all technical specifications, code samples, or detailed instructions that address the query
- Preserve product names, model numbers, and version information relevant to the query
- Remove generic content, filler text, or background information that doesn't help answer the query
- When evaluating a URL, consider where it leads and whether that destination would help someone with this query

Error detection guidelines:
- Set "status" to "error" if the content appears to be in a different language than expected
- Set "status" to "error" if the content is about a completely different topic than the query \
(e.g., query about technology but content about tourism)
- Set "status" to "error" if the content contains obvious factual errors or contradictions
- Set "status" to "error" if URLs lead to unrelated content like tourism sites when querying for technical information
- Set "status" to "error" if the content appears to be machine-translated or unintelligible
- When setting "status" to "error", include a brief explanation in the summary field

The goal is intelligent filtering with error detection - keeping all information and \
links that would be valuable for someone with this specific query, while removing \
everything else and flagging irrelevant or incorrect content.
DO NOT add any other parts like ```json which may cause parse error of json."""

        query = prompt.replace('{query}', query).replace('{content}', content)
        messages = [{'role': 'user', 'content': query}]

        if len(query) < 80000:
            content, _ = self.generate_response(messages, self.model, **kwargs)
        else:
            content = 'Content too long, you need to try another website or search another keyword'
        return content

    async def connect_all_servers(self, query):

        assert self.mcp, 'MCP config is required'

        tools: dict = self.generate_config(self.mcp['mcpServers'])

        # tools: {'MiniMax-MCP': {xxx}, 'notebook': {xxx}}
        for tool in tools:
            cmd = tools[tool]
            env_dict = cmd.pop('env', {})
            env_dict = {
                key: value if value else os.environ.get(key, '')
                for key, value in env_dict.items()
            }
            await self.connect_to_server(server_name=tool, env=env_dict, **cmd)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
