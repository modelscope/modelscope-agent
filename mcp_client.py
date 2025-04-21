import argparse
import json
import os
import shutil

from contextlib import AsyncExitStack
from typing import Dict, List, Any, Literal, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from openai import OpenAI
from openai.types.chat import ChatCompletion

EncodingErrorHandler = Literal["strict", "ignore", "replace"]

DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER: EncodingErrorHandler = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5

class MCPClient:

    default_system = ('You are an assistant which helps me to finish a complex job. Tools may be given to you '
                      'and you must choose some of them one per round to finish my request.')

    def __init__(self, mcp):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.current_server = None
        self.args = self.parse_args()
        self.mcp_servers_config = self.generate_config(mcp)
        self.client = OpenAI(
            api_key=self.args.token or os.getenv('MODELSCOPE_API_KEY'),
            base_url=self.args.base_url,
        )

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_url", type=str, default="https://api-inference.modelscope.cn/v1/")
        parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct")
        parser.add_argument("--token", type=str)
        return parser.parse_args()

    def generate_response(self, messages, model, max_tokens=500, tools=None) -> ChatCompletion:
        if tools:
            tools = [
                {
                    'type': 'function',
                    'function': {
                        'name': tool['name'],
                        'description': tool['description'],
                        'parameters': tool['input_schema']
                    }
                } for tool in tools
            ]
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=0.8,
            temperature=0.5,
            tools=tools,
            max_completion_tokens=max_tokens,
        )
        return completion

    @staticmethod
    def generate_config(mcp: Union[Dict, str]) -> Dict[str, Any]:
        if isinstance(mcp, Dict):
            config_json = mcp.get('mcpServers') or mcp
            return config_json
        if not os.path.exists(mcp):
            return {}
        config_json = {}
        if os.path.basename(mcp) == 'mcp_config.json':
            config_file = mcp
            with open(config_file, 'r') as f:
                content = json.load(f)
                config_json = content.get('mcpServers') or content
        if os.path.basename(mcp) == 'mcp-central':
            mcp_path = os.path.abspath(mcp)
            for base_dir, dirs, files in os.walk(os.path.join(mcp_path, 'mcp_central')):
                mcp_servers = dirs
                break
            for mcp_server in mcp_servers:
                mcp_abs_path = os.path.join(mcp_path, 'mcp_central', mcp_server)
                config_file = os.path.join(mcp_abs_path, 'config.json')
                with open(config_file, 'r') as f:
                    content = json.load(f)
                    mcp_content = content[mcp_server]
                    command = mcp_content['command']
                    if 'fastmcp' in command:
                        command = shutil.which("fastmcp")
                        if not command:
                            raise FileNotFoundError(f'Cannot locate the fastmcp command file, '
                                                    f'please install fastmcp by `pip install fastmcp`')
                        mcp_content['command'] = command
                    if 'uv' in command:
                        command = shutil.which("uv")
                        if not command:
                            raise FileNotFoundError(f'Cannot locate the uv command, '
                                                    f'please consider your installation of Python.')

                    args = mcp_content['args']
                    for idx in range(len(args)):
                        if 'mcp.py' in args[idx]:
                            args[idx] = os.path.join(mcp_abs_path, 'mcp.py')
                config_json[mcp_server] = mcp_content
        return config_json

    async def connect_to_server(self, server_name: str, **kwargs):
        print(f'connect_to_server')
        command = kwargs.get('command')
        url = kwargs.get('url')
        session_kwargs = kwargs.get("session_kwargs")
        if url:
            # transport: 'mse'
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(url,
                           kwargs.get("headers"),
                           kwargs.get("timeout", DEFAULT_HTTP_TIMEOUT),
                           kwargs.get("sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT))
            )
            read, write = sse_transport
            session_kwargs = session_kwargs or {}
            session = await self.exit_stack.enter_async_context(ClientSession(read, write, **session_kwargs))

            await session.initialize()
            # Store session
            self.sessions[server_name] = session

            # Set as current if it's the first one
            if self.current_server is None:
                self.current_server = server_name

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

        elif command:
            # transport: 'stdio'
            args = kwargs.get('args')
            if not args:
                raise ValueError("'args' parameter is required for stdio connection")
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=kwargs.get("env"),
                encoding=kwargs.get("encoding", DEFAULT_ENCODING),
                encoding_error_handler=kwargs.get(
                    "encoding_error_handler", DEFAULT_ENCODING_ERROR_HANDLER
                ),
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )

            await session.initialize()

            # Store session
            self.sessions[server_name] = session

            # Set as current if it's the first one
            if self.current_server is None:
                self.current_server = server_name

            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

        else:
            raise ValueError("'url' or 'command' parameter is required for connection")

        return server_name

    async def _connect_to_server(self, command, args, server_name: str = None):
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )

        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )

        await session.initialize()

        # Store session
        self.sessions[server_name] = session

        # Set as current if it's the first one
        if self.current_server is None:
            self.current_server = server_name

        # List available tools
        response = await session.list_tools()
        tools = response.tools
        print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

        return server_name

    async def __aenter__(self):
        await self.connect_all_servers()
        return self

    async def switch_server(self, server_name: str):
        """Switch to a different connected server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected. Available servers: {list(self.sessions.keys())}")

        self.current_server = server_name
        print(f"Switched to server: {server_name}")

        # List available tools on current server
        response = await self.sessions[server_name].list_tools()
        tools = response.tools
        print(f"Available tools:", [tool.name for tool in tools])

    async def list_servers(self):
        """List all connected servers"""
        if not self.sessions:
            print("No servers connected")
            return

        print("\nConnected servers:")
        for name in self.sessions.keys():
            marker = "* " if name == self.current_server else "  "
            print(f"{marker}{name}")

    async def process_query(self, query: str) -> str:
        messages = [{'role': 'system', 'content': self.default_system}, {"role": "user", "content": query}]

        tools = []
        for key, session in self.sessions.items():
            response = await session.list_tools()
            available_tools = [
                {
                    "name": key + '.' + tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in response.tools
            ]
            tools.extend(available_tools)

        final_text = []
        print(f'tools: {tools}')
        while True:
            response = self.generate_response(messages, self.args.model, tools=tools)
            message = response.choices[0].message
            content = message.content
            final_text.append(content)
            messages.append({
                "role": "assistant",
                "content": content,
                'tool_calls': message.tool_calls,
            })
            if message.tool_calls:
                for tool in message.tool_calls:
                    name = tool.function.name
                    args = tool.function.arguments
                    key, tool_name = name.split('.')
                    args = json.loads(args)
                    print(f'key: {key}, tool_name: {tool_name}')
                    result = await self.sessions[key].call_tool(tool_name, args)
                    final_text.append(f"[Calling tool {name} with args {args}]")
                    messages.append({
                        'role': 'tool',
                        'content': result.content[0].text,
                        'tool_call_id': tool.id,
                    })
            else:
                break
        return "\n".join(final_text)

    async def connect_all_servers(self, query=None):
        # config = self.generate_config(self.args.mcp)
        # if not self.args.mcp:
        #     keys = config.keys()
        #     messages = [dict(role='system',
        #                      content=(
        #                          'You are an assistant which helps me to finish a complex job. '
        #                          'Tools may be given to you '
        #                          'and you must choose which tools are required list them in a '
        #                          'json array and wraps it in a <box></box>')),
        #                 {
        #                     'role': 'user',
        #                     'content': f'The user job: {query}, all available tools: {list(keys)}',
        #                 }]
        #     response = self.generate_response(messages, self.args.model)
        #     content = response.choices[0].message.content
        #     if '<box>' not in content:
        #         return
        #     _, server = content.split('<box>')
        #     tools, _ = tools.split('</box>')
        #     tools = tools.strip()
        #     tools = json.loads(tools)
        # else:
        #     tools = self.args.mcp
        mcp_servers = self.mcp_servers_config
        for server_name, config in mcp_servers.items():
            print('00')
            await self.connect_to_server(server_name, **config)
        print('11')

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        while True:
            try:
                user_input = input("\nQuery: ").strip()
                await self.connect_all_servers(user_input)
                response = await self.process_query(user_input)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    def chat_loop_(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        import asyncio

        while True:
            try:
                user_input = input("\nQuery: ").strip()
                asyncio.run(self.connect_all_servers(user_input))
                response = asyncio.run(self.process_query(user_input))
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

if __name__ == '__main__':
    mcp_config = {
        "mcpServers": {
            "time": {
                "type": "sse",
                "url": "https://agenttor-mod-dd-cbwtrtihpn.cn-zhangjiakou.fcapp.run/sse"
            },
            "fetch": {
                "type":
                "sse",
                "url":
                "https://mcp-cdb79f47-15a7-4a72.api-inference.modelscope.cn/sse"
            }
        }
    }

    mcp_client = MCPClient(mcp=mcp_config)
    mcp_client.chat_loop_()

    #import asyncio

    #asyncio.run(mcp_client.chat_loop())