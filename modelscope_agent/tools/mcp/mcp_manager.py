import asyncio
import threading
from typing import Any, Dict, List, Union

import json

from ..base import BaseTool
from .mcp_client import MCPClient

DEFAULT_TOOL_EXCLUDES = [{'amap-maps': ['maps_geo']}]


class MCPManager:

    def __init__(self,
                 mcp_config: Dict[str, Any],
                 api_config: Dict[str, Any],
                 tool_includes: List[Union[str, Dict[str, List]]]
                 or None = None,
                 tool_excludes: List[Union[str, Dict[str, List]]]
                 or None = DEFAULT_TOOL_EXCLUDES):
        self.tool_meta: Dict[str, ]
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.start_loop, daemon=True)
        self.loop_thread.start()

        self.client: MCPClient = MCPClient(mcp_config, api_config)
        self.all_tools = self.init_tools()
        self.tool_includes = tool_includes
        self.tool_excludes = tool_excludes

    def get_tools(self):
        allowed_tools = []
        for tool in self.all_tools:
            service_name = tool.name.split('---')[0]
            api_name = tool.name.split(
                '---')[1] if '---' in tool.name else None

            # 默认行为: 如果includes为None则包含所有工具
            should_include = True if self.tool_includes is None else False

            if self.tool_excludes:
                for exclude in self.tool_excludes:
                    if isinstance(exclude, str) and exclude == service_name:
                        should_include = False
                        break
                    if isinstance(exclude, dict):
                        for service, apis in exclude.items():
                            if service == service_name and api_name in apis:
                                should_include = False
                                break

            # 检查includes规则
            if self.tool_includes:
                for include in self.tool_includes:
                    if isinstance(include, str) and include == service_name:
                        should_include = True
                        break
                    if isinstance(include, dict):
                        for service, apis in include.items():
                            if service == service_name and api_name in apis:
                                should_include = True
                                break

            if should_include:
                allowed_tools.append(tool)

        return allowed_tools

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def init_tools(self):
        future = asyncio.run_coroutine_threadsafe(self.init_tools_async(),
                                                  self.loop)
        try:
            result = future.result()
            return result
        except Exception as e:
            print(f'Failed in initializing MCP tools: {e}')
            raise e

    async def init_tools_async(self):
        client = self.client
        await client.connect_all_servers(None)
        mcp_tools = await client.get_tools()
        ms_tools = []
        for server_name, tools in mcp_tools.items():
            for tool in tools:
                parameters = tool.inputSchema
                # The required field in inputSchema may be empty and needs to be initialized.
                if 'required' not in parameters:
                    parameters['required'] = []
                # Remove keys from parameters that do not conform to the standard OpenAI schema
                # Check if the required fields exist
                required_fields = {'type', 'properties', 'required'}
                missing_fields = required_fields - parameters.keys()
                if missing_fields:
                    raise ValueError(
                        f'Missing required fields in schema: {missing_fields}')

                # Keep only the necessary fields
                cleaned_parameters = {
                    'type': parameters['type'],
                    'properties': parameters['properties'],
                    'required': parameters['required']
                }
                register_name = server_name + '---' + tool.name
                agent_tool = self.create_tool_class(
                    register_name=register_name,
                    mcp_server_name=server_name,
                    tool_name=tool.name,
                    tool_desc=tool.description,
                    tool_parameters=cleaned_parameters)
                ms_tools.append(agent_tool)
        return ms_tools

    def create_tool_class(self, register_name, mcp_server_name, tool_name,
                          tool_desc, tool_parameters):
        manager = self
        mcp_server_name = mcp_server_name

        class ToolClass(BaseTool):
            name = register_name
            description = tool_desc
            parameters = tool_parameters

            def call(self, params: Union[str, dict], **kwargs) -> str:
                tool_args = json.loads(params) if isinstance(params,
                                                             str) else params
                # Submit coroutine to the event loop and wait for the result
                future = asyncio.run_coroutine_threadsafe(
                    manager.client.call_tool(mcp_server_name, tool_name,
                                             tool_args), manager.loop)
                try:
                    result = future.result()
                    return result
                except Exception as e:
                    print(f'Failed in executing MCP tool: {e}')
                    raise e

        ToolClass.__name__ = f'{register_name}_Class'
        return ToolClass()

    def shutdown(self):
        asyncio.run_coroutine_threadsafe(self.client.cleanup(), self.loop)
        del self.client
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
