import asyncio
import copy
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import json
from modelscope_agent import Agent
from modelscope_agent.agent import enable_run_callback
from modelscope_agent.llm.base import BaseChatModel
from modelscope_agent.tools.base import (TOOL_REGISTRY, BaseTool,
                                         ToolServiceProxy)
from modelscope_agent.tools.mcp import MCPManager
from modelscope_agent.tools.mcp.utils import fix_json_brackets
from modelscope_agent.utils.qwen_agent.base import get_chat_model
from modelscope_agent.utils.qwen_agent.schema import (CONTENT, ROLE, SYSTEM,
                                                      ContentItem, Message)


class AgentWithMCP(Agent):

    default_system = f"""You are an assistant that helps generate comprehensive documentations or \
    webpages from gathered information. Today is {datetime.now().strftime("%Y-%m-%d")}.

        ## Planning

        You need to create a CONCISE, FOCUSED plan with ONLY meaningful, actionable steps, \
        rely on the plan after you made it.

        If you are making website, just make one single step for writing code to avoid too much messages. \
        When developing a website, please implement complete and ready-to-use code. \
        There is no need to save space when implementing the code. Please implement every line of code. \
        Use proper event delegation or direct event binding

        Give your final result(documentation/code) in <result></result> block.

        Here shows a plan example:

         ```
        1. Research & Content Gathering:
           1.1. Search and collect comprehensive information on [topic] using user's language
           1.2. Identify and crawl authoritative sources for detailed content
           1.3. Crawl enough high-quality medias(e.g. image links) from compatible platforms

        2. Content Creation & Organization:
           2.1. Develop main content sections with complete information
           2.3. Organize information with logical hierarchy and flow

        3. Design & Animation Implementation:
           3.1. Create responsive layout with modern aesthetic, with all the useful information collected
           3.2. Implement key animations for enhanced user experience
           3.3. Write the final code...
        ```

        When executing specific task steps, please pay attention to the consistency of the previous and next content. \
        When generating a series of images, you need to ensure that the images are generated consistently. \
        Please clearly describe the main features such as color, type, and shape when generating each image.

        History messages of the previous main step will not be kept, \
        so you need to WRITE a concise but essential summary_and_result \
        when calling `notebook---advance_to_next_step` for each sub-step.
        In the later steps, you can only see the plans you made and the summary_and_result from the previous steps.
        So you must MINIMIZE DEPENDENCIES between the the steps in the plan.
        Note: The URL needs to retain complete information.

        Here are some summary_and_result examples:

        · Topic X has three primary categories: A, B, and C
        · Latest statistics show 45% increase in adoption since 2023
        · Expert consensus indicates approach Y is most effective
        · Primary source: https://example.com/comprehensive-guide (contains detailed sections on implementation)
        · Images: ["https://example.com/image1.jpg?Expires=a&KeyId=b&Signature=c", "https://example.com/image2.jpg", \
        "https://example.com/diagram.png"] (Please copy the entire content of the url without doing any changes)
        · Reference documentation: https://docs.example.com/api (sections 3.2-3.4 particularly relevant)
        · Will focus on mobile-first approach due to 78% of users accessing via mobile devices
        · Selected blue/green color scheme based on industry standards and brand compatibility
        · Decided to implement tabbed interface for complex data presentation
        · CODE:
        ```
        ... # complete and ready-to-use code here
        ```
        """

    connector = '\n\nHere gives the user query:\n\n'

    # used to record all the tools' instance, moving here to avoid `del` method crash.
    function_map: dict = {}

    def __init__(self,
                 function_list: Union[Dict, List[Union[str, Dict]],
                                      None] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 storage_path: Optional[str] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 instruction: Union[str, dict] = None,
                 use_tool_api: bool = False,
                 callbacks: list = None,
                 openapi_list: Optional[List[Union[str, Dict]]] = None,
                 **kwargs):
        """
        init tools/llm/instruction for one agent

        Args:
            function_list: A list of tools
                (1)When str: tool names
                (2)When Dict: tool cfg
            llm: The llm config of this agent
                (1) When Dict: set the config of llm as {'model': '', 'api_key': '', 'model_server': ''}
                (2) When BaseChatModel: llm is sent by another agent
            storage_path: If not specified otherwise, all data will be stored here in KV pairs by memory
            name: the name of agent
            description: the description of agent, which is used for multi_agent
            instruction: the system instruction of this agent
            use_tool_api: whether to use the tool service api, else to use the tool cls instance
            callbacks: the callbacks that could be used during different phase of agent loop
            openapi_list: the openapi list for remote calling only
            kwargs: other potential parameters
        """

        self.api_config = {}
        if isinstance(llm, Dict):
            self.api_config = llm
            llm = get_chat_model(cfg=llm)

        assert len(self.api_config
                   ) > 0, 'Only support OpenAI api format config for now'

        self.mcp_manager: MCPManager = None

        super().__init__(
            function_list=function_list,
            llm=llm,
            storage_path=storage_path,
            name=name,
            description=description,
            instruction=instruction,
            use_tool_api=use_tool_api,
            callbacks=callbacks,
            openapi_list=openapi_list,
            **kwargs)

    @enable_run_callback
    def run(self, messages: List[Union[Dict, 'Message']], **kwargs
            ) -> Union[Iterator[List['Message']], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        # Check system prompt
        system: bool = kwargs.pop(SYSTEM, True)
        system_msg = [item for item in new_messages if item.role == SYSTEM]
        if system and len(system_msg) < 1:
            new_messages.insert(
                0, Message(role=SYSTEM, content=self.default_system))

        if self.instruction:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                # Add the system instruction to the agent
                new_messages.insert(
                    0, Message(role=SYSTEM, content=self.instruction))
            else:
                # Already got system message in new_messages
                if isinstance(new_messages[0][CONTENT], str):
                    new_messages[0][
                        CONTENT] = self.instruction + '\n\n' + new_messages[0][
                            CONTENT]
                else:
                    assert isinstance(new_messages[0][CONTENT], list)
                    assert new_messages[0][CONTENT][0].text
                    new_messages[0][CONTENT] = [
                        ContentItem(text=self.instruction + '\n\n')
                    ] + new_messages[0][CONTENT]  # noqa

        print(f'_return_message_type: {_return_message_type}')
        print(f'new_messages: {new_messages}')

        for res in self._run(messages=new_messages, **kwargs):
            yield {'role': 'assistant', 'content': res}

    def _run(self, messages: List, *args, **kwargs):

        # TODO: check ...
        task_done_cnt = 0
        final_result = ''
        result_section = False

        def call_tool_sync(
            mcp_server_name: str,
            in_tool_name: str,
            in_tool_args: Optional[Dict[str, Any]] = None,
            read_timeout_seconds: Optional[timedelta] = None,
        ):
            """
            Call the tool synchronously.
            """
            future = asyncio.run_coroutine_threadsafe(
                self.mcp_manager.client.call_tool(mcp_server_name,
                                                  in_tool_name, in_tool_args),
                self.mcp_manager.loop)
            try:
                result = future.result()
                return result
            except Exception as e:
                print(f'Failed in executing MCP tool: {e}')
                raise e

        while True:
            # for i in range(3):
            tools = [{
                'name': func_name,
                'description': tool_cls.description,
                'input_schema': tool_cls.parameters,
            } for func_name, tool_cls in self.function_map.items()]
            content, tool_calls = self.mcp_manager.client.generate_response(
                messages, self.mcp_manager.client.model, tools=tools, **kwargs)
            print(f'content: {content}, tool_calls: {tool_calls}')
            if '<task_done>' in content or task_done_cnt >= 4:
                break

            if '<result>' in content and '</result>' in content:
                pattern = r'<result>(.*?)</result>'
                final_result = re.findall(pattern, content, re.DOTALL)
            elif '<result>' in content:
                result_section = True
                final_result += content.split('<result>')[1]
            elif '</result>' in content:
                final_result += content.split('</result>')[0]
                result_section = False
            elif result_section:
                final_result += content

            if content.strip() or tool_calls:
                messages.append({
                    'role':
                    'assistant',
                    'content':
                    content.strip(),
                    'tool_calls':
                    tool_calls if not tool_calls else [tool_calls[0]],
                })
            if tool_calls:
                for tool in tool_calls:
                    try:
                        name = tool.function.name
                        args = tool.function.arguments
                        elems = name.split('---')
                        if len(elems) == 2:
                            key, tool_name = elems
                        elif len(elems) == 1:
                            tool_name = elems[0]
                            original_name = next(
                                (t['name'] for t in tools
                                 if t['name'].endswith(tool_name)), None)
                            key, _ = original_name.split('---')
                        else:
                            messages.append({
                                'role':
                                'user',
                                'content':
                                f'Tool {name} called with error: The tool name is incorrect, please check.',
                            })
                        print(f'>origin_args: {args}')
                        # args = re.sub(r'\n(?!\\n)|(?<!\\)\n', r'\\n', args)
                        args = re.sub(r'(?<!\\)\n', '', args)
                        args = fix_json_brackets(args)
                        print(f'>processed_args: {args}')
                        args = json.loads(args)
                        if tool.function.name == 'notebook---initialize_task':
                            user_query = args.get('user_query', '')
                            user_query = user_query.split(self.connector)
                            if len(user_query) > 1:
                                user_query = user_query[1]
                                args['user_query'] = user_query
                        elif tool.function.name == 'notebook---verify_task_completion':
                            task_done_cnt += 1
                        if 'advance_to_next_step' in tool.function.name:
                            if 'summary_and_result' not in args:
                                args['summary_and_result'] = ''
                            if args['summary_and_result'] is None:
                                args['summary_and_result'] = ''
                            start = 1
                            _messages = [messages[0]]
                            # tool_call_messages = []
                            if messages[0]['role'] == 'system':
                                start = 2
                                _messages.append(messages[1])
                            for i in range(start, len(messages) - 1, 2):
                                resp = messages[i]
                                qry = messages[i + 1]
                                if resp.get(
                                        'tool_calls'
                                ) and 'advance_to_next_step' in resp[
                                        'tool_calls'][0].function.name:
                                    # tool_call_messages.append(resp)
                                    # tool_call_messages.append(qry)
                                    continue
                                _messages.append(resp)
                                _messages.append(qry)
                            if _messages[-1] is not messages[-1]:
                                _messages.append(messages[-1])
                            messages = _messages

                        if tool.function.name == 'web-search---tavily-search':
                            args['include_domains'] = []
                            args['include_raw_content'] = False

                        print(
                            f'>> Start call_tool_sync for mcp_server_name: {key}, '
                            f'tool_name: {tool_name}, arguments: {args}')
                        tool_result = call_tool_sync(
                            mcp_server_name=key,
                            in_tool_name=tool_name,
                            in_tool_args=args)

                        if key in ('web-search'):
                            _args: dict = self.summary(user_query, tool_result,
                                                       **kwargs)
                            _print_origin_result = tool_result
                            if len(_print_origin_result) > 512:
                                _print_origin_result = _print_origin_result[:
                                                                            512] + '...'
                            print(tool_name, args, _print_origin_result)
                            tool_result = str(_args)
                        # if tool.function.name == 'notebook---store_intermediate_results':
                        #     messages[-2]['content'] = f'Tool result cached to notebook with title: {args["title"]}'
                        if 'advance_to_next_step' in tool.function.name:
                            content_and_system = json.loads(tool_result)
                            tool_result = content_and_system[0]
                            if 'Previous main task done' in tool_result:
                                _messages = [messages[0]]
                                if messages[0]['role'] == 'system':
                                    _messages.append(messages[1])
                                messages = _messages
                            if 'Previous main task done' in tool_result:
                                messages.append({
                                    'role': 'user',
                                    'content': tool_result,
                                })
                            else:
                                messages.append({
                                    'role': 'tool',
                                    'content': tool_result,
                                    'tool_call_id': tool.id,
                                })
                        else:
                            messages.append({
                                'role': 'tool',
                                'content': tool_result,
                                'tool_call_id': tool.id,
                            })
                        _print_result = tool_result  # result.content[0].text or ''
                        yield f'{content}\n\n tool call: {name}, {args}\n\n tool result: {_print_result}'
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        messages.append({
                            'role':
                            'tool',
                            'content':
                            f'Tool {name} called with error: ' + str(e),
                            'tool_call_id':
                            tool.id,
                        })
                    print(f'messages len: {len(str(messages))}')
                    break
            else:
                # TODO: FOR claude
                if content:
                    yield content
                messages.append({'role': 'user', 'content': '请继续'})
                break

    def _register_tool(self,
                       tool: Union[str, Dict],
                       tenant_id: str = 'default',
                       **kwargs):
        """
        Instantiate the tool for the agent

        Args:
            tool: the tool should be either in a string format with name as value
            and in a dict format, example
            (1) When str: amap_weather
            (2) When dict: {'amap_weather': {'token': 'xxx'}}
            tenant_id: the tenant_id of the tool is now  for code interpreter that need to keep track of the tenant
        Returns:

        """
        tool_name = tool
        tool_cfg = {}
        if isinstance(tool, dict) and 'mcpServers' in tool:
            from modelscope_agent.tools.mcp import MCPManager
            # tools = MCPManager(tool).get_tools()
            self.mcp_manager = MCPManager(
                mcp_config=tool, api_config=self.api_config)
            for tool in self.mcp_manager.get_tools():
                tool_name = tool.name
                if tool_name not in self.function_map:
                    self.function_map[tool_name] = tool
            return
        if isinstance(tool, dict):
            tool_name = next(iter(tool))
            tool_cfg = tool[tool_name]
        if tool_name not in TOOL_REGISTRY and not self.use_tool_api:
            raise NotImplementedError
        if tool_name not in self.function_list:
            self.function_list.append(tool_name)

            try:
                tool_class_with_tenant = TOOL_REGISTRY[tool_name]

                # adapt the TOOL_REGISTRY[tool_name] to origin tool class
                if (isinstance(tool_class_with_tenant, type) and issubclass(
                        tool_class_with_tenant, BaseTool)) or isinstance(
                            tool_class_with_tenant, BaseTool):
                    tool_class_with_tenant = {
                        'class': TOOL_REGISTRY[tool_name]
                    }
                    TOOL_REGISTRY[tool_name] = tool_class_with_tenant

            except KeyError as e:
                print(e)
                if not self.use_tool_api:
                    raise KeyError(
                        f'Tool {tool_name} is not registered in TOOL_REGISTRY, please register it first.'
                    )
                tool_class_with_tenant = {'class': ToolServiceProxy}

            # check if the tenant_id of tool instance or tool service are exists
            # TODO: change from use_tool_api=True to False, to get the tenant_id of the tool changes to

            if tenant_id in tool_class_with_tenant and self.use_tool_api:
                return

            try:
                if self.use_tool_api:
                    # get service proxy as tool instance, call method will call remote tool service
                    tool_instance = ToolServiceProxy(tool_name, tool_cfg,
                                                     tenant_id, **kwargs)

                    # if the tool name is running in studio, remove the studio prefix from tool name
                    # TODO: it might cause duplicated name from different studio
                    in_ms_studio = os.getenv('MODELSCOPE_ENVIRONMENT', 'none')
                    if in_ms_studio == 'studio':
                        tool_name = tool_name.split('/')[-1]

                else:
                    # instantiation tool class as tool instance
                    tool_instance = TOOL_REGISTRY[tool_name]['class'](tool_cfg)

                self.function_map[tool_name] = tool_instance

            except TypeError:
                # When using OpenAPI, tool_class is already an instantiated object, not a corresponding class
                self.function_map[tool_name] = TOOL_REGISTRY[tool_name][
                    'class']
            except Exception as e:
                raise RuntimeError(e)

            # store the instance of tenant to tool registry on this tool
            tool_class_with_tenant[tenant_id] = self.function_map[tool_name]

    def _detect_tool(self,
                     message: Union[str, dict]) -> Tuple[bool, str, list, str]:
        """
        A built-in tool call detection for func_call format

        Args:
            message: one message
                (1) When dict: Determine whether to call the tool through the function call format.
                (2) When str: The tool needs to be parsed from the string, and at this point, the agent subclass needs
                              to implement a custom _detect_tool function.

        Returns:
            - bool: need to call tool or not
            - list: tool list
            - str: text replies except for tool calls
        """
        # TODO: check

        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text
