import importlib
import traceback
from copy import deepcopy
from typing import Dict, List, Optional, Union

from .agent_types import AgentType
from .llm import LLM
from .output_parser import OutputParser, get_output_parser
from .output_wrapper import display
from .prompt import PromptGenerator, get_prompt_generator
from .retrieve import KnowledgeRetrieval, ToolRetrieval
from .tools import TOOL_INFO_LIST


class AgentExecutor:

    def __init__(self,
                 llm: LLM,
                 tool_cfg: Optional[Dict] = {},
                 agent_type: AgentType = AgentType.DEFAULT,
                 additional_tool_list: Optional[Dict] = {},
                 prompt_generator: Optional[PromptGenerator] = None,
                 output_parser: Optional[OutputParser] = None,
                 tool_retrieval: Optional[Union[bool, ToolRetrieval]] = True,
                 knowledge_retrieval: Optional[KnowledgeRetrieval] = None):
        """
        the core class of ms agent. It is responsible for the interaction between user, llm and tools,
        and return the execution result to user.

        Args:
            llm (LLM): llm model, can be load from local or a remote server.
            tool_cfg (Optional[Dict]): cfg of default tools
            agent_type (AgentType, optional): agent type. Defaults to AgentType.DEFAULT, decide which type of agent
            reasoning type to use
            additional_tool_list (Optional[Dict], optional): user-defined additional tool list. Defaults to {}.
            prompt_generator (Optional[PromptGenerator], optional): this module is responsible for generating prompt
            according to interaction result. Defaults to use MSPromptGenerator.
            output_parser (Optional[OutputParser], optional): this module is responsible for parsing output of llm
            to executable actions. Defaults to use MsOutputParser.
            tool_retrieval (Optional[Union[bool, ToolRetrieval]], optional): Retrieve related tools by input task,
            since most of the tools may be useless for LLM in specific task.
            If it is bool type and is True, will use default tool_retrieval. Defaults to True.
            knowledge_retrieval (Optional[KnowledgeRetrieval], optional): If user want to use extra knowledge,
            this component can be used to retrieve related knowledge. Defaults to None.
        """

        self.llm = llm

        self.agent_type = agent_type
        self.llm.set_agent_type(agent_type)
        self.prompt_generator = prompt_generator or get_prompt_generator(
            agent_type)
        self.output_parser = output_parser or get_output_parser(agent_type)

        self._init_tools(tool_cfg, additional_tool_list)

        if isinstance(tool_retrieval, bool) and tool_retrieval:
            tool_retrieval = ToolRetrieval()
        self.tool_retrieval = tool_retrieval
        if self.tool_retrieval:
            self.tool_retrieval.construct(
                [str(t) for t in self.tool_list.values()])
        self.knowledge_retrieval = knowledge_retrieval
        self.reset()
        self.seed = None

    def _init_tools(self,
                    tool_cfg: Dict = {},
                    additional_tool_list: Dict = {}):
        """init tool list of agent. We provide a default tool list, which is initialized by a cfg file.
        user can also provide user-defined tools by additional_tool_list.
        The key of additional_tool_list is tool name, and the value is corresponding object.

        Args:
            tool_cfg (Dict): default tool cfg.
            additional_tool_list (Dict, optional): user-defined tools. Defaults to {}.
        """
        self.tool_list = {}
        tool_info_list = {**TOOL_INFO_LIST, **additional_tool_list}
        tools_module = importlib.import_module('modelscope_agent.tools')
        for tool_name in tool_cfg.keys():
            if tool_cfg[tool_name].get('use', False):
                assert tool_name in tool_info_list, f'Invalid tool name: {tool_name}, ' \
                    f'available ones are: {tool_info_list.keys()}'
                tool_class_name = tool_info_list[tool_name]
                tool_class = getattr(tools_module, tool_class_name)
                tool_name = tool_class.name
                self.tool_list[tool_name] = tool_class(tool_cfg)

        self.tool_list = {**self.tool_list, **additional_tool_list}
        # self.available_tool_list = deepcopy(self.tool_list)
        self.set_available_tools(self.tool_list.keys())

    def set_available_tools(self, available_tool_list):
        # TODO @wenmeng.zwm refine tool init
        for t in available_tool_list:
            if t not in self.tool_list:
                raise ValueError(
                    f'Unsupported tools found:{t}, please check, valid ones: {self.tool_list.keys()}'
                )

        self.available_tool_list = {
            k: self.tool_list[k]
            for k in available_tool_list
        }

    def retrieve_tools(self, query: str) -> List[str]:
        """retrieve tools given query

        Args:
            query (str): query

        """
        if self.tool_retrieval:
            retrieve_tools = self.tool_retrieval.retrieve(query)
            self.set_available_tools(available_tool_list=retrieve_tools.keys())
        return self.available_tool_list.values()

    def get_knowledge(self, query: str, append_files: list = []) -> List[str]:
        """retrieve knowledge given query

        Args:
            query (str): query
            append_files(str): user insert append_files during runtime

        """
        if len(append_files) > 0:
            # get the sub list of files only end with .txt, .pdf, .md
            append_files = [
                item for item in append_files
                if item.endswith(('.txt', '.pdf', '.md'))
            ]
            if not self.knowledge_retrieval:
                self.knowledge_retrieval = KnowledgeRetrieval.from_file(
                    append_files)
            else:
                self.knowledge_retrieval.add_file(append_files)
        return self.knowledge_retrieval.retrieve(
            query) if self.knowledge_retrieval else []

    def run(self,
            task: str,
            remote: bool = False,
            print_info: bool = False,
            append_files: list = []) -> List[Dict]:
        """ use llm and tools to execute task given by user

        Args:
            task (str): concrete task
            remote (bool, optional): whether to execute tool in remote mode. Defaults to False.
            print_info (bool, optional): whether to print prompt info. Defaults to False.
            append_files(list): the list of append_files that need to add to knowledge or refered
        Returns:
            List[Dict]: execute result. One task may need to interact with llm multiple times,
            so a list of dict is returned. Each dict contains the result of one interaction.
        """

        # retrieve tools
        tool_list = self.retrieve_tools(task)
        knowledge_list = self.get_knowledge(task, append_files)

        self.prompt_generator.init_prompt(
            task, tool_list, knowledge_list, append_files=append_files)
        function_list = self.prompt_generator.get_function_list(tool_list)

        llm_result, exec_result = '', ''

        idx = 0
        final_res = []

        while True:
            idx += 1

            # generate prompt and call llm
            llm_artifacts = self.prompt_generator.generate(
                llm_result, exec_result)
            try:
                llm_result = self.llm.generate(llm_artifacts, function_list)
            except RuntimeError as e:
                return [{'exec_result': str(e)}]

            if print_info:
                print(f'|LLM inputs in round {idx}: {llm_artifacts}')

            # parse and get tool name and arguments
            try:
                action, action_args = self.output_parser.parse_response(
                    llm_result)
            except ValueError as e:
                return [{'exec_result': f'{e}'}]

            if action is None:
                # in chat mode, the final result of last instructions should be updated to prompt history
                _ = self.prompt_generator.generate(llm_result, '')

                # for summarize
                display(llm_result, {}, idx, self.agent_type)
                return final_res

            if action in self.available_tool_list:
                action_args = self.parse_action_args(action_args)
                tool = self.tool_list[action]

                # TODO @wenmeng.zwm remove this hack logic for image generation
                if action == 'image_gen' and self.seed:
                    action_args['seed'] = self.seed
                try:
                    exec_result = tool(**action_args, remote=remote)
                    if print_info:
                        print(f'|exec_result: {exec_result}')

                    # parse exec result and store result to agent state
                    final_res.append(exec_result)
                    self.parse_exec_result(exec_result)
                except Exception as e:
                    exec_result = f'Action call error: {action}: {action_args}. \n Error message: {e}'
                    return [{'exec_result': exec_result}]
            else:
                exec_result = f"Unknown action: '{action}'. "
                return [{'exec_result': exec_result}]

            # display result
            display(llm_result, exec_result, idx, self.agent_type)

    def stream_run(self,
                   task: str,
                   remote: bool = True,
                   print_info: bool = False,
                   append_files: list = []) -> Dict:
        """this is a stream version of run, which can be used in scenario like gradio.
        It will yield the result of each interaction, so that the caller can display the result

        Args:
            task (str): concrete task
            remote (bool, optional): whether to execute tool in remote mode. Defaults to True.
            print_info (bool, optional): whether to print prompt info. Defaults to False.
            append_files(list of str) files that individually used in each run

        Yields:
            Iterator[Dict]: iterator of llm response and tool execution result
        """

        # retrieve tools
        tool_list = self.retrieve_tools(task)
        knowledge_list = self.get_knowledge(task, append_files)

        self.prompt_generator.init_prompt(
            task,
            tool_list,
            knowledge_list,
            append_files=append_files,
        )
        function_list = self.prompt_generator.get_function_list(tool_list)

        llm_result, exec_result = '', ''

        idx = 0

        while True:
            idx += 1
            llm_artifacts = self.prompt_generator.generate(
                llm_result, exec_result)
            if print_info:
                print(f'|LLM inputs in round {idx}:\n{llm_artifacts}')

            llm_result = ''
            try:
                for s in self.llm.stream_generate(llm_artifacts,
                                                  function_list):
                    llm_result += s
                    yield {'llm_text': s}
            except RuntimeError:
                s = self.llm.generate(llm_artifacts)
                llm_result += s
                yield {'llm_text': s}
            except Exception as e:
                yield {'llm_text': str(e)}

            # parse and get tool name and arguments
            try:
                action, action_args = self.output_parser.parse_response(
                    llm_result)
            except ValueError as e:
                yield {'exec_result': f'{e}'}
                return

            if action is None:
                # in chat mode, the final result of last instructions should be updated to prompt history
                _ = self.prompt_generator.generate(llm_result, '')
                yield {'is_final': True}
                return

            if action in self.available_tool_list:
                # yield observation to as end of action input symbol asap
                yield {'llm_text': 'Observation: '}
                action_args = self.parse_action_args(action_args)
                tool = self.tool_list[action]

                # TODO @wenmeng.zwm remove this hack logic for image generation
                if action == 'image_gen' and self.seed:
                    action_args['seed'] = self.seed
                try:
                    exec_result = tool(**action_args, remote=remote)
                    yield {'exec_result': exec_result}

                    # parse exec result and update state
                    self.parse_exec_result(exec_result)
                except Exception as e:
                    exec_result = f'Action call error: {action}: {action_args}. \n Error message: {e}'
                    yield {'exec_result': exec_result}
                    self.prompt_generator.reset()
                    return
            else:
                exec_result = f"Unknown action: '{action}'. "
                yield {'exec_result': exec_result}
                self.prompt_generator.reset()
                return

    def reset(self):
        """
        clear history and agent state
        """
        self.prompt_generator.reset()
        self.agent_state = {}

    def parse_action_args(self, action_args):
        """
        replace action_args in str to Image/Video/Audio Wrapper, so that tool can handle them
        """
        parsed_action_args = {}
        for name, arg in action_args.items():
            try:
                true_arg = self.agent_state.get(arg, arg)
            except Exception as e:
                print(f'Error when parsing action args: {e}, using fall back')
                true_arg = arg
            parsed_action_args[name] = true_arg
        return parsed_action_args

    def parse_exec_result(self, exec_result, *args, **kwargs):
        """
        update exec result to agent state.
        key is the str representation of the result.
        """
        for k, v in exec_result.items():
            self.agent_state[str(v)] = v
