import re
from typing import Dict, Tuple

import json
from modelscope_agent import action_parser_register
from modelscope_agent.agent_types import AgentType


class ActionParser:
    """Output parser for llm response
    """

    def parse_response(self, response):
        raise NotImplementedError

    # use to handle the case of false parsing the action_para result, if there is no valid action then
    # throw Error
    @staticmethod
    def handle_fallback(action: str, action_para: str):
        if action is not None and action != '':
            parameters = {'fallback': action_para}
            return action, parameters
        else:
            raise ValueError('Wrong response format for action parser')


@action_parser_register
class MsActionParser(ActionParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if '<|startofthink|>' not in response or '<|endofthink|>' not in response:
            return None, None

        action, parameters = '', ''
        try:
            # use regular expression to get result
            re_pattern1 = re.compile(
                pattern=r'<\|startofthink\|>([\s\S]+)<\|endofthink\|>')
            think_content = re_pattern1.search(response).group(1)

            re_pattern2 = re.compile(r'{[\s\S]+}')
            think_content = re_pattern2.search(think_content).group()

            json_content = json.loads(think_content.replace('\n', ''))
            action = json_content.get('api_name',
                                      json_content.get('name', 'unknown'))
            parameters = json_content.get('parameters', {})

            return action, parameters
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(action, parameters)


@action_parser_register
class ChatGLMActionParser(ActionParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """
        if 'tool_call' not in response:
            return None, None
        action, action_para = '', ''
        try:
            # use regular expression to get result from MRKL format
            re_pattern1 = re.compile(
                pattern=r'([\s\S]+)```([\s\S]+)tool_call\(([\s\S]+)```')
            res = re_pattern1.search(response)
            action_list = re.split('<|>|\|', res.group(1).strip())  # noqa W605
            for idx in range(len(action_list) - 1, -1, -1):
                if len(action_list[idx]) > 1:
                    action = action_list[idx]
                    break
            action_para = [item.strip() for item in res.group(3).split(',')]
            parameters = {}
            re_pattern2 = re.compile(pattern=r'([\s\S]+)=\'([\s\S]+)\'')
            for para in action_para:
                res = re_pattern2.search(para)
                parameters[res.group(1)] = res.group(2)
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(action, action_para)

        print(f'\n\naction: {action}\n parameters: {parameters}\n\n')
        return action, parameters


@action_parser_register
class MRKLActionParser(ActionParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'Action' not in response or 'Action Input:' not in response:
            return None, None
        action, action_para = '', ''
        try:
            # use regular expression to get result from MRKL format
            re_pattern1 = re.compile(
                pattern=r'Action:([\s\S]+)Action Input:([\s\S]+)')
            res = re_pattern1.search(response)
            action = res.group(1).strip()
            action_para = res.group(2)

            parameters = json.loads(action_para.replace('\n', ''))

            return action, parameters
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(action, action_para)


@action_parser_register
class OpenAiFunctionsActionParser(ActionParser):

    def parse_response(self, response: dict) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters


        Args:
            response (str): llm response, it should be an openai response message
            such as
            {
                "content": null,
                "function_call": {
                  "arguments": "{\n  \"location\": \"Boston, MA\"\n}",
                  "name": "get_current_weather"
                },
                "role": "assistant"
            }
        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'function_call' not in response or response['function_call'] == {}:
            return None, None
        function_call = response['function_call']

        try:
            # parse directly
            action = function_call['name']
            arguments = json.loads(function_call['arguments'].replace(
                '\n', ''))

            return action, arguments
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return ActionParser.handle_fallback(function_call['name'],
                                                function_call['arguments'])
