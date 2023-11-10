import re
from typing import Dict, Tuple

import json
from modelscope_agent.agent_types import AgentType


def get_output_parser(agent_type: AgentType = AgentType.DEFAULT):
    if AgentType.DEFAULT == agent_type or agent_type == AgentType.MS_AGENT:
        return MsOutputParser()
    elif AgentType.MRKL == agent_type:
        return MRKLOutputParser()
    elif AgentType.OPENAI_FUNCTIONS == agent_type:
        return OpenAiFunctionsOutputParser()
    else:
        raise NotImplementedError


class OutputParser:
    """Output parser for llm response
    """

    def parse_response(self, response):
        raise NotImplementedError

    # use to handle false parsing the action_para result, if no action return then
    # throw Error
    def handle_fallback(self, action: str, action_para: str):
        if action is not None:
            parameters = {'fallback': action_para}
            return action, parameters
        else:
            raise ValueError('Wrong response format for output parser')


class MsOutputParser(OutputParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if '<|startofthink|>' not in response or '<|endofthink|>' not in response:
            return None, None
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
            return self.handle_fallback(action, parameters)


class MRKLOutputParser(OutputParser):

    def parse_response(self, response: str) -> Tuple[str, Dict]:
        """parse response of llm to get tool name and parameters

        Args:
            response (str): llm response, it should conform to some predefined format

        Returns:
            tuple[str, dict]: tuple of tool name and parameters
        """

        if 'Action' not in response or 'Action Input:' not in response:
            return None, None
        try:
            # use regular expression to get result
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
            return self.handle_fallback(action, action_para)


class OpenAiFunctionsOutputParser(OutputParser):

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
        try:
            # use regular expression to get result
            function_call = response['function_call']
            action = function_call['name']
            arguments = json.loads(function_call['arguments'].replace(
                '\n', ''))

            return action, arguments
        except Exception as e:
            print(
                f'Error during parse action might be handled with detail {e}')
            return self.handle_fallback(action, arguments)
