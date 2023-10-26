import re
from typing import Dict, Tuple

import json
import traceback

class OutputParser:

    def parse_response(self, response):
        raise NotImplementedError


class MsOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

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
        except Exception:
            raise ValueError('Wrong response format for output parser')


class QwenOutputParser(OutputParser):

    def __init__(self):
        super().__init__()

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
        except Exception:
            raise ValueError('Wrong response format for output parser')