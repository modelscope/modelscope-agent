import os
from typing import List, Optional

import json
import requests
from pydantic import BaseModel, ValidationError
from requests.exceptions import RequestException, Timeout

MODELSCOPE_API_TOKEN = os.getenv('MODELSCOPE_API_TOKEN')

MAX_RETRY_TIMES = 3


class ParametersSchema(BaseModel):
    name: str
    description: str
    required: Optional[bool] = True


class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: List[ParametersSchema]


class Tool:
    """
    a base class for tools.
    when you inherit this class and implement new tool, you should provide name, description
    and parameters of tool that conforms with schema.

    each tool may have two call method: _local_call(execute tool in your local environment)
    and _remote_call(construct a http request to remote server).
    corresponding to preprocess and postprocess method may need to be overrided to get correct result.
    """
    name: str = 'tool'
    description: str = 'This is a tool that ...'
    parameters: list = []

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})
        self.is_remote_tool = self.cfg.get('is_remote_tool', False)

        # remote call
        self.url = self.cfg.get('url', '')
        self.token = self.cfg.get('token', '')
        self.header = {
            'Authorization': self.token or f'Bearer {MODELSCOPE_API_TOKEN}'
        }

        try:
            all_para = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_para)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.json(ensure_ascii=False)

    def __call__(self, remote=False, *args, **kwargs):
        if self.is_remote_tool or remote:
            return self._remote_call(*args, **kwargs)
        else:
            return self._local_call(*args, **kwargs)

    def _remote_call(self, *args, **kwargs):
        if self.url == '':
            raise ValueError(
                f"Could not use remote call for {self.name} since this tool doesn't have a remote endpoint"
            )

        remote_parsed_input = json.dumps(
            self._remote_parse_input(*args, **kwargs))

        origin_result = None
        retry_times = MAX_RETRY_TIMES
        while retry_times:
            retry_times -= 1
            try:
                response = requests.request(
                    'POST',
                    self.url,
                    headers=self.header,
                    data=remote_parsed_input)
                if response.status_code != requests.codes.ok:
                    response.raise_for_status()

                origin_result = json.loads(
                    response.content.decode('utf-8'))['Data']

                final_result = self._parse_output(origin_result, remote=True)
                return final_result
            except Timeout:
                continue
            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                    error message: {e.response.content.decode("utf-8")}')

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def _local_call(self, *args, **kwargs):
        return

    def _remote_parse_input(self, *args, **kwargs):
        return kwargs

    def _local_parse_input(self, *args, **kwargs):
        return args, kwargs

    def _parse_output(self, origin_result, *args, **kwargs):
        return origin_result

    def __str__(self):
        return self._str
