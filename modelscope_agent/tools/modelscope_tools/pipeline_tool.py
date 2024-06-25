import os
from typing import Dict, Optional

import json
import requests
from modelscope_agent.constants import ApiNames
from modelscope_agent.tools.base import BaseTool
from modelscope_agent.utils.utils import get_api_key
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3


class ModelscopePipelineTool(BaseTool):
    default_model: str = ''
    task: str = ''
    model_revision = None

    def __init__(self, cfg: Optional[Dict] = {}):
        """
        初始化一个ModelscopePipelineTool类
        Initialize a ModelscopePipelineTool class.

        Args:
            cfg: (Dict[str, object]) 配置字典，包含了初始化对象所需要的参数
        """
        super().__init__(cfg)

        self.api_url = self.cfg.get('url', self.url)
        # local call should be used by only cfg

        self.use_local = not self.cfg.get('is_remote_tool', True)
        self.is_initialized = False

        if self.use_local:
            self.model = self.cfg.get('model', None) or self.default_model
            self.model_revision = self.cfg.get('model_revision',
                                               None) or self.model_revision

            self.pipeline_params = self.cfg.get('pipeline_params', {})
            self.pipeline = None
            self.is_initialized = False

    def setup(self):
        from modelscope.pipelines import pipeline

        # only initialize when this tool is really called to save memory
        if not self.is_initialized:
            self.pipeline = pipeline(
                task=self.task,
                model=self.model,
                model_revision=self.model_revision,
                **self.pipeline_params)
        self.is_initialized = True

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        if self.use_local:
            return self._local_call(params, **kwargs)
        else:
            return self._remote_call(params, **kwargs)

    def _remote_call(self, params: dict, **kwargs):
        data = json.dumps(params, ensure_ascii=False)
        try:
            api_token = get_api_key(ApiNames.modelscope_api_key, **kwargs)
        except AssertionError:
            raise ValueError('Please set valid MODELSCOPE_API_TOKEN!')

        headers = {'Authorization': f'Bearer {api_token}'}
        retry_times = MAX_RETRY_TIMES

        while retry_times:
            retry_times -= 1
            try:
                response = requests.request(
                    'POST', self.api_url, headers=headers, data=data)
                if response.status_code != requests.codes.ok:
                    response.raise_for_status()

                return json.loads(response.content.decode('utf-8'))

            except Timeout:
                continue

            except RequestException as e:
                raise ValueError(
                    f'Remote call failed with error code: {e.response.status_code},\
                           error message: {e.response.content.decode("utf-8")}'
                )

        raise ValueError(
            'Remote call max retry times exceeded! Please try to use local call.'
        )

    def _local_call(self, params: dict, **kwargs):
        kwargs.update(**params)
        try:
            self.setup()
            origin_result = self.pipeline(**kwargs)
            return json.dumps(origin_result, default=str, ensure_ascii=False)
        except RuntimeError as e:
            import traceback
            raise RuntimeError(
                f'Local call failed with error: {e}, with detail {traceback.format_exc()}'
            )
