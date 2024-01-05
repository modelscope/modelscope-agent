from modelscope_agent.tools.base import BaseTool, register_tool
import json
import requests
import os

@register_tool('pipeline')
class ModelscopePipelineTool(BaseTool):
    API_URL = ""
    API_KEY = ""

    def __init__(self, cfg):
        """
        初始化一个ModelscopePipelineTool类
        Initialize a ModelscopePipelineTool class.
        参数：
        cfg (Dict[str, object]): 配置字典，包含了初始化对象所需要的参数
        """
        super().__init__(cfg)
        self.API_URL = self.cfg.get(self.name, {}).get('url',None) or self.API_URL
        self.API_KEY = os.getenv('MODELSCOPE_API_KEY', None) or self.API_KEY

    
    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        parsed_kwargs = self._remote_parse_input(**params)
        data = json.dumps(parsed_kwargs)
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        response = requests.request("POST", self.API_URL, headers=headers,data=data)
        origin_result = json.loads(response.content.decode("utf-8"))
        final_result = self._remote_parse_output(origin_result, remote=False)
        return final_result

