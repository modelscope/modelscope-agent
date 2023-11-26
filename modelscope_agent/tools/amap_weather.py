import os

import pandas as pd
import requests
from modelscope_agent.tools.tool import Tool, ToolSchema
from pydantic import ValidationError


class AMAPWeather(Tool):
    description = '获取对应城市的天气数据'
    name = 'amap_weather'
    parameters: list = [{
        'name': 'location',
        'description': 'get temperature for a specific location',
        'required': True
    }]

    def __init__(self, cfg={}):
        self.cfg = cfg.get(self.name, {})

        # remote call
        self.url = 'https://restapi.amap.com/v3/weather/weatherInfo?city={city}&key={key}'
        self.token = self.cfg.get('token', os.environ.get('AMAP_TOKEN', ''))
        self.city_df = pd.read_excel(
            'https://modelscope.oss-cn-beijing.aliyuncs.com/resource/agent/AMap_adcode_citycode.xlsx'
        )
        assert self.token != '', 'weather api token must be acquired through ' \
            'https://lbs.amap.com/api/webservice/guide/create-project/get-key and set by AMAP_TOKEN'

        try:
            all_param = {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters
            }
            self.tool_schema = ToolSchema(**all_param)
        except ValidationError:
            raise ValueError(f'Error when parsing parameters of {self.name}')

        self._str = self.tool_schema.model_dump_json()
        self._function = self.parse_pydantic_model_to_openai_function(
            all_param)

    def get_city_adcode(self, city_name):
        filtered_df = self.city_df[self.city_df['中文名'] == city_name]
        if len(filtered_df['adcode'].values) == 0:
            raise ValueError(
                f'location {city_name} not found, availables are {self.city_df["中文名"]}'
            )
        else:
            return filtered_df['adcode'].values[0]

    def __call__(self, *args, **kwargs):
        location = kwargs['location']
        response = requests.get(
            self.url.format(
                city=self.get_city_adcode(location), key=self.token))
        data = response.json()
        if data['status'] == '0':
            raise RuntimeError(data)
        else:
            weather = data['lives'][0]['weather']
            temperature = data['lives'][0]['temperature']
            return {'result': f'{location}的天气是{weather}温度是{temperature}度。'}
