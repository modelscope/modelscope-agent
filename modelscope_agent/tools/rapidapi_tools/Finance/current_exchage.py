import os

import requests
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.rapidapi_tools.basetool_for_alpha_umi import \
    BasetoolAlphaUmi
from requests.exceptions import JSONDecodeError, RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')

RAPID_API_TOKEN = os.getenv('RAPID_API_TOKEN', None)


@register_tool('listquotes_for_current_exchange')
class ListquotesForCurrentExchange(BasetoolAlphaUmi):
    description = 'List the available quotes in JSON Array this API support, \
        all the available quotes can be used in source and destination quote. \
        Refer exchange endpoint for more information how to call the \
        currency exchange from the source quote to destination quote.'

    name = 'listquotes_for_current_exchange'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://currency-exchange.p.rapidapi.com/listquotes'

        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'currency-exchange.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to listquotes_for_current_exchange:', kwargs)
        return kwargs


@register_tool('exchange_for_current_exchange')
class exchange_for_current_exchange(BasetoolAlphaUmi):
    description = 'Get Currency Exchange by specifying the quotes of source (from) and destination (to), \
        and optionally the source amount to calculate which to get the destination amount, \
            by default the source amount will be 1.'

    name = 'exchange_for_current_exchange'
    parameters: list = [{
        'name': 'from',
        'description': 'Source Quote',
        'required': True,
        'type': 'string'
    }, {
        'name': 'to',
        'description': 'Destination Quote',
        'required': True,
        'type': 'string'
    }, {
        'name': 'q',
        'description': 'Source Amount',
        'required': False,
        'type': 'number'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://currency-exchange.p.rapidapi.com/exchange'

        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'currency-exchange.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to listquotes_for_current_exchange:', kwargs)
        return kwargs
