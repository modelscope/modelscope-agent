import os

import requests
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.rapidapi_tools.basetool_for_alpha_umi import \
    BasetoolAlphaUmi
from requests.exceptions import JSONDecodeError, RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')

RAPID_API_TOKEN = os.getenv('RAPID_API_TOKEN', None)


@register_tool('get_data_fact_for_numbers')
class GetDataFactForNumbers(BasetoolAlphaUmi):
    description = 'Get a fact about a day of year'
    name = 'get_data_fact_for_numbers'
    parameters: list = [{
        'name': 'month',
        'description': 'The 1-indexed month (eg. 6 for June)',
        'required': True,
        'type': 'string'
    }, {
        'name': 'day',
        'description': 'The day of the month',
        'required': True,
        'type': 'string'
    }, {
        'name': 'fragment',
        'description':
        'Add "?fragment=true" to return the fact as a sentence fragment \
            that can be easily included as part of a larger sentence. \
            This means that the first word is lowercase and ending punctuation is omitted. \
            For trivia and math, a noun phrase is returned that can be used in a sentence \
            like “We now have more users than [fact as fragment]!”.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'json',
        'description':
        'Specify "true" to return result as JSON instead of plaintext.',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://numbersapi.p.rapidapi.com/6/21/date'

        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'numbersapi.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to detect_for_google_translate:', kwargs)
        return kwargs


@register_tool('get_math_fact_for_numbers')
class GetMathFactForNumbers(BasetoolAlphaUmi):
    description = 'Get a mathematical property about a number'
    name = 'get_math_fact_for_numbers'
    parameters: list = [{
        'name': 'number',
        'description': 'The integer of interest',
        'required': True,
        'type': 'string'
    }, {
        'name': 'fragment',
        'description':
        'Add "?fragment=true" to return the fact as a sentence fragment that \
            can be easily included as part of a larger sentence. \
            This means that the first word is lowercase and ending punctuation is omitted. \
            For trivia and math, a noun phrase is returned that can be used in a sentence \
            like “We now have more users than [fact as fragment]!”.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'json',
        'description':
        'Specify "true" to return result as JSON instead of plaintext.',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://numbersapi.p.rapidapi.com/1729/math'

        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'numbersapi.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to detect_for_google_translate:', kwargs)
        return kwargs


@register_tool('get_year_fact_for_numbers')
class GetYearFactForNumbers(BasetoolAlphaUmi):
    description = 'Get a fact about a year'
    name = 'get_year_fact_for_numbers'
    parameters: list = [{
        'name': 'year',
        'description': 'The integer of interest',
        'required': True,
        'type': 'string'
    }, {
        'name': 'fragment',
        'description':
        'Add "?fragment=true" to return the fact as a sentence fragment \
            that can be easily included as part of a larger sentence. \
            This means that the first word is lowercase and ending punctuation is omitted. \
            For trivia and math, a noun phrase is returned that can be used in a sentence \
            like “We now have more users than [fact as fragment]!”.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'json',
        'description':
        'Specify "true" to return result as JSON instead of plaintext.',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://numbersapi.p.rapidapi.com/1492/year'

        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'numbersapi.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to detect_for_google_translate:', kwargs)
        return kwargs
