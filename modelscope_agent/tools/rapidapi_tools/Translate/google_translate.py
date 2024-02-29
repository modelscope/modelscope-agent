import os

import requests
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.rapidapi_tools.basetool_for_alpha_umi import \
    BasetoolAlphaUmi
from requests.exceptions import JSONDecodeError, RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')

RAPID_API_TOKEN = os.getenv('RAPID_API_TOKEN', None)


@register_tool('detect_for_google_translate')
class DetectForGoogleTranslate(BasetoolAlphaUmi):
    description = 'Detects the language of text within a request.'
    name = 'detect_for_google_translate'
    parameters: list = [{
        'name': 'q',
        'description':
        'The input text upon which to perform language detection. \
            Repeat this parameter to perform language detection on multiple text inputs.',
        'required': True,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://google-translate1.p.rapidapi.com/language/translate/v2/detect'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'Accept-Encoding': 'application/gzip',
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'google-translate1.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except AttributeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to detect_for_google_translate:', kwargs)
        return kwargs


@register_tool('languages_for_google_translate')
class LanguagesForGoogleTranslate(BasetoolAlphaUmi):
    description = 'Returns a list of supported languages for translation.'
    name = 'languages_for_google_translate'
    parameters: list = [{
        'name': 'target',
        'description': 'The target language code for the results. \
            If specified, then the language names are returned in the name field of the response, \
            localized in the target language. \
            If you do not supply a target language, \
            then the name field is omitted from the response and only the language codes are returned.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'model',
        'description': 'The translation model of the supported languages. \
            Can be either base to return languages supported by the Phrase-Based Machine Translation (PBMT) model, \
            or nmt to return languages supported by the Neural Machine Translation (NMT) model. \
            If omitted, then all supported languages are returned.',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://google-translate1.p.rapidapi.com/language/translate/v2/languages'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'Accept-Encoding': 'application/gzip',
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'google-translate1.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to languages_for_google_translate:', kwargs)
        return kwargs


@register_tool('translate_for_google_translate')
class TranslateForGoogleTranslate(BasetoolAlphaUmi):
    description = 'Returns a list of supported languages for translation.'
    name = 'translate_for_google_translate'
    parameters: list = [{
        'name': 'q',
        'description': 'The input text to translate. \
        Repeat this parameter to perform translation operations on multiple text inputs.',
        'required': True,
        'type': 'string'
    }, {
        'name': 'target',
        'description':
        'The language to use for translation of the input text, \
        set to one of the language codes listed in the overview tab',
        'required': True,
        'type': 'string'
    }, {
        'name': 'format',
        'description':
        'The format of the source text, in either HTML (default) or plain-text. \
            A value of html indicates HTML and a value of text indicates plain-text.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'source',
        'description':
        'The language of the source text, set to one of the language codes listed in Language Support. \
            If the source language is not specified, \
            the API will attempt to detect the source language automatically and return it within the response.',
        'required': False,
        'type': 'string'
    }, {
        'name': 'model',
        'description': 'The translation model. \
            Can be either base to use the Phrase-Based Machine Translation (PBMT) model, \
            or nmt to use the Neural Machine Translation (NMT) model. \
            If omitted, then nmt is used. If the model is nmt, \
            and the requested language translation pair is not supported for the NMT model, \
            then the request is translated using the base model.',
        'required': False,
        'type': 'string'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://google-translate1.p.rapidapi.com/language/translate/v2'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'content-type': 'application/x-www-form-urlencoded',
            'Accept-Encoding': 'application/gzip',
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host': 'google-translate1.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print('kwargs passed to languages_for_google_translate:', kwargs)
        return kwargs
