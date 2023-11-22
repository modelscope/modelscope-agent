import os
import re
import traceback

import appdirs
import json

from .code_interpreters.create_code_interpreter import create_code_interpreter
from .code_interpreters.language_map import language_map
from .code_interpreters.truncate_output import truncate_output
from .tool import Tool


class CodeInterpreter(Tool):
    """
        using open interpreter to interpret code
        by https://github.com/KillianLucas/open-interpreter
    """
    description = 'Executes code on the user\'s machine, **in the users local environment**, and returns the output'
    name = 'code_interpreter'
    parameters: list = [{
        'name': 'language',
        'description':
        'The programming language (required parameter to the `execute` function)',
        'required': True
    }, {
        'name': 'code',
        'description': 'The code to execute (required)',
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.create_code_interpreter = create_code_interpreter
        self.language_map = language_map
        self.truncate_output = truncate_output

        self._code_interpreters = {}
        self.max_output = self.cfg.get('max_output', 2000)

    def _local_call(self, *args, **kwargs):

        language, code = self._handle_input_fallback(**kwargs)

        try:
            # Fix a common error where the LLM thinks it's in a Jupyter notebook
            if language == 'python' and code.startswith('!'):
                code = code[1:]
                language = 'shell'

            if language in self.language_map:
                if language not in self._code_interpreters:
                    self._code_interpreters[
                        language] = self.create_code_interpreter({'language': language, 'vision': '1'})
                code_interpreter = self._code_interpreters[language]
            else:
                # This still prints code but don't allow code to run. Let Open-Interpreter know through output message
                error_output = f'Error: Open Interpreter does not currently support {language}.'
                print(error_output)
                output = '\n' + error_output
                return {'result': output.strip()}

            output = ''
            for line in code_interpreter.run(code):
                if 'output' in line:
                    output += '\n' + line['output']

                    # Truncate output
                    output = self.truncate_output(output, self.max_output)
        except Exception as e:
            error = traceback.format_exc()
            output = ' '.join(f'{key}:{value}'
                              for key, value in kwargs.items())
            output += f'\nDetail error is {e}.\n{error}'

        return {'result': output.strip()}

    def _handle_input_fallback(self, **kwargs):
        """
        an alternative method is to parse code in content not from function call
        such as:
            text = response['content']
            code_block = re.search(r'```([\s\S]+)```', text)  # noqa W^05
            if code_block:
                result = code_block.group(1)
                language = result.split('\n')[0]
                code = '\n'.join(result.split('\n')[1:])

        :param fallback_text:
        :return: language, cocde
        """

        language = kwargs.get('language', None)
        code = kwargs.get('code', None)
        fallback = kwargs.get('fallback', None)

        if language and code:
            return language, code
        elif fallback:
            try:
                text = fallback
                code_block = re.search(r'```([\s\S]+)```', text)  # noqa W^05
                if code_block:
                    result = code_block.group(1)
                    language = result.split('\n')[0]
                    if language == 'py' or language == 'python':
                        # handle py case
                        # ```py code ```
                        language = 'python'
                        code = '\n'.join(result.split('\n')[1:])
                        return language, code

                    if language == 'json':
                        # handle json case
                        # ```json {language,code}```
                        parameters = json.loads('\n'.join(
                            result.split('\n')[1:]).replace('\n', ''))
                        return parameters['language'], parameters['code']
            except ValueError:
                return language, code
        else:
            return language, code
