import re
import traceback

from interpreter.code_interpreters.create_code_interpreter import \
    create_code_interpreter
from interpreter.code_interpreters.language_map import language_map
from interpreter.utils.truncate_output import truncate_output

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
        self._code_interpreters = {}
        self.max_output = 2000

    def _local_call(self, *args, **kwargs):

        language, code = self._handle_input_fallback(**kwargs)

        try:
            # Fix a common error where the LLM thinks it's in a Jupyter notebook
            if language == 'python' and code.startswith('!'):
                code = code[1:]
                language = 'shell'

            if language in language_map:
                if language not in self._code_interpreters:
                    self._code_interpreters[
                        language] = create_code_interpreter(language)
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
                    output = truncate_output(output, self.max_output)
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
                    code = '\n'.join(result.split('\n')[1:])

                    # handle py case
                    if language == 'py':
                        language = 'python'
                    return language, code
            except ValueError:
                return language, code
        else:
            return language, code
