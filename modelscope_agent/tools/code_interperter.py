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
    name = 'code-interpreter'
    parameters: list = [{
        'name': 'language',
        'description':
        'The programming language (required parameter to the `execute` function)',
        'required': True
    }, {
        'name': 'text',
        'description': 'The code to execute (required)',
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self._code_interpreters = {}
        self.max_output = 2000

    def _local_call(self, *args, **kwargs):
        language = kwargs.get('language')
        code = kwargs.get('code')

        # Fix a common error where the LLM thinks it's in a Jupyter notebook
        if language == 'python' and code.startswith('!'):
            code = code[1:]
            language = 'shell'
        try:
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
            print(e)
            output = traceback.format_exc()
        return {'result': output.strip()}
