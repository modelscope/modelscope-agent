import asyncio
import atexit
import base64
import glob
import io
import os
import re
import time
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import json
import json5
import nbformat
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import extract_code

try:
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError
    from nbformat import NotebookNode
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
    from rich.console import Console
    from rich.syntax import Syntax
except ImportError:
    raise ImportError(
        'Please install nbclient, nbformat, rich by running `pip install nbclient nbformat rich`'
    )


@register_tool('code_interpreter')
class CodeInterpreter(BaseTool):
    """
        using jupyter kernel client to interpret python code,
        should not be used the other code interpreter tool at the same time
    """
    name = 'code_interpreter'
    description = '代码解释器，可用于执行Python代码。'  # noqa E501
    parameters = [{'name': 'code', 'type': 'string', 'description': '待执行的代码'}]

    def __init__(self, cfg={}):
        super().__init__(cfg)

        self.timeout = 180
        self.nb = nbformat.v4.new_notebook()  # noqa E501
        self.nb_client = NotebookClient(self.nb, timeout=180)
        self.console = Console()
        self.interaction = ''
        self.silent_mode = False
        # timeout: int = 600

    def __del__(self):
        # make sure all the kernels are killed during __del__
        self.terminate()

    def build(self):
        if self.nb_client.kc is None or not self.nb_client.kc.is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    def terminate(self):
        """kill NotebookClient"""
        if self.nb_client.km is not None and self.nb_client.km.is_alive():
            self.nb_client.km.shutdown_kernel(now=True)
            self.nb_client.km.cleanup_resources()

            channels = [
                self.nb_client.kc.
                stdin_channel,  # The channel for handling standard input to the kernel.
                self.nb_client.kc.hb_channel,
                # The channel for heartbeat communication between the kernel and client.
                self.nb_client.kc.
                control_channel,  # The channel for controlling the kernel.
            ]
            # Stops all the running channels for this kernel
            for channel in channels:
                if channel.is_alive():
                    channel.stop()

            self.nb_client.kc = None
            self.nb_client.km = None

    def reset(self):
        """reset NotebookClient"""
        self.terminate()
        # sleep 1s to wait for the kernel to be cleaned up completely
        time.sleep(1)
        self.build()
        self.nb_client = NotebookClient(self.nb, timeout=self.timeout)

    def add_code_cell(self, code: str):
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str):
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def _display(self,
                 code: str,
                 language: Literal['python', 'markdown'] = 'python'):
        if language == 'python':
            code = Syntax(
                code, 'python', theme='paraiso-dark', line_numbers=True)
            self.console.print(code)
        # elif language == "markdown":
        #     display_markdown(code)
        else:
            raise ValueError(
                f'Only support for python, markdown, but got {language}')

    def add_output_to_cell(self, cell: NotebookNode, output: str):
        """add outputs of code execution to notebook cell."""
        if 'outputs' not in cell:
            cell['outputs'] = []
        else:
            cell['outputs'].append(
                new_output(
                    output_type='stream', name='stdout', text=str(output)))

    def remove_escape_and_color_codes(input_str: str):
        # 使用正则表达式去除jupyter notebook输出结果中的转义字符和颜色代码
        # Use regular expressions to get rid of escape characters and color codes in jupyter notebook output.
        pattern = re.compile(r'\x1b\[[0-9;]*[mK]')
        result = pattern.sub('', input_str)
        return result

    def parse_outputs(self,
                      outputs: list[str],
                      keep_len: int = 20000) -> Tuple[bool, str]:
        """Parses the outputs received from notebook execution."""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_text = ''
            if output['output_type'] == 'stream' and not any(
                    tag in output['text'] for tag in [
                        '| INFO     | metagpt', '| ERROR    | metagpt',
                        '| WARNING  | metagpt', 'DEBUG', 'FutureWarning'
                    ]) and output['name'] == 'stdout':
                output_text = output['text']
            elif output['output_type'] == 'display_data':
                if 'image/png' in output['data']:
                    if not self.silent_mode:
                        self.show_bytes_figure(output['data']['image/png'],
                                               self.interaction)

            elif output['output_type'] == 'execute_result':
                output_text = output['data']['text/plain']
            elif output['output_type'] == 'error':
                output_text, is_success = '\n'.join(output['traceback']), False
            output_text = output_text[:keep_len] if is_success else output_text[
                -keep_len:]

            parsed_output.append(output_text)
        return is_success, ','.join(parsed_output)

    def show_bytes_figure(self, image_base64: str,
                          interaction_type: Literal['ipython', None]):
        image_bytes = base64.b64decode(image_base64)
        if interaction_type == 'ipython':
            from IPython.display import Image, display

            display(Image(data=image_bytes))
        else:
            import io  # noqa F401

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def run_cell(self, cell: NotebookNode,
                 cell_index: int) -> Tuple[bool, str]:
        """set timeout for run code.
        returns the success or failure of the cell execution, and an optional error message.
        """
        try:
            self.nb_client.execute_cell(cell, cell_index)
            return self.parse_outputs(self.nb.cells[-1].outputs)
        except CellTimeoutError:
            assert self.nb_client.km is not None
            self.nb_client.km.interrupt_kernel()
            asyncio.sleep(1)
            error_msg = 'Cell execution timed out: Execution exceeded the time limit and was stopped;\
             consider optimizing your code for better performance.'

            return False, error_msg
        except DeadKernelError:
            self.reset()
            return False, 'DeadKernelError'
        except Exception:
            return self.parse_outputs(self.nb.cells[-1].outputs)

    def run(self,
            code: str,
            language: Literal['python',
                              'markdown'] = 'python') -> Tuple[str, bool]:
        """
        return the output of code execution, and a success indicator (bool) of code execution.
        """
        self._display(code, language)

        if language == 'python':
            # add code to the notebook
            self.add_code_cell(code=code)

            # build code executor
            self.build()

            # run code
            cell_index = len(self.nb.cells) - 1
            success, outputs = self.run_cell(self.nb.cells[-1], cell_index)

            # if '!pip' in code:
            #     success = False

            return outputs, success

        elif language == 'markdown':
            # add markdown content to markdown cell in a notebook.
            self.add_markdown_cell(code)
            # return True, beacuse there is no execution failure for markdown cell.
            return code, True
        else:
            raise ValueError(
                f'Only support for language: python, markdown, but got {language}, '
            )

    def call(self,
             params: str,
             timeout: Optional[int] = 30,
             nb_mode: bool = False,
             silent_mode: Optional[bool] = False,
             **kwargs) -> (bool, str):
        try:
            try:
                params = json5.loads(params)
                code = params['code']
            except Exception:
                code = extract_code(params)
            if not code.strip():
                return ''

            # if timeout:
            #     code = f'_M6CountdownTimer.start({timeout})\n{code}'

            fixed_code = []
            for line in code.split('\n'):
                fixed_code.append(line)
                if line.startswith('sns.set_theme('):
                    fixed_code.append(
                        'plt.rcParams["font.family"] = _m6_font_prop.get_name()'
                    )
            fixed_code = '\n'.join(fixed_code)
            if nb_mode:
                self.silent_mode = silent_mode
                result, success = self.run(code=fixed_code, )
                return success, result
        except Exception as e:
            return False, str(e)

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

        code = kwargs.get('code', None)
        fallback = kwargs.get('fallback', None)

        if code:
            return code
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
                        return code

                    if language == 'json':
                        # handle json case
                        # ```json {language,code}```
                        parameters = json.loads('\n'.join(
                            result.split('\n')[1:]).replace('\n', ''))
                        return parameters['code']
            except ValueError:
                return code
        else:
            return code


if __name__ == '__main__':
    code_interpreter = CodeInterpreter()
