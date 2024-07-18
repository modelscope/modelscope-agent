import asyncio
import atexit
import base64
import glob
import io
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import json
import json5
import matplotlib
import nbformat
import PIL.Image
from jupyter_client import BlockingKernelClient
from modelscope_agent.tools.base import BaseTool, register_tool
from modelscope_agent.utils.utils import extract_code
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError, DeadKernelError
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')

STATIC_URL = os.getenv('CODE_INTERPRETER_STATIC_URL',
                       'http://127.0.0.1:7866/static')

LAUNCH_KERNEL_PY = """
from ipykernel import kernelapp as app
app.launch_new_instance()
"""

INIT_CODE_FILE = str(
    Path(__file__).absolute().parent / 'code_interpreter_init_kernel.py')

ALIB_FONT_FILE = str(
    Path(__file__).absolute().parent / 'AlibabaPuHuiTi-3-45-Light.ttf')

_KERNEL_CLIENTS: Dict[int, BlockingKernelClient] = {}


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
        self.image_server = self.cfg.get('image_server', False)
        self.kernel_clients: Dict[int, BlockingKernelClient] = {}
        atexit.register(self._kill_kernels)
        # pid: int = os.getpid()
        pid = random.randint(1, 9999999)
        if pid in self.kernel_clients:
            kc = self.kernel_clients[pid]
        else:
            self._fix_matplotlib_cjk_font_issue()
            kc = self._start_kernel(pid)
            with open(INIT_CODE_FILE) as fin:
                start_code = fin.read()
                start_code = start_code.replace('{{M6_FONT_PATH}}',
                                                repr(ALIB_FONT_FILE)[1:-1])
            print(self._execute_code(kc, start_code))
            self.kernel_clients[pid] = kc
        self.timeout = 300
        self.nb = nbformat.v4.new_notebook()  # noqa E501
        self.nb_client = NotebookClient(self.nb, timeout=300)
        self.console = Console()
        self.interaction = ''
        # timeout: int = 600
        self.kc = kc

    def __del__(self):
        # make sure all the kernels are killed during __del__
        signal.signal(signal.SIGTERM, self._kill_kernels)
        signal.signal(signal.SIGINT, self._kill_kernels)

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
                      keep_len: int = 2000) -> Tuple[bool, str]:
        """Parses the outputs received from notebook execution."""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_text = ''
            if output['output_type'] == 'stream' and not any(
                    tag in output['text'] for tag in [
                        '| INFO     | metagpt', '| ERROR    | metagpt',
                        '| WARNING  | metagpt', 'DEBUG'
                    ]):
                output_text = output['text']
            elif output['output_type'] == 'display_data':
                if 'image/png' in output['data']:
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
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def is_ipython(self) -> bool:
        try:
            # 如果在Jupyter Notebook中运行，__file__ 变量不存在
            from IPython import get_ipython

            if get_ipython() is not None and 'IPKernelApp' in get_ipython(
            ).config:
                return True
            else:
                return False
        except NameError:
            return False

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

            if '!pip' in code:
                success = False

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

    def _start_kernel(self, pid) -> BlockingKernelClient:
        connection_file = os.path.join(WORK_DIR,
                                       f'kernel_connection_file_{pid}.json')
        launch_kernel_script = os.path.join(WORK_DIR,
                                            f'launch_kernel_{pid}.py')
        for f in [connection_file, launch_kernel_script]:
            if os.path.exists(f):
                print(f'WARNING: {f} already exists')
                shutil.rmtree(f, ignore_errors=True)

        os.makedirs(WORK_DIR, exist_ok=True)

        with open(launch_kernel_script, 'w') as fout:
            fout.write(LAUNCH_KERNEL_PY)

        available_envs = ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']
        envs = {}
        for k in available_envs:
            if os.getenv(k) is not None:
                envs[k] = os.getenv(k)

        args = (
            sys.executable,
            launch_kernel_script,
            '--IPKernelApp.connection_file',
            connection_file,
            '--matplotlib=inline',
            '--quiet',
        )
        kernel_process = subprocess.Popen([*args], env=envs,
                                          cwd=WORK_DIR)  # noqa E126
        print(f"INFO: kernel process's PID = {kernel_process.pid}")

        # Wait for kernel connection file to be written
        max_retry = 10
        try_times = 0
        while True:
            if not os.path.isfile(connection_file):
                time.sleep(0.1)
                if try_times > 0:
                    try_times += 1
            else:
                # Keep looping if JSON parsing fails, file may be partially written
                try:
                    with open(connection_file, 'r') as fp:
                        json.load(fp)
                    break
                except json.JSONDecodeError:
                    pass
                except FileNotFoundError:
                    # handle the situation that pass in line 120, while fail in 127
                    try_times += 1
                    pass
            if try_times >= max_retry:
                raise (
                    f'kernel process PID {kernel_process.pid} s config json {connection_file}'
                    f'has been deleted by other process. please try again.')

        # Client
        kc = BlockingKernelClient(connection_file=connection_file)
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        kc.load_connection_file()
        kc.start_channels()
        kc.wait_for_ready()
        return kc

    def _kill_kernels(self):
        for v in self.kernel_clients.values():
            v.shutdown()
        for k in list(self.kernel_clients.keys()):
            del self.kernel_clients[k]

    def _serve_image(self, image_base64: str, image_type: str) -> str:
        image_file = f'{uuid.uuid4()}.{image_type}'
        local_image_file = os.path.join(WORK_DIR, image_file)

        png_bytes = base64.b64decode(image_base64)
        assert isinstance(png_bytes, bytes)

        if image_type == 'gif':
            with open(local_image_file, 'wb') as file:
                file.write(png_bytes)
        else:
            bytes_io = io.BytesIO(png_bytes)
            PIL.Image.open(bytes_io).save(local_image_file, image_type)

        if self.image_server:
            image_url = f'{STATIC_URL}/{image_file}'
            return image_url
        else:
            return local_image_file

    def _escape_ansi(self, line: str) -> str:
        ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', line)

    def _fix_matplotlib_cjk_font_issue(self):
        ttf_name = os.path.basename(ALIB_FONT_FILE)
        local_ttf = os.path.join(
            os.path.abspath(
                os.path.join(matplotlib.matplotlib_fname(), os.path.pardir)),
            'fonts', 'ttf', ttf_name)
        if not os.path.exists(local_ttf):
            try:
                shutil.copy(ALIB_FONT_FILE, local_ttf)
                font_list_cache = os.path.join(matplotlib.get_cachedir(),
                                               'fontlist-*.json')
                for cache_file in glob.glob(font_list_cache):
                    with open(cache_file) as fin:
                        cache_content = fin.read()
                    if ttf_name not in cache_content:
                        os.remove(cache_file)
            except Exception:
                traceback.format_exc()

    def _execute_code(self, kc: BlockingKernelClient, code: str) -> str:
        kc.wait_for_ready()
        kc.execute(code)
        result = ''
        image_idx = 0
        # video ready *.mp4
        re_pattern = re.compile(pattern=r'([\s\S]+)video ready ([\s\S]+).mp4')
        while True:
            text = ''
            image = ''
            finished = False
            msg_type = 'error'
            try:
                msg = kc.get_iopub_msg()
                msg_type = msg['msg_type']
                if msg_type == 'status':
                    if msg['content'].get('execution_state') == 'idle':
                        finished = True
                elif msg_type == 'execute_result':
                    text = msg['content']['data'].get('text/plain', '')
                    if 'image/png' in msg['content']['data']:
                        image_b64 = msg['content']['data']['image/png']
                        image_url = self._serve_image(image_b64, 'png')
                        image_idx += 1
                        image = '![IMAGEGEN](%s)' % (image_url)
                    elif 'text/html' in msg['content']['data']:
                        text += '\n' + msg['content']['data']['text/html']
                    elif 'image/gif' in msg['content']['data']:
                        image_b64 = msg['content']['data']['image/gif']
                        image_url = self._serve_image(image_b64, 'gif')
                        image_idx += 1
                        image = '![IMAGEGEN](%s)' % (image_url)
                elif msg_type == 'display_data':
                    if 'image/png' in msg['content']['data']:
                        image_b64 = msg['content']['data']['image/png']
                        image_url = self._serve_image(image_b64, 'png')
                        image_idx += 1
                        image = '![IMAGEGEN](%s)' % (image_url)
                    else:
                        text = msg['content']['data'].get('text/plain', '')
                elif msg_type == 'stream':
                    res = re_pattern.search(msg['content']['text'])
                    repr = ''
                    if res:
                        path = os.path.join(WORK_DIR, res.group(2) + '.mp4')
                        repr = f'<audio src="{path}"/>'
                    msg_type = msg['content']['name']  # stdout, stderr
                    text = msg['content']['text'] + repr
                elif msg_type == 'error':
                    text = self._escape_ansi('\n'.join(
                        msg['content']['traceback']))
                    if 'M6_CODE_INTERPRETER_TIMEOUT' in text:
                        text = 'Timeout: Code execution exceeded the time limit.'
            except queue.Empty:
                text = 'Timeout: Code execution exceeded the time limit.'
                finished = True
            except Exception:
                text = 'The code interpreter encountered an unexpected error.'
                traceback.format_exc()
                finished = True
            if text:
                result += f'\n{text}'
            if image:
                result += f'\n\n{image}'
            if finished:
                break
        result = result.lstrip('\n')
        if not result:
            result += 'The code executed successfully.'
        return result

    def call(self,
             params: str,
             timeout: Optional[int] = 30,
             nb_mode: bool = False,
             **kwargs) -> str:
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
                    'plt.rcParams["font.family"] = _m6_font_prop.get_name()')
        fixed_code = '\n'.join(fixed_code)
        if nb_mode:
            result, success = self.run(code=fixed_code)
            return result if success else 'Error: ' + result
            pass

        else:
            result = self._execute_code(self.kc, fixed_code)

            # if timeout:
            #     self._execute_code(self.kc, '_M6CountdownTimer.cancel()')

            return result

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
