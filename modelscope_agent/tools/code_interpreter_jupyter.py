import asyncio
import atexit
import base64
import glob
import io
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional

import json
import matplotlib
import PIL.Image
from jupyter_client import BlockingKernelClient

from .tool import Tool

WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')

STATIC_URL = os.getenv('CODE_INTERPRETER_STATIC_URL',
                       'http://127.0.0.1:7866/static')

LAUNCH_KERNEL_PY = """
from ipykernel import kernelapp as app
app.launch_new_instance()
"""

INIT_CODE_FILE = str(
    Path(__file__).absolute().parent / 'code_interpreter_utils'
    / 'code_interpreter_init_kernel.py')

ALIB_FONT_FILE = str(
    Path(__file__).absolute().parent / 'code_interpreter_utils'
    / 'AlibabaPuHuiTi-3-45-Light.ttf')

_KERNEL_CLIENTS: Dict[int, BlockingKernelClient] = {}


class CodeInterpreterJupyter(Tool):
    """
        using jupyter kernel client to interpret python code,
        should not be used the other code interpreter tool at the same time
    """
    description = '代码解释器，可用于执行Python代码。'
    name = 'code_interpreter'
    parameters: list = [{
        'name': 'code',
        'description': '待执行的代码',
        'required': True
    }]

    def __init__(self, cfg={}):
        super().__init__(cfg)
        self.timeout = self.cfg.get('timeout', 30)
        self.image_server = self.cfg.get('image_server', False)
        self.kernel_clients: Dict[int, BlockingKernelClient] = {}
        atexit.register(self._kill_kernels)

        pid: int = os.getpid()
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

        self.kc = kc

    # def __del__(self):
    #     # make sure all the kernels are killed before
    #     signal.signal(signal.SIGTERM, self._kill_kernels)
    #     signal.signal(signal.SIGINT, self._kill_kernels)

    def _start_kernel(self, pid) -> BlockingKernelClient:
        connection_file = os.path.join(WORK_DIR,
                                       f'kernel_connection_file_{pid}.json')
        launch_kernel_script = os.path.join(WORK_DIR,
                                            f'launch_kernel_{pid}.py')
        for f in [connection_file, launch_kernel_script]:
            if os.path.exists(f):
                print(f'WARNING: {f} already exists')
                os.remove(f)

        os.makedirs(WORK_DIR, exist_ok=True)

        with open(launch_kernel_script, 'w') as fout:
            fout.write(LAUNCH_KERNEL_PY)

        kernel_process = subprocess.Popen([
            sys.executable,
            launch_kernel_script,
            '--IPKernelApp.connection_file',
            connection_file,
            '--matplotlib=inline',
            '--quiet',
        ],
                                          cwd=WORK_DIR)  # noqa E126
        print(f"INFO: kernel process's PID = {kernel_process.pid}")

        # Wait for kernel connection file to be written
        while True:
            if not os.path.isfile(connection_file):
                time.sleep(0.1)
            else:
                # Keep looping if JSON parsing fails, file may be partially written
                try:
                    with open(connection_file, 'r') as fp:
                        json.load(fp)
                    break
                except json.JSONDecodeError:
                    pass

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

    def _serve_image(self, image_base64: str) -> str:
        image_file = f'{uuid.uuid4()}.png'
        local_image_file = os.path.join(WORK_DIR, image_file)

        png_bytes = base64.b64decode(image_base64)
        assert isinstance(png_bytes, bytes)
        bytes_io = io.BytesIO(png_bytes)
        PIL.Image.open(bytes_io).save(local_image_file, 'png')

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
                        image_url = self._serve_image(image_b64)
                        image_idx += 1
                        image = '![IMAGEGEN](%s)' % (image_url)
                elif msg_type == 'display_data':
                    if 'image/png' in msg['content']['data']:
                        image_b64 = msg['content']['data']['image/png']
                        image_url = self._serve_image(image_b64)
                        image_idx += 1
                        image = '![IMAGEGEN](%s)' % (image_url)
                    else:
                        text = msg['content']['data'].get('text/plain', '')
                elif msg_type == 'stream':
                    msg_type = msg['content']['name']  # stdout, stderr
                    text = msg['content']['text']
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

    def _local_call(self, *args, **kwargs):
        code = self._handle_input_fallback(**kwargs)
        if not code.strip():
            return ''

        if self.timeout:
            code = f'_M6CountdownTimer.start({self.timeout})\n{code}'

        fixed_code = []
        for line in code.split('\n'):
            fixed_code.append(line)
            if line.startswith('sns.set_theme('):
                fixed_code.append(
                    'plt.rcParams["font.family"] = _m6_font_prop.get_name()')
        fixed_code = '\n'.join(fixed_code)
        result = self._execute_code(self.kc, fixed_code)

        if self.timeout:
            self._execute_code(self.kc, '_M6CountdownTimer.cancel()')

        return {'result': result}

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
