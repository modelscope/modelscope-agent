import ctypes
import gc
import os
import platform
import shutil
import threading
import time
import zipfile
from collections import OrderedDict
from typing import Tuple

from builder_core import AgentBuilder, init_builder_chatbot_agent
from config_utils import (get_user_builder_history_dir,
                          get_user_preview_history_dir, parse_configuration)
from modelscope_agent.agents.role_play import RolePlay
from modelscope_agent.memory import MemoryWithRetrievalKnowledge
from server_logging import logger
from user_core import init_user_chatbot_agent

STATIC_FOLDER = 'statics'

IMPORT_ZIP_TEMP_DIR = '/tmp/import_zip/'


def static_file(source_path):
    file_name = os.path.basename(source_path)
    target_path = os.path.join(STATIC_FOLDER, file_name)
    shutil.move(source_path, target_path)
    return file_name


def unzip_with_folder(zip_filepath):
    dest_path = os.path.join(os.path.dirname(zip_filepath), 'archive')
    shutil.rmtree(dest_path, ignore_errors=True)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)

    files = os.listdir(dest_path)
    if len(files) == 1 and os.path.isdir(os.path.join(dest_path, files[0])):
        inner_dir = os.path.join(dest_path, files[0])
        for file in os.listdir(inner_dir):
            shutil.move(os.path.join(inner_dir, file), dest_path)
        os.rmdir(inner_dir)
    return dest_path


class ExpiringDict(OrderedDict):

    def __init__(self, max_age, cleanup_interval):
        self.max_age = max_age
        self.cleanup_interval = cleanup_interval
        self.last_access = OrderedDict()
        self.lock = threading.Lock()
        super().__init__()
        self._start_cleanup_thread()

    def __setitem__(self, key, value):
        with self.lock:
            current_time = time.time()
            self.last_access[key] = current_time
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self.lock:
            current_time = time.time()
            if key in self.last_access:
                self.last_access[key] = current_time
                return super().__getitem__(key)

    def delete_key(self, key):
        with self.lock:
            if key in self.last_access:
                del self[key]
                del self.last_access[key]
                gc.collect()
                if platform.uname()[0] != 'Darwin':
                    libc = ctypes.cdll.LoadLibrary('libc.{}'.format('so.6'))
                    libc.malloc_trim(0)
                logger.info(f'Done deleting the key {key}')

    def _start_cleanup_thread(self):
        self.cleanup_thread = threading.Timer(self.cleanup_interval,
                                              self._cleanup)
        self.cleanup_thread.daemon = True  # 设置为守护线程，确保主程序退出时线程也会退出
        self.cleanup_thread.start()

    def _cleanup(self):
        with self.lock:
            current_time = time.time()
            keys_and_age = {
                key: current_time - last_time
                for key, last_time in self.last_access.items()
            }
            keys_to_delete = [
                key for key, age in keys_and_age.items() if age >= self.max_age
            ]
            for key in keys_to_delete:
                del self[key]
                del self.last_access[key]
            logger.info(
                f'expiring_dict_clean_up: keys_and_age {keys_and_age}, keys_to_delete {keys_to_delete}, '
                f'remaining keys {self.keys()}')

        # 重新启动定时器
        self._start_cleanup_thread()

    def stop_cleanup(self):
        self.cleanup_thread.cancel()  # 取消后台清理线程


# 简单进行内存级别的会话管理
class SessionManager:

    def __init__(self):
        self.builder_bots = ExpiringDict(max_age=3600, cleanup_interval=60)
        self.user_bots = ExpiringDict(max_age=3600, cleanup_interval=60)

    def get_builder_bot(
            self,
            builder_id,
            renew=False) -> Tuple[AgentBuilder, MemoryWithRetrievalKnowledge]:
        builder_agent = self.builder_bots[builder_id]
        if renew or builder_agent is None:
            logger.info(f'init_builder_chatbot_agent: {builder_id} ')
            builder_agent = init_builder_chatbot_agent(builder_id)
            self.builder_bots[builder_id] = builder_agent
        return builder_agent

    def clear_builder_bot(self, builder_id):
        builder_agent = self.builder_bots[builder_id]
        if builder_agent is not None:
            self.builder_bots.delete_key(builder_id)
        shutil.rmtree(
            get_user_builder_history_dir(builder_id), ignore_errors=True)

    def get_user_bot(
            self,
            builder_id,
            session,
            renew=False,
            user_token=None) -> Tuple[RolePlay, MemoryWithRetrievalKnowledge]:
        unique_id = builder_id + '_' + session
        user_agent = self.user_bots[unique_id]
        if renew or user_agent is None:
            logger.info(f'init_user_chatbot_agent: {builder_id} {session}')
            user_agent = init_user_chatbot_agent(
                builder_id, session, use_tool_api=True, user_token=user_token)
            self.user_bots[unique_id] = user_agent
        return user_agent

    def clear_user_bot(self, builder_id, session):
        unique_id = builder_id + '_' + session
        user_agent = self.user_bots[unique_id]
        if user_agent is not None:
            self.user_bots.delete_key(unique_id)
        shutil.rmtree(
            get_user_preview_history_dir(builder_id, session),
            ignore_errors=True)
