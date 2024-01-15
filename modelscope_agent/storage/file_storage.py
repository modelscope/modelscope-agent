import hashlib
import os

from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.utils import (print_traceback, read_text_from_file,
                                          save_text_to_file)

from .base import BaseStorage

DEFAULT_DOCUMENT_STORAGE = ''


def hash_sha256(key):
    hash_object = hashlib.sha256(key.encode())
    key = hash_object.hexdigest()
    return key


class DocumentStorage(BaseStorage):

    def __init__(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.root = path
        # load all keys
        self.data = {}
        for file in os.listdir(path):
            self.data[file] = None

    def add(self, key: str, value: str):
        """
        add one key to db
        :param key: str
        :param value: str

        """
        # one file for one key value pair
        key = hash_sha256(key)

        msg = save_text_to_file(os.path.join(self.root, key), value)
        if msg == 'SUCCESS':
            self.data[key] = value
            return msg
        else:
            print_traceback()

    def search(self, key: str, re_load: bool = True):
        """
        search one value by key
        :param key: str
        :return: value: str
        """
        key = hash_sha256(key)
        if key in self.data and self.data[key] and (not re_load):
            return self.data[key]
        try:
            # lazy reading
            content = read_text_from_file(os.path.join(self.root, key))
            self.data[key] = content
            return content
        except Exception:
            return 'Not Exist'

    def delete(self, key):
        """
        delete one key value pair
        :param key: str

        """
        key = hash_sha256(key)
        try:
            if key in self.data:
                os.remove(os.path.join(self.root, key))
                self.data.pop(key)

            logger.info(f"Remove '{key}'")
        except OSError as ex:
            logger.error(f'Failed to remove: {ex}')

    def scan(self):
        for key in self.data.keys():
            yield [key, self.get(key)]
