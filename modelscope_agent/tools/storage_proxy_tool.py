import hashlib
from typing import Dict, Optional

from modelscope_agent.storage import DocumentStorage
from modelscope_agent.tools.base import BaseTool, register_tool


def hash_sha256(key):
    hash_object = hashlib.sha256(key.encode())
    key = hash_object.hexdigest()
    return key


@register_tool('storage')
class Storage(BaseTool):
    """
    This is a special tool for data storage
    """
    name = 'storage'
    description = '数据在文件系统中存储和读取'
    parameters = [{
        'name': 'path',
        'type': 'string',
        'description': '数据存储的目录',
        'required': True
    }, {
        'name': 'operate',
        'type': 'string',
        'description':
        '数据操作类型，可选项为["add", "search", "delete", "scan"]之一，分别为存数据、取数据、删除数据、遍历数据',
        'required': True
    }, {
        'name': 'key',
        'type': 'string',
        'description': '数据的名称，是一份数据的唯一标识'
    }, {
        'name': 'value',
        'type': 'string',
        'description': '数据的内容，仅存数据时需要'
    }]

    def __init__(self, cfg: Optional[Dict] = None):
        """

        :param schema: Format of tools, default to oai format, in case there is a need for other formats
        """
        super().__init__(cfg)
        self.root = None
        self.data = {}

    def call(self, params: str, **kwargs):
        """
        init one database: one folder
        Args:
            params:
            **kwargs:

        Returns:

        """
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        path = params['path']

        # get data store class
        datastore_cls = kwargs.get('db', DocumentStorage)
        datastore = datastore_cls(path)

        operate = params['operate']
        if operate == 'add':
            return datastore.add(params['key'], params['value'])
        elif operate == 'search':
            return datastore.search(params['key'])
        elif operate == 'delete':
            return datastore.delete(params['key'])
        else:
            return datastore.scan()
