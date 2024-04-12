import sys

from modelscope_agent.utils import _LazyModule

_import_structure = {
    'demo': ['AliyunRenewInstanceTool'],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
