import sys

from ....utils import _LazyModule

_import_structure = {
    'text_ie_tool':
    ['TextinfoextracttoolForAlphaUmi', 'exchange_for_current_exchange'],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
