import sys

from ....utils import _LazyModule

_import_structure = {
    'google_translate': [
        'DetectForGoogleTranslate', 'LanguagesForGoogleTranslate',
        'TranslateForGoogleTranslate'
    ],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
