import sys

from modelscope_agent.utils import _LazyModule

_import_structure = {
    'Finance':
    ['ListquotesForCurrentExchange', 'exchange_for_current_exchange'],
    'Modelscope': ['TextinfoextracttoolForAlphaUmi'],
    'Movies': [
        'GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload',
        'SearchTorrentsForMovieTvMusicSearchAndDownload'
    ],
    'Number': [
        'GetDataFactForNumbers', 'GetMathFactForNumbers',
        'GetYearFactForNumbers'
    ],
    'Translate': [
        'DetectForGoogleTranslate', 'LanguagesForGoogleTranslate',
        'TranslateForGoogleTranslate'
    ]
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
