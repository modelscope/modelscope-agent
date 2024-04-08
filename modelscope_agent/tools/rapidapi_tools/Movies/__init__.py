import sys

from ....utils import _LazyModule

_import_structure = {
    'movie_tv_music_search_and_download': [
        'SearchTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload',
        'GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload'
    ],
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
