import os

import requests
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.rapidapi_tools.basetool_for_alpha_umi import \
    BasetoolAlphaUmi
from requests.exceptions import JSONDecodeError, RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')
RAPID_API_TOKEN = os.getenv('RAPID_API_TOKEN', None)


@register_tool('search_torrents_for_movie_tv_music_search_and_download')
class SearchTorrentsForMovieTvMusicSearchAndDownload(BasetoolAlphaUmi):
    description = 'Get downloadable  torrent link by movie name.'
    name = 'search_torrents_for_movie_tv_music_search_and_download'
    parameters: list = [{
        'name': 'keywords',
        'description': '',
        'required': True,
        'type': 'string'
    }, {
        'name': 'quantity',
        'description': 'MAX:40',
        'required': True,
        'type': 'string'
    }, {
        'name': 'page',
        'description': '',
        'required': False,
        'type': 'int'
    }]

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/search'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to search_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs


@register_tool(
    'get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download'
)
class GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload(
        BasetoolAlphaUmi):
    description = '"Monthly Top 100 Music Torrents"'
    name = 'get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_music'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs


@register_tool(
    'get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download'
)
class GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload(
        BasetoolAlphaUmi):
    description = '"Monthly Top 100 Games Torrents"'
    name = 'get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_games'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]
        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs


@register_tool(
    'get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download'
)
class GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload(
        BasetoolAlphaUmi):
    description = '"Monthly Top 100 TV shows Torrents"'
    name = 'get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_tv_shows'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs


@register_tool(
    'get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download'
)
class GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload(
        BasetoolAlphaUmi):
    description = '"Monthly Top 100 Movies Torrents"'
    name = 'get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_movies'
        querystring = {}
        for p in self.parameters:
            if p['name'] in params.keys():
                querystring[p['name']] = params[p['name']]

        headers = {
            'X-RapidAPI-Key': RAPID_API_TOKEN,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except JSONDecodeError:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs
