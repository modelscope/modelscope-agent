import os

import requests
from modelscope_agent.tools.base import register_tool
from modelscope_agent.tools.rapidapi_tools.basetool_for_alpha_umi import \
    BaseTool_alpha_umi
from requests.exceptions import RequestException, Timeout

MAX_RETRY_TIMES = 3
WORK_DIR = os.getenv('CODE_INTERPRETER_WORK_DIR', '/tmp/ci_workspace')


@register_tool('search_torrents_for_movie_tv_music_search_and_download')
class search_torrents_for_movie_tv_music_search_and_download(
        BaseTool_alpha_umi):
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
        toolbench_rapidapi_key = 'NrZV7wugc4YCn9W83a5GmBhHIKk0OztQTLRElq6xvAUioJPFyM'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/search'
        querystring = {
            'keywords': params['keywords'],
            'quantity': params['quantity'],
        }
        if 'page' in params:
            querystring['page'] = params['page']

        headers = {
            'X-RapidAPI-Key': toolbench_rapidapi_key,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except Exception:
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
class get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download(
        BaseTool_alpha_umi):
    description = '"Monthly Top 100 Music Torrents"'
    name = 'get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        toolbench_rapidapi_key = 'NrZV7wugc4YCn9W83a5GmBhHIKk0OztQTLRElq6xvAUioJPFyM'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_music'
        querystring = {}

        headers = {
            'X-RapidAPI-Key': toolbench_rapidapi_key,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except Exception:
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
class get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download(
        BaseTool_alpha_umi):
    description = '"Monthly Top 100 Games Torrents"'
    name = 'get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        toolbench_rapidapi_key = 'NrZV7wugc4YCn9W83a5GmBhHIKk0OztQTLRElq6xvAUioJPFyM'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_games'
        querystring = {}

        headers = {
            'X-RapidAPI-Key': toolbench_rapidapi_key,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except Exception:
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
class get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download(
        BaseTool_alpha_umi):
    description = '"Monthly Top 100 TV shows Torrents"'
    name = 'get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        toolbench_rapidapi_key = 'NrZV7wugc4YCn9W83a5GmBhHIKk0OztQTLRElq6xvAUioJPFyM'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_tv_shows'
        querystring = {}

        headers = {
            'X-RapidAPI-Key': toolbench_rapidapi_key,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except Exception:
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
class get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download(
        BaseTool_alpha_umi):
    description = '"Monthly Top 100 Movies Torrents"'
    name = 'get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download'
    parameters: list = []

    def call(self, params: str, **kwargs) -> str:
        toolbench_rapidapi_key = 'NrZV7wugc4YCn9W83a5GmBhHIKk0OztQTLRElq6xvAUioJPFyM'
        url = 'https://movie-tv-music-search-and-download.p.rapidapi.com/monthly_top100_movies'
        querystring = {}

        headers = {
            'X-RapidAPI-Key': toolbench_rapidapi_key,
            'X-RapidAPI-Host':
            'movie-tv-music-search-and-download.p.rapidapi.com'
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            observation = response.json()
        except BaseException:
            observation = response.text
        return observation

    def _remote_parse_input(self, *args, **kwargs):
        print(
            'kwargs passed to get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download:',
            kwargs)
        return kwargs
