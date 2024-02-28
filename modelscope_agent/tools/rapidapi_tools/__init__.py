from .Finance.current_exchage import (exchange_for_current_exchange,
                                      listquotes_for_current_exchange)
from .Modelscope.text_ie_tool import TextInfoExtractTool_for_alpha_umi
from .Movies.movie_tv_music_search_and_download import (
    get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download,
    get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download,
    search_torrents_for_movie_tv_music_search_and_download)
from .Number.numbers import (get_data_fact_for_numbers,
                             get_math_fact_for_numbers,
                             get_year_fact_for_numbers)
from .Translate.google_translate import (detect_for_google_translate,
                                         languages_for_google_translate,
                                         translate_for_google_translate)
