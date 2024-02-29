def test_rapidapi_current_exchange():
    from modelscope_agent.tools.rapidapi_tools.Finance.current_exchage import \
    (listquotes_for_current_exchange,
     exchange_for_current_exchange)
    kwargs = """{}"""
    tool = listquotes_for_current_exchange()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'from':'USD', 'to':'CNY', 'q':1}"""
    tool = exchange_for_current_exchange()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)


def test_rapidapi_numbers():
    from modelscope_agent.tools.rapidapi_tools.Number.numbers import \
    (get_data_fact_for_numbers,
     get_math_fact_for_numbers,
     get_year_fact_for_numbers)
    kwargs = """{'month':'7', 'day':'1'}"""
    tool = get_data_fact_for_numbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'number':'1497'}"""
    tool = get_math_fact_for_numbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'year':'1947'}"""
    tool = get_year_fact_for_numbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)


def test_rapidapi_movie_tv_music_search_and_download():
    from modelscope_agent.tools.rapidapi_tools.Movies.movie_tv_music_search_and_download import \
    (search_torrents_for_movie_tv_music_search_and_download,
     get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download,
     get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download,
     get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download,
     get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download)
    kwargs = """{'keywords':'sports', 'quantity':'10'}"""
    tool = search_torrents_for_movie_tv_music_search_and_download()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = get_monthly_top_100_games_torrents_for_movie_tv_music_search_and_download()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = get_monthly_top_100_movies_torrents_torrents_for_movie_tv_music_search_and_download()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = get_monthly_top_100_music_torrents_for_movie_tv_music_search_and_download()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = get_monthly_top_100_tv_shows_torrents_for_movie_tv_music_search_and_download()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)

