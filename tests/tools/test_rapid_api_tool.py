def test_rapidapi_current_exchange():
    from modelscope_agent.tools.rapidapi_tools.Finance.current_exchage import (
        ListquotesForCurrentExchange, exchange_for_current_exchange)
    kwargs = """{}"""
    tool = ListquotesForCurrentExchange()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'from':'USD', 'to':'CNY', 'q':1}"""
    tool = exchange_for_current_exchange()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)


def test_rapidapi_numbers():
    from modelscope_agent.tools.rapidapi_tools.Number.numbers import (
        GetDataFactForNumbers, GetMathFactForNumbers, GetYearFactForNumbers)
    kwargs = """{'month':'7', 'day':'1'}"""
    tool = GetDataFactForNumbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'number':'1497'}"""
    tool = GetMathFactForNumbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{'year':'1947'}"""
    tool = GetYearFactForNumbers()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)


def test_rapidapi_movie_tv_music_search_and_download():
    from modelscope_agent.tools.rapidapi_tools.Movies.movie_tv_music_search_and_download import (
        SearchTorrentsForMovieTvMusicSearchAndDownload,
        GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload,
        GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload,
        GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload,
        GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload)
    kwargs = """{'keywords':'sports', 'quantity':'10'}"""
    tool = SearchTorrentsForMovieTvMusicSearchAndDownload()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = GetMonthlyTop100GamesTorrentsForMovieTvMusicSearchAndDownload()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = GetMonthlyTop100MoviesTorrentsTorrentsForMovieTvMusicSearchAndDownload(
    )
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = GetMonthlyTop100MusicTorrentsForMovieTvMusicSearchAndDownload()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
    kwargs = """{}"""
    tool = GetMonthlyTop100TvShowsTorrentsForMovieTvMusicSearchAndDownload()
    res = tool.call(kwargs)
    assert isinstance(res, dict) or isinstance(res, str)
