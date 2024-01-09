class WebSearcher:
    timeout = 1000

    def __call__(self, **kwargs):
        raise NotImplementedError()
