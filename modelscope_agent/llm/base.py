from abc import abstractmethod

import json


class LLM:
    name = ''

    def __init__(self, cfg):
        self.cfg = cfg.get(self.name, {})

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """each llm should implement this function to generate response

        Args:
            prompt (str): prompt
        Returns:
            str: response
        """
        raise NotImplementedError

    @abstractmethod
    def stream_generate(self, prompt: str) -> str:
        """stream generate response, which yields a generator of response in each step

        Args:
            prompt (str): prompt
        Yields:
            Iterator[str]: iterator of step response
        """
        raise NotImplementedError
