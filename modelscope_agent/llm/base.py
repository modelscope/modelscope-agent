from abc import abstractmethod
from typing import List

import json


class LLM:
    name = ''

    def __init__(self, cfg):
        self.cfg = cfg
        self.agent_type = None
        self.model = None
        self.model_id = self.model

    def set_agent_type(self, agent_type):
        self.agent_type = agent_type

    @abstractmethod
    def generate(self, prompt: str, functions: list = [], **kwargs) -> str:
        """each llm should implement this function to generate response

        Args:
            prompt (str): prompt
            functions (list): list of functions object including: name, description, parameters
        Returns:
            str: response
        """
        raise NotImplementedError

    @abstractmethod
    def stream_generate(self,
                        prompt: str,
                        functions: list = [],
                        **kwargs) -> str:
        """stream generate response, which yields a generator of response in each step

        Args:
            prompt (str): prompt
            functions (list): list of functions object including: name, description, parameters
        Yields:
            Iterator[str]: iterator of step response
        """
        raise NotImplementedError

    def tokenize(self, input_text: str) -> List[int]:
        """tokenize is used to calculate the length of the text to meet the model's input length requirements

        Args:
            input_text (str): input text
        Returns:
            list[int]: token_ids
        """
        raise NotImplementedError

    def detokenize(self, input_ids: List[int]) -> str:
        """detokenize

        Args:
            input_ids (list[int]): input token_ids
        Returns:
            str: text
        """
        raise NotImplementedError
