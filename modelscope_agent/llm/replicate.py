import os
from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.llm.base import BaseChatModel, register_llm
from modelscope_agent.utils.retry import retry

@register_llm('replicate')
class ReplicateModel(BaseChatModel):

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        api_key = kwargs.get('api_key',
                             os.getenv('REPLICATE_API_TOKEN',
                                       '')).strip()
        assert api_key, 'REPLICATE_API_TOKEN is required.'

    def _chat_stream(self,
                     messages: List[Dict],
                     top_p: int = 1,
                     max_new_tokens: int = 500,
                     **kwargs) -> Iterator[str]:
        # lazy import 
        import replicate

        logger.info(
            f'call replicate api. messages: {str(messages)}, '
            f'model: {self.model}'
        )

        prompt = messages[-1]['content']
        system_prompt = messages[0]['content']
        
        input = {
            "top_p": top_p,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens
        }
        for event in replicate.stream(
            self.model,
            input=input
        ):
            yield str(event)

    def _chat_no_stream(self,
                     messages: List[Dict],
                     top_p: int = 1,
                     max_new_tokens: int = 500,
                     **kwargs) -> str:
        pass