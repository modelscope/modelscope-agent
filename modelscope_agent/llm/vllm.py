from typing import Dict, Iterator, List, Optional, Union

from modelscope_agent.utils.logger import agent_logger as logger
from modelscope_agent.utils.retry import retry

from .base import BaseChatModel, register_llm


@register_llm('vllm')
class VllmLLM(BaseChatModel):
    # from vllm import LLM, SamplingParams
    def __init__(self, model: str, model_server: str, llm, tokenizer,
                 sampling_params, **kwargs):
        super().__init__(model, model_server)
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "The package 'vllm' and 'transformers' are required for this module. Please install it using 'pip "
                "install vllm transformers>=4.33'.") from e
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        logger.info(
            f'chat stream vllm, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: True, args: {str(kwargs)}')
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        self.llm._validate_and_add_requests(
            inputs=inputs,
            params=self.sampling_params,
            lora_request=None,
        )
        total_toks = 0
        # TODO: support stop word
        while self.llm.llm_engine.has_unfinished_requests():
            step_outputs = self.llm.llm_engine.step()
            for output in step_outputs:
                for stp in output.outputs:
                    total_toks += len(stp.token_ids)
                    yield stp.text

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        logger.info(
            f'call vllm, model: {self.model}, messages: {str(messages)}, '
            f'stop: {str(stop)}, stream: False, args: {str(kwargs)}')
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        outputs = self.llm.generate(
            prompts=inputs, sampling_params=self.sampling_params)
        logger.info(f'call vllm success, output: {outputs[0].outputs[0].text}')
        # TODO: support stop word
        return outputs[0].outputs[0].text

    @retry(max_retries=3, delay_seconds=0.5)
    def chat(self,
             prompt: Optional[str] = None,
             messages: Optional[List[Dict]] = None,
             stop: Optional[List[str]] = None,
             stream: bool = False,
             **kwargs) -> Union[str, Iterator[str]]:
        if self.support_raw_prompt():
            return self.chat_with_raw_prompt(
                prompt=prompt, stream=stream, stop=stop, **kwargs)
        if not messages and prompt and isinstance(prompt, str):
            messages = [{'role': 'user', 'content': prompt}]
        return super().chat(
            messages=messages, stop=stop, stream=stream, **kwargs)
