import os
import sys
from typing import Dict, Iterator, List, Optional

from .base import BaseChatModel, register_llm


@register_llm('modelscope')
class ModelScopeLLM(BaseChatModel):
    """
    This is language models from modelscope: Supports text input, text output
    """

    def __init__(self, model: str, model_server: str, **kwargs):
        super().__init__(model, model_server)
        try:
            import torch
        except ImportError:
            raise ImportError(
                'Please install torch first: `pip install torch` or refer https://pytorch.org/ '
            )

        from modelscope import (AutoModelForCausalLM, AutoTokenizer,
                                GenerationConfig, snapshot_download)

        # Download model based on model version
        self.model_version = kwargs.get('model_version', None)
        self.cache_dir = kwargs.get('cache_dir', None)
        if not os.path.exists(self.model):
            self.model_dir = snapshot_download(
                self.model, self.model_version, cache_dir=self.cache_dir)
        else:
            self.model_dir = self.model
        sys.path.append(self.model_dir)

        # Setup model config
        self.model_cls = kwargs.get('model_cls', AutoModelForCausalLM)
        self.tokenizer_cls = kwargs.get('tokenizer_cls', AutoTokenizer)
        self.use_raw_generation_config = kwargs.get(
            'use_raw_generation_config', False)

        self.device_map = kwargs.get('device_map', 'auto')
        self.generation_cfg = GenerationConfig(
            **kwargs.get('generate_cfg', {}))

        self.use_lora = kwargs.get('use_lora', False)
        self.lora_ckpt_dir = kwargs.get('lora_ckpt_dir',
                                        None) if self.use_lora else None
        self.custom_chat = kwargs.get('custom_chat', False)
        self.end_token = kwargs.get('end_token', '<|endofthink|>')
        self.include_end = kwargs.get('include_end', True)

        # load model
        self.model = self.model_cls.from_pretrained(
            self.model_dir,
            device_map=self.device_map,
            # device='cuda:0',
            torch_dtype=torch.float16,
            trust_remote_code=True)
        self.model = self.model.eval()

        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.model_dir, trust_remote_code=True)

        if self.use_lora:
            try:
                from swift import Swift
            except ImportError:
                raise ImportError(
                    'Please install swift first: `pip install ms-swift`')
            self.load_from_lora()

        if self.use_raw_generation_config:
            self.model.generation_config = GenerationConfig.from_pretrained(
                self.model_dir, trust_remote_code=True)

    def load_from_lora(self):
        from swift import Swift

        model = self.model.bfloat16()
        # transform to lora
        model = Swift.from_pretrained(model, self.lora_ckpt_dir)
        self.model = model

    def _chat_stream(self,
                     messages: List[Dict],
                     stop: Optional[List[str]] = None,
                     **kwargs) -> Iterator[str]:
        # agents = messages[0]['content']
        # todo: implement the streaming chat
        raise NotImplementedError

    def _chat_no_stream(self,
                        messages: List[Dict],
                        stop: Optional[List[str]] = None,
                        **kwargs) -> str:
        # The ModelScopeLLM only supports str inputs?
        prompt = messages[0]['content']

        if self.custom_chat and self.model.chat:
            response = self.model.chat(
                self.tokenizer, prompt, history=[], system='')[0]
        else:
            response = self._inference(prompt)

        end_idx = response.find(self.end_token)
        if end_idx != -1:
            end_idx += len(self.end_token) if self.include_end else 0
            response = response[:end_idx]

        return response

    def _inference(self, prompt: str) -> str:
        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        result = self.model.generate(
            input_ids=input_ids, generation_config=self.generation_cfg)

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)

        return response


@register_llm('modelscope_chatglm')
class ModelScopeChatGLM(ModelScopeLLM):

    def _inference(self, prompt: str) -> str:
        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command('<|user|>'),
            self.tokenizer.get_command('<|observation|>')
        ]
        result = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_cfg,
            eos_token_id=eos_token_id)

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)
        # 遇到生成'<', '|', 'user', '|', '>'的case
        response = response.split('<|user|>')[0].split('<|observation|>')[0]

        return response
