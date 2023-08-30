import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from modelscope import GenerationConfig, snapshot_download
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from .base import LLM


class LocalLLM(LLM):

    def __init__(self, name, cfg):
        self.name = name
        super().__init__(cfg)

        model_id = self.cfg.get('model_id', '')
        model_revision = self.cfg.get('model_revision', None)

        if not os.path.exists(model_id):
            model_dir = snapshot_download(model_id, model_revision)
        else:
            model_dir = model_id
        self.model_dir = model_dir
        sys.path.append(self.model_dir)

        self.model_cls = self.cfg.get('model_cls', AutoModelForCausalLM)
        self.tokenizer_cls = self.cfg.get('tokenizer_cls', AutoTokenizer)

        self.device_map = self.cfg.get('device_map', 'auto')
        self.generate_cfg = self.cfg.get('generate_cfg', {})

        self.use_lora = self.cfg.get('use_lora', False)
        self.lora_cfg = LoRAConfig(
            **self.cfg.get('lora_cfg', {})) if self.use_lora else None

        self.end_token = self.cfg.get('eod_token', '<|endofthink|>')

        self.setup()

    def setup(self):
        model_cls = self.model_cls
        tokenizer_cls = self.tokenizer_cls

        self.model = model_cls.from_pretrained(
            self.model_dir,
            device_map=self.device_map,
            # device='cuda:0',
            torch_dtype=torch.float16,
            trust_remote_code=True)
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.model_dir, trust_remote_code=True)
        self.model = self.model.eval()

        if self.use_lora:
            self.load_from_lora(self.lora_cfg)

        if self.cfg.get('use_raw_generation_config', False):
            self.model.generation_config = GenerationConfig.from_pretrained(
                self.model_dir, trust_remote_code=True)

    def generate(self, prompt):

        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        result = self.model.generate(input_ids, **self.generate_cfg)

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)

        end_idx = response.find(self.end_token)
        response = response[:end_idx
                            + len(self.end_token)] if end_idx > 0 else response

        return response

    def load_from_lora(self, lora_config: LoRAConfig):

        model = self.model.bfloat16()
        # transform to lora
        Swift.prepare_model(model, lora_config)

        self.model = model
