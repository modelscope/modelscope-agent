import ast
import copy
import datetime
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import json
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torchmetrics import Accuracy, MeanMetric
from tqdm.contrib import tzip

# from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                        EpochBasedTrainer, TrainingArgs, read_config,
                        snapshot_download)
from modelscope.metainfo import Trainers
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.models import Model
from modelscope.models.nlp import ChatGLM2Tokenizer, ChatGLMTokenizer
from modelscope.msdatasets.dataset_cls.custom_datasets.torch_custom_dataset import \
    TorchCustomDataset
from modelscope.preprocessors import TextGenerationTransformersPreprocessor
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from modelscope.trainers import build_trainer
from modelscope.utils.registry import default_group

DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'
IGNORE_INDEX = -100
MAX_LENGTH = 2048


@dataclass(init=False)
class TextGenerationArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
        })

    lr_scheduler: str = field(
        default=None,
        metadata={
            'help': 'The lr scheduler type',
            'cfg_node': 'train.lr_scheduler.type'
        })

    world_size: int = field(
        default=None,
        metadata={
            'help': 'The parallel world size',
            'cfg_node': 'megatron.world_size'
        })

    tensor_model_parallel_size: int = field(
        default=None,
        metadata={
            'help': 'The tensor model parallel size',
            'cfg_node': 'megatron.tensor_model_parallel_size'
        })

    use_megatron: bool = field(
        default=None, metadata={
            'help': 'Whether to use MegatronHook',
        })

    bf16: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use bf16',
            'cfg_node': 'train.bf16'
        })

    deepspeed: str = field(
        default=None,
        metadata={
            'help': 'The location of DeepSpeed json config file.',
        })

    T_max: int = field(
        default=None,
        metadata={
            'help': 'The T_max for CosineAnnealingLR',
            'cfg_node': 'train.lr_scheduler.T_max'
        })

    use_lora: int = field(
        default=0,
        metadata={'help': 'Whether to use lora to train the model.'},
    )

    enable_gradient_checkpoint: int = field(
        default=1,
        metadata={
            'help': 'Whether to user gradient checkpoint to save memory'
        })

    lora_rank: int = field(
        default=8,
        metadata={'help': 'The lora rank'},
    )

    lora_alpha: int = field(
        default=32,
        metadata={'help': 'The lora alpha'},
    )

    lora_dropout: float = field(
        default=0.1,
        metadata={'help': 'The lora dropout'},
    )

    lora_replace_module: str = field(
        default='',
        metadata={'help': 'The lora replace module'},
    )

    device_map: str = field(
        default=None,
        metadata={
            'help': 'A map that specifies where each submodule should go.'
        })

    max_length: int = field(
        default=2048, metadata={'help': 'max length of sequence'})


config, args = TextGenerationArguments().parse_cli().to_config()
print(config, args)
MAX_LENGTH = args.max_length


class MyToolDataset(TorchCustomDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, origin_data, tokenizer):

        data_dict = preprocess(origin_data, tokenizer)

        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def preprocess(origin_data, tokenizer):
    all_inputs_str = []
    all_inputs_flag = []

    for d in origin_data:
        content = d['conversations']

        # ilegal data
        if len(content) == 0 or content[0]['from'] != 'system':
            continue

        system_str = '<|system|>:' + content[0]['value']

        inputs_str = [system_str]  # segment of conservations
        inputs_flag = [
            0
        ]  # a flag to indicate whether the segment is assistant response

        for i in range(len(content) // 2):
            if len(content[2 * i + 2]['value']) == 0:
                continue

            assert content[2 * i + 1]['from'] == 'user'
            assert content[2 * i + 2]['from'] == 'assistant'
            # user input
            inputs_str.append('\n\n<|user|>:' + content[2 * i + 1]['value'])
            inputs_flag.append(0)

            # assistant response
            origin_response_str = '\n\n<|assistant|>:' + content[2 * i
                                                                 + 2]['value']

            idx2 = 0

            iter1 = re.finditer('<\|startofexec\|>', origin_response_str)
            iter2 = re.finditer('<\|endofexec\|>', origin_response_str)

            for i1, i2 in zip(iter1, iter2):
                idx1 = i1.start()

                # llm response
                inputs_str.append(origin_response_str[idx2:idx1])
                inputs_flag.append(1)

                idx2 = i2.end()

                # exec result
                inputs_str.append(origin_response_str[idx1:idx2])
                inputs_flag.append(0)

            if idx2 != len(origin_response_str):
                inputs_str.append(origin_response_str[idx2:])
                inputs_flag.append(1)

        all_inputs_str.append(inputs_str)
        all_inputs_flag.append(inputs_flag)

    # tokenize
    input_ids = []
    labels = []
    for input_str, input_flag in tzip(all_inputs_str, all_inputs_flag):
        input_tokenized = tokenizer(
            input_str,
            # return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=False)['input_ids']

        # transform list to tensor and cat them in dim=0
        input_id = torch.cat([torch.tensor(inp) for inp in input_tokenized],
                             dim=0)
        # input_id = torch.cat(input_tokenized, dim=0)
        if input_id.shape[0] > MAX_LENGTH:
            continue
        # if flag = 1, the token should be origin result, if 0, it should be ignored
        label = []
        for inp, flag in zip(input_tokenized, input_flag):
            inp = torch.tensor(inp)
            label.append(inp.clone() if flag == 1 else torch.ones_like(inp)
                         * IGNORE_INDEX)

        label = torch.cat(label, dim=0)

        input_ids.append(input_id)
        labels.append(label)

    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: AutoTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).to(
                torch.int64),
        )


if __name__ == '__main__':

    origin_data = []
    with open(args.dataset_json_file, 'r') as f:
        # for line in f.readlines():
        #     origin_data.append(json.loads(line))
        origin_data = json.load(f)
    # random split train and eval dataset
    perm = np.random.permutation(len(origin_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_origin_data = [origin_data[i] for i in train_indices]
    eval_origin_data = [origin_data[i] for i in eval_indices]

    revision = None
    if args.model == 'baichuan-inc/baichuan-7B':
        model_cls = AutoModelForCausalLM
        tokenizer_cls = AutoTokenizer
    elif args.model == 'ZhipuAI/chatglm2-6b':
        model_cls = Model
        tokenizer_cls = ChatGLM2Tokenizer
        revision = 'v1.0.7'
    else:
        model_cls = AutoModelForCausalLM
        tokenizer_cls = AutoTokenizer

    model_dir = snapshot_download(args.model)
    sys.path.append(model_dir)

    model = model_cls.from_pretrained(
        model_dir, trust_remote_code=True, revision=revision)
    if args.enable_gradient_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg_file = os.path.join(model_dir, 'configuration.json')

    tokenizer = tokenizer_cls.from_pretrained(
        model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = MyToolDataset(train_origin_data, tokenizer)
    eval_dataset = MyToolDataset(eval_origin_data, tokenizer)

    data_collator_fn = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    if args.use_lora != 0:
        lora_config = LoRAConfig(
            replace_modules=[args.lora_replace_module],
            rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout)
        model = model.bfloat16()
        Swift.prepare_model(model, lora_config)

    def cfg_modify_fn(cfg):
        if args.use_model_config:
            cfg.merge_from_dict(config)
        else:
            cfg = config

        # cfg.train.lr_scheduler = {
        #     'type': 'CosineAnnealingLR',
        #     'T_max': 5000,
        #     'options': {
        #         'by_epoch': False,
        #         "warmup": {
        #             'type': 'LinearWarmup',
        #             'warmup_ratio': 0.1,
        #             "warmup_iters": 200
        #         }
        #     }
        # }
        cfg.train.optimizer = {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'options': {
                'cumulative_iters': 8,
                'grad_clip': {
                    'norm_type': 2,
                    'max_norm': 2.0
                },
                "warmup": {
                    'warmup_ratio': 0.1,
                }
            }
        }
        if 'hooks' not in cfg.train:
            cfg.train['hooks'] = []
        if args.use_megatron:
            cfg.train.hooks.append({'type': 'MegatronHook'})
        if args.deepspeed:
            cfg.train.hooks.append({
                'type': 'DeepspeedHook',
                'config': args.deepspeed,
                'save_zero_checkpoint': True,
                'with_mpu': False,
            })

        # cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # #
        # cfg.train.work_dir = os.path.join(cfg.train.work_dir, cur_time)
        return cfg

    kwargs = dict(
        model=model,
        cfg_file=cfg_file,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        seed=args.seed,
        cfg_modify_fn=cfg_modify_fn,
        # No placement for model, leave the model to `device_map`
        data_collator=data_collator_fn,
        max_epochs=1,
    )

    trainer: EpochBasedTrainer = build_trainer(
        name=args.trainer, default_args=kwargs)
    trainer.train()
