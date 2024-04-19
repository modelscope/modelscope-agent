import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import json
import torch
from swift import Swift, get_logger
from swift.utils import parse_args, print_model_info, seed_everything
from tqdm import tqdm
from transformers import BitsAndBytesConfig, GenerationConfig, TextStreamer
from utils import (DEFAULT_PROMPT, MODEL_MAPPING, evaluate,
                   get_model_tokenizer, get_ms_tool_dataset_test, inference,
                   process_dataset, select_bnb, select_dtype, show_layers)

logger = get_logger()


@dataclass
class InferArguments:
    model_type: str = field(
        default='qwen-7b', metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    ckpt_dir: str = '/path/to/your/vx_xxx/checkpoint-xxx'
    eval_human: bool = False  # False: eval test_dataset

    seed: int = 42
    dtype: str = field(
        default='bf16', metadata={'choices': {'bf16', 'fp16', 'fp32'}})
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh', metadata={'help': 'dataset'})
    dataset_seed: int = 42
    dataset_sample: int = 20000  # -1: all dataset
    dataset_test_size: float = 0.01
    prompt: str = DEFAULT_PROMPT
    max_length: Optional[int] = 1024

    quantization_bit: Optional[int] = field(
        default=None, metadata={'choices': {4, 8}})
    bnb_4bit_comp_dtype: str = field(
        default='fp16', metadata={'choices': {'fp16', 'bf16', 'fp32'}})
    bnb_4bit_quant_type: str = field(
        default='nf4', metadata={'choices': {'fp4', 'nf4'}})
    bnb_4bit_use_double_quant: bool = True

    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 10
    top_p: float = 0.8

    def __post_init__(self):
        if not os.path.isdir(self.ckpt_dir):
            raise ValueError(f'Please enter a valid ckpt_dir: {self.ckpt_dir}')
        self.torch_dtype, _, _ = select_dtype(self.dtype)
        self.bnb_4bit_compute_dtype, self.load_in_4bit, self.load_in_8bit = select_bnb(
            self.quantization_bit, self.bnb_4bit_comp_dtype)


def llm_infer(args: InferArguments) -> None:
    logger.info(f'device_count: {torch.cuda.device_count()}')
    seed_everything(args.seed)

    # ### Loading Model and Tokenizer
    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            args.load_in_8bit,
            args.load_in_4bit,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant)
        logger.info(f'quantization_config: {quantization_config.__dict__}')
        kwargs['quantization_config'] = quantization_config

    if args.sft_type == 'full':
        kwargs['model_dir'] = args.ckpt_dir
    model, tokenizer = get_model_tokenizer(
        args.model_type, torch_dtype=args.torch_dtype, **kwargs)

    # ### Preparing lora
    if args.sft_type == 'lora':
        model = Swift.from_pretrained(model, args.ckpt_dir)

    show_layers(model)
    print_model_info(model)

    # ### Inference

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')

    dataset = get_ms_tool_dataset_test(args.dataset)
    test_dataset, _ = process_dataset(dataset, args.dataset_test_size,
                                      args.dataset_sample, args.dataset_seed)
    del dataset
    preds = []
    labels = []
    for data in tqdm(test_dataset):
        label = data['labels']
        inputs = data['inputs']
        preds.append(inference(inputs, model, tokenizer, generation_config))
        labels.append(label)
    return labels, preds


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    labels, preds = llm_infer(args)

    res_dir = os.path.join(
        args.ckpt_dir,
        f'tool_eval_result_{args.quantization_bit}_{args.sft_type}.json')
    label_dir = os.path.join(
        args.ckpt_dir,
        f'eval_label_{args.quantization_bit}_{args.sft_type}.json')
    pred_dir = os.path.join(
        args.ckpt_dir,
        f'eval_preds_{args.quantization_bit}_{args.sft_type}.json')

    with open(label_dir, 'w') as f:
        json.dump(labels, f, ensure_ascii=False)

    with open(pred_dir, 'w') as f:
        json.dump(preds, f, ensure_ascii=False)

    res = evaluate(labels, preds)

    print(res)

    # 打开文件并将数据写入 JSON 格式
    with open(res_dir, "w") as file:
        json.dump(res, file, ensure_ascii=False)
