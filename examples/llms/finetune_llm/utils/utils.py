import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import json
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from rouge import Rouge
from swift import get_logger
from swift.utils.tb_utils import (TB_COLOR, TB_COLOR_SMOOTH,
                                  read_tensorboard_file, tensorboard_smoothing)
from torch import dtype as Dtype
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from transformers import (GenerationConfig, PreTrainedTokenizerBase,
                          TextStreamer)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
logger = get_logger()

DEFAULT_PROMPT = """Here's a conversation between a human and an AI assistant. \
The AI assistant provides detailed, friendly answers for the human.

### Human:
{instruction}

### AI:
"""

DTYPE_MAPPING = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp32': torch.float32
}


def get_dist_setting() -> Tuple[int, int, int]:
    """return rank, local_rank, world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    return rank, local_rank, world_size


def is_dist():
    rank = int(os.getenv('RANK', -1))
    return rank >= 0


def show_layers(model: Module, max_lines: Optional[int] = 20) -> None:
    named_p = list(model.named_parameters())
    for i, (n, p) in enumerate(named_p):
        if max_lines is not None and i >= max_lines:
            logger.info('...')
            break
        logger.info(
            f'[{n}]: requires_grad={p.requires_grad}, dtype={p.dtype}, device={p.device}'
        )


def plot_images(images_dir: str,
                tb_dir: str,
                smooth_key: List[str],
                smooth_val: float = 0.9,
                figsize: Tuple[int, int] = (8, 5),
                dpi: int = 100) -> None:
    os.makedirs(images_dir, exist_ok=True)
    fname = [
        fname for fname in os.listdir(tb_dir)
        if os.path.isfile(os.path.join(tb_dir, fname))
    ][0]
    tb_path = os.path.join(tb_dir, fname)
    data = read_tensorboard_file(tb_path)

    for k in data.keys():
        _data = data[k]
        steps = [d['step'] for d in _data]
        values = [d['value'] for d in _data]
        if len(values) == 0:
            continue
        _, ax = plt.subplots(1, 1, squeeze=True, figsize=figsize, dpi=dpi)
        ax.set_title(k)
        if len(values) == 1:
            ax.scatter(steps, values, color=TB_COLOR_SMOOTH)
        elif k in smooth_key:
            ax.plot(steps, values, color=TB_COLOR)
            values_s = tensorboard_smoothing(values, smooth_val)
            ax.plot(steps, values_s, color=TB_COLOR_SMOOTH)
        else:
            ax.plot(steps, values, color=TB_COLOR_SMOOTH)
        fpath = os.path.join(images_dir, k.replace('/', '_'))
        plt.savefig(fpath, dpi=dpi, bbox_inches='tight')


def inference(
    data,
    model,
    tokenizer,
    generation_config: Optional[GenerationConfig] = None,
) -> str:

    input_ids = tokenizer(data, return_tensors='pt').input_ids.cuda()
    input_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    result_id = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config)
    result_id[result_id >= tokenizer.vocab_size] = tokenizer.pad_token_id
    result_id = result_id[0].tolist()[input_len:]
    result = tokenizer.decode(result_id, skip_special_tokens=True)

    return result


def select_dtype(dtype: str) -> Tuple[Dtype, bool, bool]:
    """
    dtype: Literal['fp16', 'bf16', 'fp32']
    """
    torch_dtype = DTYPE_MAPPING[dtype]

    assert torch_dtype in {torch.float16, torch.bfloat16, torch.float32}
    if torch_dtype == torch.float16:
        fp16, bf16 = True, False
    elif torch_dtype == torch.bfloat16:
        support_bf16 = torch.cuda.is_bf16_supported()
        if not support_bf16:
            logger.warning(f'support_bf16: {support_bf16}')
        fp16, bf16 = False, True
    else:
        fp16, bf16 = False, False
    return torch_dtype, fp16, bf16


def find_all_linear_for_lora(model: Module,
                             quantization_bit: Optional[int],
                             model_type: Optional[str] = None) -> List[str]:
    """ref: https://github.com/artidoro/qlora"""
    head_module_name = 'lm_head'
    if model_type.startswith('chatglm2-6b'):
        head_module_name = 'output_layer'
    if model_type.startswith('qwen-vl'):
        return ['c_attn', 'attn.c_proj', 'w1', 'w2']
    if quantization_bit == 4:
        from bitsandbytes.nn import Linear4bit
        linear_cls = Linear4bit
    elif quantization_bit == 8:
        from bitsandbytes.nn import Linear8bitLt
        linear_cls = Linear8bitLt
    else:
        linear_cls = Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            module_name = name.split('.')[-1]
            if head_module_name not in module_name:
                lora_module_names.add(module_name)
    return list(lora_module_names)


def select_bnb(quantization_bit: Optional[int],
               bnb_4bit_compute_dtype: str) -> Tuple[Dtype, bool, bool]:
    bnb_4bit_compute_dtype = DTYPE_MAPPING[bnb_4bit_compute_dtype]
    assert bnb_4bit_compute_dtype in {
        torch.float16, torch.bfloat16, torch.float32
    }
    if quantization_bit == 4:
        load_in_4bit, load_in_8bit = True, False
    elif quantization_bit == 8:
        load_in_4bit, load_in_8bit = False, True
    else:
        load_in_4bit, load_in_8bit = False, False

    return bnb_4bit_compute_dtype, load_in_4bit, load_in_8bit


def broadcast_string(string: Optional[str], buffer_size: int = 100) -> str:
    """
    string: main rank: str
        other rank: None
    return: all rank: str
    """
    assert dist.is_initialized()
    rank, local_rank, _ = get_dist_setting()
    assert rank >= 0
    if rank == 0:
        assert string is not None
        tensor = torch.tensor(
            [ord(c) for c in string] + [0] * (buffer_size - len(string)),
            dtype=torch.int64,
            device=local_rank)
    else:
        tensor = torch.zeros(buffer_size, dtype=torch.int64, device=local_rank)
    dist.broadcast(tensor, 0)
    first_zero = (tensor == 0).nonzero()[0].item()
    res = tensor.tolist()[:first_zero]
    return ''.join([chr(x) for x in res])


def evaluate(refs, preds):
    # action: em
    # action input: em
    # answer: rouge
    re_pattern1 = re.compile(
        pattern=r'<\|startofthink\|>([\s\S]+)<\|endofthink\|>')
    re_pattern2 = re.compile(r'{[\s\S]+}')
    action_em = []
    input_em = []
    ref_seqs = []
    pred_seqs = []
    for i, (ref, pred) in enumerate(zip(refs, preds)):
        try:
            think_content = re_pattern1.search(ref).group(1)
            think_content = re_pattern2.search(think_content).group()
            r = json.loads(think_content.replace('\n', ''))
            try:
                think_content = re_pattern1.search(pred).group(1)
                think_content = re_pattern2.search(think_content).group()
                p = json.loads(think_content.replace('\n', ''))
                if p['api_name'] == r['api_name']:
                    action_em.append(1)
                else:
                    action_em.append(0)
                r_input = r['parameters']
                p_input = p['parameters']
                match = True
                for k, v in r_input.items():
                    if k in p_input.keys() and p_input[k] == v:
                        continue
                    else:
                        match = False
                        break
                for k in p_input.keys():
                    if k not in r_input.keys():
                        match = False
                        break
                if match:
                    input_em.append(1)
                else:
                    input_em.append(0)
            except Exception:
                action_em.append(0)
                input_em.append(0)
        except Exception:
            ref_seqs.append(ref)
            pred_seqs.append(pred if pred else ' ')

    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=pred_seqs, refs=ref_seqs, avg=True)
    rougel = rouge_score['rouge-l']['f']

    return {
        'action_em': sum(action_em) / len(action_em),
        'input_em': sum(input_em) / len(input_em),
        'rouge': rougel
    }


def data_collate_fn(batch: List[Dict[str, Any]],
                    tokenizer: PreTrainedTokenizerBase,
                    padding_to: Optional[int] = None) -> Dict[str, Any]:
    """
    Args:
        batch(`List[Dict[str, Any]]`): The input data in batch
        padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
            will be padded to the `longest`
    """
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    if padding_to is not None:
        padding_len = padding_to - input_ids[0].shape[-1]
        if padding_len > 0:
            input_ids[0] = F.pad(input_ids[0], (0, padding_len), 'constant',
                                 tokenizer.pad_token_id)
            attention_mask[0] = F.pad(attention_mask[0], (0, padding_len),
                                      'constant', 0)
            labels[0] = F.pad(labels[0], (0, padding_len), 'constant', -100)

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    res = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return res
