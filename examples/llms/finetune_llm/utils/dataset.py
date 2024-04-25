import ast
import os
import re
from typing import List, Optional, Tuple

import json
import numpy as np
import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset, concatenate_datasets
from swift.utils import get_seed

from modelscope import MsDataset

IGNORE_INDEX = -100


def get_ms_tool_dataset(dataset_name_or_file) -> HfDataset:
    # ms_tool_dataset: for train
    # each data may contain multiple segments, they are organized as a list
    # and split by flag. 0 for user input/ tool execute result..., 1 for label

    if os.path.isfile(dataset_name_or_file):
        with open(dataset_name_or_file, 'r') as f:
            if dataset_name_or_file.endswith('.json'):
                origin_data = json.load(f)
            elif dataset_name_or_file.endswith('.jsonl'):
                origin_data = []
                for line in f:
                    origin_data.append(json.loads(line))
    else:
        origin_data = MsDataset.load('damo/MSAgent-Bench', split='train')

    all_inputs_str = []
    all_inputs_flag = []
    for d in origin_data:
        content = d['conversations']
        if isinstance(content, str):
            content = ast.literal_eval(content)

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
            origin_response_str = '\n\n<|assistant|>:' + content[
                2 * i + 2]['value'] + '\n\n</s>'

            idx1, idx2 = -1, 0
            iter1 = re.finditer(r'<\|startofexec\|>', origin_response_str)
            iter2 = re.finditer(r'<\|endofexec\|>', origin_response_str)

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

        if len(inputs_flag) == 1:
            continue
        all_inputs_str.append(inputs_str)
        all_inputs_flag.append(inputs_flag)

    dataset = HfDataset.from_dict({
        'inputs': all_inputs_str,
        'flags': all_inputs_flag
    })
    return dataset


def get_ms_tool_dataset_test(dataset_name_or_file):
    # ms_tool_dataset: for train
    # each data may contain multiple segments, they are organized as different samples
    all_inputs_str = []
    all_labels_str = []

    if os.path.isfile(dataset_name_or_file):
        with open(dataset_json_file, 'r') as f:
            if dataset_json_file.endswith('.json'):
                origin_data = json.load(f)
            elif dataset_json_file.endswith('.jsonl'):
                origin_data = []
                for line in f:
                    origin_data.append(json.loads(line))
    else:
        origin_data = MsDataset.load('damo/MSAgent-Bench', split='validation')

    for d in origin_data:
        content = d['conversations']
        if isinstance(content, str):
            content = ast.literal_eval(content)

        # ilegal data
        if len(content) == 0 or content[0]['from'] != 'system':
            continue

        system_str = '<|system|>:' + content[0]['value']

        input_str = system_str

        for i in range(len(content) // 2):
            if len(content[2 * i + 2]['value']) == 0:
                continue

            assert content[2 * i + 1]['from'] == 'user'
            assert content[2 * i + 2]['from'] == 'assistant'
            # user input
            input_str += ('\n\n<|user|>:' + content[2 * i + 1]['value'])

            # assistant response
            origin_response_str = '\n\n<|assistant|>:' + content[2 * i
                                                                 + 2]['value']

            idx2 = 0

            iter1 = re.finditer(r'<\|startofexec\|>', origin_response_str)
            iter2 = re.finditer(r'<\|endofexec\|>', origin_response_str)

            for i1, i2 in zip(iter1, iter2):
                idx1 = i1.start()

                # llm response
                llm_response = origin_response_str[idx2:idx1]
                all_inputs_str.append(input_str)
                all_labels_str.append(llm_response)

                input_str += llm_response

                idx2 = i2.end()

                # exec result
                exec_result = origin_response_str[idx1:idx2]
                input_str += exec_result

            # summarize
            if idx2 != len(origin_response_str):
                final_summarize = origin_response_str[idx2:]
                all_inputs_str.append(input_str)
                all_labels_str.append(final_summarize)

    dataset = HfDataset.from_dict({
        'inputs': all_inputs_str,
        'labels': all_labels_str
    })
    return dataset


def process_dataset(dataset: HfDataset, dataset_test_size: float,
                    dataset_sample: int,
                    dataset_seed: int) -> Tuple[HfDataset, HfDataset]:
    random_state = np.random.RandomState(dataset_seed)
    if dataset_sample >= 0:
        index = random_state.permutation(len(dataset))[:dataset_sample]
        dataset = dataset.select(index)
    if dataset_test_size == 1.0:
        return dataset, None
    dataset = dataset.train_test_split(
        dataset_test_size, seed=get_seed(random_state))
    return dataset['train'], dataset['test']


def tokenize_function(example, tokenizer, max_length: Optional[int] = None):

    input_str, input_flag = example['inputs'], example['flags']
    input_tokenized = tokenizer(
        input_str,
        # return_tensors='pt',
        add_special_tokens=True,
        return_attention_mask=False)['input_ids']

    # transform list to tensor and cat them in dim=0
    input_id = torch.cat([torch.tensor(inp) for inp in input_tokenized], dim=0)
    # if flag = 1, the token should be origin result, if 0, it should be ignored
    label = []
    for inp, flag in zip(input_tokenized, input_flag):
        inp = torch.tensor(inp)
        label.append(inp.clone() if flag == 1 else torch.ones_like(inp)
                     * IGNORE_INDEX)

    label = torch.cat(label, dim=0)

    if max_length is not None and input_id.shape[0] > max_length:

        input_id = input_id[-max_length:]
        label = label[-max_length:]

    if torch.max(label) == IGNORE_INDEX:
        print(input_str)
        print('Warning: no label in this data')

    return dict(input_ids=input_id, labels=label)
