import argparse
import os
import re
import sys

import json
from rouge import Rouge
from tqdm import tqdm

from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from modelscope.pipelines import pipeline
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from modelscope.utils.constant import Tasks

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, required=True, help='predict data file path')
# args = parser.parse_args()


def preprocess(origin_data):
    all_inputs_str = []
    all_labels_str = []

    for d in origin_data:
        content = d['conversations']

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

            iter1 = re.finditer('<\|startofexec\|>', origin_response_str)
            iter2 = re.finditer('<\|endofexec\|>', origin_response_str)

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

    return all_inputs_str, all_labels_str


def inference(data, model, tokenizer):

    pred = []
    device = model.device

    for d in tqdm(data):
        input_ids = tokenizer(d, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]
        result_id = model.generate(
            input_ids=input_ids, max_new_tokens=512, do_sample=True)
        result_id = result_id[0].tolist()[input_len:]
        pred.append(tokenizer.decode(result_id))

    return pred


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
    for (ref, pred) in zip(refs, preds):
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
                r_input = json.loads(r['parameters'])
                p_input = json.loads(p['parameters'])
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
            pred_seqs.append(pred)

    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=pred_seqs, refs=ref_seqs, avg=True)
    rougel = rouge_score["rouge-l"]["f"]

    return {
        'action_em': sum(action_em) / len(action_em),
        'input_em': sum(input_em) / len(input_em),
        'rouge': rougel
    }


if __name__ == '__main__':

    test_data = []
    data_path = 'demo/tool_agent_finetune/train_v1.3_plugins_sample.json'
    with open(data_path, 'r') as f:
        for line in f.readlines():
            test_data.append(json.loads(line))

    all_inputs, all_labels = preprocess(test_data)

    model_id = 'baichuan-inc/baichuan-7B'
    model_dir = snapshot_download(model_id)
    sys.path.append(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, device_map='sequential')
    model = model.bfloat16()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True)

    ckpt_path = 'modelweight/baichuan-7b/output_best/pytorch_model.bin'
    lora_replace_module = ['W_pack']

    lora_rank = 8
    lora_alpha = 32
    lora_dropout = 0
    lora_config = LoRAConfig(
        replace_modules=lora_replace_module,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        pretrained_weights=ckpt_path)
    Swift.prepare_model(model, lora_config)

    all_preds = inference(all_inputs, model, tokenizer)
    res = evaluate(all_labels, all_preds)
    print(res)
