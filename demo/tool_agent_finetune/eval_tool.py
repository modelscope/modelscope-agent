import argparse

import json
from rouge import Rouge

parser = argparse.ArgumentParser()
parser.add_argument(
    '--ref_path', type=str, required=True, help='reference file path')
parser.add_argument(
    '--pred_path', type=str, required=True, help='prediction file path')
args = parser.parse_args()


def evaluate(refs, preds):
    # action: em
    # action input: em
    # answer: rouge
    action_em = []
    input_em = []
    ref_seqs = []
    pred_seqs = []
    for (ref, pred) in zip(refs, preds):
        ref_ans = ref['answer']
        pred_ans = pred['answer']
        for i, r in enumerate(ref_ans):
            if i >= len(pred_ans):
                p = ''
            else:
                p = pred_ans[i]
            try:
                r = json.loads(r)
                try:
                    p = json.loads(p)
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
                ref_seqs.append(r)
                pred_seqs.append(p)

    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=pred_seqs, refs=ref_seqs, avg=True)
    rougel = rouge_score["rouge-l"]["f"]

    return {
        'action_em': sum(action_em) / len(action_em),
        'input_em': sum(input_em) / len(input_em),
        'rouge': rougel
    }


if __name__ == "__main__":
    refs = []
    with open(args.ref_path) as f:
        refs = json.load(f)
    preds = []
    with open(args.pred_path) as f:
        preds = json.load(f)

    metric = evaluate(refs, preds)
    print(metric)
