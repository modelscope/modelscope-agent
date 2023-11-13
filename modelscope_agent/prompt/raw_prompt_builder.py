def qwen_chatml_prompt_preprocessor(messages, system_prompt):
    if messages[0]['role'] == 'user':
        messages[0]['content'] = system_prompt + messages[0]['content']

    prompt = ''
    for message in messages:
        if message['role'] == 'assistant' and message['content'] == '':
            prompt += '\n<|im_start|>assistant\n'
        else:
            prompt = prompt + '<|im_start|>{role}\n{content}<|im_end|>\n'.format(
                role=message['role'], content=message['content'])

    # in the case of the assistant message is not in the last one, such as function result
    if messages[-1]['role'] == 'function':
        prompt += '\n<|im_start|>assistant\n'
    return prompt


def plate_preprocessor(messages, system_prompt):
    return system_prompt + messages[-1]['content']


def build_raw_prompt(model_name):
    if model_name.startswith('qwen'):
        return qwen_chatml_prompt_preprocessor
    else:
        return plate_preprocessor
