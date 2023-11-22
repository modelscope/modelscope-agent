def qwen_chatml_prompt_preprocessor(messages):
    prompt = ''
    for message in messages:
        if message['role'] == 'assistant' and message['content'] == '':
            prompt += '<|im_start|>assistant\n'
        else:
            prompt = prompt + '<|im_start|>{role}\n{content}<|im_end|>\n'.format(
                role=message['role'],
                content=message['content'].lstrip('\n').rstrip())

    # in the case of the assistant message is not in the last one, such as function result
    if messages[-1]['role'] == 'assistant':
        last_assistant_message_list = messages[-1]['content'].split('\n')
        if last_assistant_message_list[-1] == '':
            last_assistant_message_list = last_assistant_message_list[:-1]
        if len(last_assistant_message_list) == 0:
            return prompt
        else:
            item_length = len('<|im_end|>\n')
            prompt = prompt[:-item_length]

    return prompt


def plate_preprocessor(messages):
    return qwen_chatml_prompt_preprocessor(messages)


def build_raw_prompt(model_name):
    if model_name.startswith('qwen'):
        return qwen_chatml_prompt_preprocessor
    else:
        return plate_preprocessor
