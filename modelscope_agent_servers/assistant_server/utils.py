from typing import List

import json
from modelscope_agent_servers.assistant_server.models import (
    ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice, ChatMessage, DeltaMessage)


def parse_tool_result(llm_result: str):
    """
    Args:
        llm_result: the result from the model

    Returns: dict

    """
    try:
        import re
        import json
        result = re.search(r'Action: (.+)\nAction Input: (.+)', llm_result)
        action = result.group(1)
        action_input = json.loads(result.group(2))
        return action, action_input
    except Exception:
        return None, None


def parse_messages(messages: List[ChatMessage]):
    """
    Args:
        messages: the list of chat messages

    Returns: Tuple[List[str], str]

    """
    history = []

    for message in messages[:-1]:
        history.append(message.dict())

    image_url = None
    content = messages[-1].content
    if isinstance(content, str):
        query = content
    else:
        query = content[0]['text']
        image_url = [con['image_url'] for con in content[1:]]
    return query, history, image_url


def stream_choice_wrapper(response, model, request_id, llm):
    for chunk in response:
        choices = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role='assistant', content=chunk),
        )
        chunk = ChatCompletionResponse(
            id=request_id,
            object='chat.completion.chunk',
            choices=[choices],
            model=model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f'data: {data}\n\n'

    choices = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason='stop')
    chunk = ChatCompletionResponse(
        id=request_id,
        object='chat.completion.chunk',
        choices=[choices],
        model=model,
        usage=llm.get_usage())
    data = chunk.model_dump_json(exclude_unset=True)
    yield f'data: {data}\n\n'
    yield 'data: [DONE]\n\n'


def choice_wrapper(response: str,
                   tool_name: str = None,
                   tool_inputs: dict = None):
    """
    output should be in the format of openai choices
    "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": null,
            "tool_calls": [
              {
                "id": "call_abc123",
                "type": "function",
                "function": {
                  "name": "get_current_weather",
                  "arguments": "{\n\"location\": \"Boston, MA\"\n}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ],

    Args:
        response: the chatresponse object

    Returns: dict

    """
    # TODO: only support one tool call for now
    choice = {
        'index': 0,
        'message': {
            'role': 'assistant',
            'content': response,
        }
    }
    if tool_name is not None:
        choice['message']['tool_calls'] = [{
            'type': 'function',
            'function': {
                'name': tool_name,
                'arguments': json.dumps(tool_inputs, ensure_ascii=False)
            }
        }]
        choice['finish_reason'] = 'tool_calls'
    else:
        choice['finish_reason'] = 'stop'
    choice = ChatCompletionResponseChoice(**choice)
    return [choice]
