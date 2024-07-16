import re
from typing import List

import json
from modelscope_agent_servers.assistant_server.models import (
    ChatCompletionResponse, ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice, ChatMessage, DeltaMessage)


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


def choice_wrapper(response: str, tool_list: list = []):
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
        tool_list:  the tool list from the output of llm
        response: the chat response object

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
    if len(tool_list) > 0:
        tool_calls = []
        for item in tool_list:
            tool_dict = {'type': 'function', 'function': item}
            tool_calls.append(tool_dict)
        choice['message']['tool_calls'] = tool_calls
        choice['finish_reason'] = 'tool_calls'
    else:
        choice['finish_reason'] = 'stop'
    choice = ChatCompletionResponseChoice(**choice)
    return [choice]
