import pytest
from agent_scope.action_parser import MsActionParser


def test_ms_action_parser():
    # test `MSActionParser` parse_response
    action_parser = MsActionParser()

    # not call tool
    response_no_call_tool = 'normal llm response'
    assert action_parser.parse_response(response_no_call_tool) == (None, None)

    # call tool
    response_call_tool = "<|startofthink|>{\"api_name\": \"some_tool\",\
    \"parameters\": {\"para1\": \"name1\"}}<|endofthink|>"

    assert action_parser.parse_response(response_call_tool) == ('some_tool', {
        'para1':
        'name1'
    })

    # wrong json format
    response_call_tool = "<|startofthink|>{'api_name': 'some_tool', 'parameters': {'para1': 'name1'}}<|endofthink|>"
    with pytest.raises(ValueError) as e:
        action_parser.parse_response(response_call_tool)
    error_msg = e.value.args[0]
    assert error_msg == 'Wrong response format for action parser'
