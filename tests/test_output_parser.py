import pytest
from modelscope_agent.output_parser import MsOutputParser


def test_ms_output_parser():
    # test `MSOutputParser` parse_response
    output_parser = MsOutputParser()

    # not call tool
    response_no_call_tool = 'normal llm response'
    assert output_parser.parse_response(response_no_call_tool) == (None, None)

    # call tool
    response_call_tool = "<|startofthink|>{\"api_name\": \"some_tool\",\
    \"parameters\": {\"para1\": \"name1\"}}<|endofthink|>"

    assert output_parser.parse_response(response_call_tool) == ('some_tool', {
        'para1':
        'name1'
    })

    # wrong json format
    response_call_tool = "<|startofthink|>{'api_name': 'some_tool', 'parameters': {'para1': 'name1'}}<|endofthink|>"
    with pytest.raises(ValueError) as e:
        output_parser.parse_response(response_call_tool)
    error_msg = e.value.args[0]
    assert error_msg == 'Wrong response format for output parser'
