import pytest
from tests.utils import MockTool


def test_tool_schema():
    # test the schema of user-defined tool
    wrong_parameters_schema = [{'name': 'x', 'type': 'str', 'required': True}]
    with pytest.raises(ValueError) as e:
        MockTool('mock_tool', lambda x: x, 'mock tool',
                 wrong_parameters_schema)
    exec_msg = e.value.args[0]
    print(e.value)
    assert exec_msg == 'Error when parsing parameters of mock_tool'


# TODO: other tools add if needed
