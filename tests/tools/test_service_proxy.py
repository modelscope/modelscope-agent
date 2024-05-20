import os

import pytest
from modelscope_agent.tools.base import ToolServiceProxy

IN_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason='no need to run this test on ci')
def test_tool_service():
    try:
        tool_service = ToolServiceProxy('RenewInstance', {'test': 'xxx'})

        result = tool_service.call(
            "{\"instance_id\": 123, \"period\": \"mon\"}")
    except Exception as e:
        assert False, f'Failed to initialize tool service with error: {e}'

    assert result == "{'result': '已完成ECS实例ID为123的续费，续费时长mon月'}"
