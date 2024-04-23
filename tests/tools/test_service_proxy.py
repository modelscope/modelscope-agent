from modelscope_agent.tools.base import ToolServiceProxy


def test_tool_service():
    try:
        tool_service = ToolServiceProxy('RenewInstance', {'test': 'xxx'})

        result = tool_service.call(
            "{\"instance_id\": 123, \"period\": \"mon\"}")
    except Exception as e:
        assert False, f'Failed to initialize tool service with error: {e}'

    assert result == "{'result': '已完成ECS实例ID为123的续费，续费时长mon月'}"
