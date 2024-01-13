import pytest
from langchain_community.tools import ShellTool
from modelscope_agent.tools.langchain_proxy_tool import LangchainTool


def test_is_langchain_tool():
    # test this tool should only be initialized by langchain.tools.tool
    with pytest.raises(ValueError) as e:
        LangchainTool('mock_tool')
    exec_msg = e.value.args[0]
    assert (exec_msg == 'langchain_tool should be type of langchain tool')


def test_run_langchin_tool():
    # test run langchain tool
    shell_tool = LangchainTool(ShellTool())
    input = """{'commands': ["echo 'Hello World!'"]}"""
    res = shell_tool.call(input)
    print(res)
    assert res == 'Hello World!\n'
