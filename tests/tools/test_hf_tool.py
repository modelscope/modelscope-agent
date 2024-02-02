import pytest
from modelscope_agent.tools.hf_tool import HFTool
from transformers import load_tool


def test_is_hf_tool():
    # test this tool should only be initialized by transformers.tools.tool
    with pytest.raises(ValueError) as e:
        HFTool('mock_tool')
    exec_msg = e.value.args[0]
    assert (exec_msg == 'HFTool should be type of HF tool')


def test_run_langchin_tool():
    # test run hf tool
    tool = load_tool('translation')
    shell_tool = HFTool(tool)
    input = """{'text': 'Hello','src_lang':'English','tgt_lang':'French'}"""
    res = shell_tool.call(input)
    print(res)
    assert res == 'Je vous salue.'
