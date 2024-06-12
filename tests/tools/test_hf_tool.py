import pytest
from modelscope_agent.tools.hf_tool import HFTool

# from transformers import load_tool


@pytest.mark.skip()
def test_is_hf_tool():
    # test this tool should only be initialized by transformers.tools.tool
    with pytest.raises(ValueError) as e:
        HFTool('mock_tool')
    exec_msg = e.value.args[0]
    assert (exec_msg == 'HFTool should be type of HF tool')


@pytest.mark.skip()
def test_run_langchin_tool():
    # test run hf tool
    tool = load_tool('translation')
    name = 'translator'
    description = 'This is a tool that translates text from a language to another. '
    parameters = [{
        'name': 'text',
        'type': 'string',
        'description': 'the text to translate',
        'required': True
    }, {
        'name': 'src_lang',
        'type': 'string',
        'description': 'the language of the text to translate',
        'required': True
    }, {
        'name': 'tgt_lang',
        'type': 'string',
        'description': 'the language for the desired ouput language',
        'required': True
    }]
    shell_tool = HFTool(
        tool, name=name, description=description, parameters=parameters)
    input = """{'text': 'Hello','src_lang':'English','tgt_lang':'French'}"""
    res = shell_tool.call(input)
    print(res)
    assert res == '"Je vous salue."'
