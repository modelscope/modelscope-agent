# flake8: noqa: F401
import gradio as gr
import json
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms


def AddMcpServerButton():
    with antd.Button(
            '添加 MCP Server', type='primary',
            size='small') as add_mcp_server_btn:
        with ms.Slot('icon'):
            antd.Icon('PlusOutlined')
    with antd.Modal(
            title='添加 MCP Server',
            footer=False,
            styles=dict(footer=dict(display='none'))) as add_mcp_server_modal:
        with antd.Tabs():
            with antd.Tabs.Item(label='表单添加'):
                with antd.Form(layout='vertical') as add_mcp_server_form:
                    with antd.Form.Item(
                            form_name='name',
                            label='名称',
                            rules=[{
                                'required': True
                            }]):
                        antd.Input(placeholder='MCP Server 名称，如 fetch、time 等')
                    with antd.Form.Item(
                            form_name='url',
                            label='SSE 链接',
                            rules=[{
                                'required': True
                            }]):
                        antd.Input(placeholder='MCP Server SSE 链接')
                    with antd.Flex(gap='small', justify='end'):
                        add_mcp_server_modal_cancel_btn = antd.Button('取消')
                        antd.Button('确定', html_type='submit', type='primary')
            with antd.Tabs.Item(label='JSON 添加'):
                with antd.Form(layout='vertical') as add_mcp_server_json_form:
                    with antd.Form.Item(
                            form_name='json',
                            label='JSON',
                            rules=[{
                                'required':
                                True,
                                'validator':
                                """(_, value) => {
                                                if (!value) {
                                                    return Promise.reject('请输入 JSON 值');
                                                }
                                                try {
                                                    const parsedValue = JSON.parse(value);
                                                    if (!parsedValue.mcpServers || typeof parsedValue.mcpServers !== 'object') {
                                                        return Promise.reject('配置必须包含正确的 mcpServers 字段');
                                                    }
                                                    return Promise.resolve()
                                                } catch {
                                                    return Promise.reject('请输入有效的 JSON 值');
                                                }
                                            } """
                            }]):
                        antd.Input.Textarea(
                            auto_size=dict(minRows=4, maxRows=8),
                            placeholder=json.dumps(
                                {
                                    'mcpServers': {
                                        'fetch': {
                                            'type': 'sse',
                                            'url': 'mcp server sse url'
                                        }
                                    }
                                },
                                indent=4))
                    with antd.Flex(gap='small', justify='end'):
                        add_mcp_server_modal_json_cancel_btn = antd.Button(
                            '取消')
                        antd.Button('确定', html_type='submit', type='primary')
    add_mcp_server_btn.click(
        fn=lambda: gr.update(open=True),
        outputs=[add_mcp_server_modal],
        queue=False)
    gr.on(
        triggers=[
            add_mcp_server_modal_cancel_btn.click,
            add_mcp_server_modal_json_cancel_btn.click,
            add_mcp_server_modal.cancel
        ],
        queue=False,
        fn=lambda: gr.update(open=False),
        outputs=[add_mcp_server_modal])
    add_mcp_server_form.finish(
        lambda: (gr.update(value={
            'name': '',
            'url': '',
        }), gr.update(open=False)),
        outputs=[add_mcp_server_form, add_mcp_server_modal])
    add_mcp_server_json_form.finish(
        lambda: (gr.update(value={'json': ''}), gr.update(open=False)),
        outputs=[add_mcp_server_json_form, add_mcp_server_modal])
    return add_mcp_server_form, add_mcp_server_json_form
