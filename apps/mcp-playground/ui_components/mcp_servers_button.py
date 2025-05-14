# flake8: noqa: F401
from typing import List

import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms
from config import max_mcp_server_count


def McpServersButton(data_source: List[dict]):
    state = gr.State({'data_source': data_source})
    with antd.Button(
            value=None, variant='text', color='primary') as mcp_servers_btn:
        with ms.Slot('icon'):
            antd.Icon('ToolOutlined')
    with antd.Modal(
            width=420,
            footer=False,
            centered=True,
            styles=dict(footer=dict(display='none'))) as mcp_servers_modal:
        with ms.Slot('title'):
            with antd.Flex(gap='small', align='center'):
                ms.Text('MCP Servers')
                mcp_servers_switch = antd.Switch(True)
                antd.Typography.Text(
                    f'最大 MCP Server 连接数：{max_mcp_server_count}',
                    type='secondary',
                    elem_style=dict(fontSize=12, fontWeight='normal'))
        with antd.List(
                data_source=data_source,
                pagination=dict(pageSize=10,
                                hideOnSinglePage=True)) as mcp_servers_list:
            with ms.Slot(
                    'renderItem',
                    params_mapping=
                    "(item) => ({ text: { value: item.name, disabled: item.disabled }, tag: { style: { display: item.internal ? undefined: 'none' } }, switch: { value: item.enabled, mcp: item.name, disabled: item.disabled }})"
            ):
                with antd.List.Item():
                    with antd.Flex(
                            justify='space-between',
                            elem_style=dict(width='100%')):
                        with antd.Flex(gap='small'):
                            antd.Typography.Text(as_item='text')
                            antd.Tag('官方示例', color='green', as_item='tag')
                        mcp_server_switch = antd.Switch(as_item='switch')

    def change_mcp_servers_switch(mcp_servers_switch_value, state_value):
        state_value['data_source'] = [{
            **item, 'disabled':
            not mcp_servers_switch_value
        } for item in state_value['data_source']]
        return gr.update(value=state_value)

    def change_mcp_server_switch(state_value, e: gr.EventData):
        mcp = e._data['component']['mcp']

        enabled = e._data['payload'][0]

        state_value['data_source'] = [{
            **item, 'enabled': enabled
        } if item['name'] == mcp else item
                                      for item in state_value['data_source']]
        return gr.update(value=state_value)

    def apply_state_change(state_value):
        has_tool_use = False
        disabled_tool_use = False
        enabled_server_count = 0
        for item in state_value['data_source']:
            if item.get('enabled'):
                if enabled_server_count >= max_mcp_server_count:
                    item['enabled'] = False
                else:
                    enabled_server_count += 1
                    if item.get('disabled'):
                        disabled_tool_use = True
                    else:
                        has_tool_use = True

        if not disabled_tool_use:
            for item in state_value['data_source']:
                if enabled_server_count >= max_mcp_server_count:
                    item['disabled'] = not item.get('enabled', False)
                else:
                    item['disabled'] = False

        return gr.update(
            data_source=state_value['data_source'],
            footer='没有可用的 MCP Server'
            if len(state_value['data_source']) == 0 else ''), gr.update(
                color='primary' if has_tool_use else 'default'), gr.update(
                    value=not disabled_tool_use), gr.update(value=state_value)

    mcp_servers_btn.click(
        fn=lambda: gr.update(open=True),
        outputs=[mcp_servers_modal],
        queue=False)
    mcp_servers_switch.change(
        fn=change_mcp_servers_switch,
        inputs=[mcp_servers_switch, state],
        outputs=[state])
    mcp_server_switch.change(
        fn=change_mcp_server_switch, inputs=[state], outputs=[state])
    state.change(
        fn=apply_state_change,
        inputs=[state],
        outputs=[mcp_servers_list, mcp_servers_btn, mcp_servers_switch, state],
        queue=False)
    mcp_servers_modal.cancel(
        fn=lambda: gr.update(open=False),
        outputs=[mcp_servers_modal],
        queue=False)
    return state
