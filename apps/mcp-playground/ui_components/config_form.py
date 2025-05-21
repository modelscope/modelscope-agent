import json
import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.base as ms

from .add_mcp_server_button import AddMcpServerButton
from config import default_mcp_config, model_options, default_sys_prompt


def ConfigForm():
    with antd.Form(layout="vertical",
                   value={
                       "sys_prompt": default_sys_prompt,
                       "model": model_options[0]["value"]
                   }) as config_form:
        with antd.Form.Item(label_col=24):
            with ms.Slot("extra"):
                with antd.Flex(justify="end", elem_style=dict(marginTop=10)):
                    mcp_config_confirm_btn = antd.Button("保存配置",
                                                         type="primary")
            with antd.Flex(wrap=True,
                           justify="space-between",
                           gap="middle",
                           elem_style=dict(paddingBottom=8)):
                with antd.Flex(gap="middle", wrap=True):
                    with antd.Flex(gap="small", align="center"):
                        antd.Typography.Text("MCP Servers",
                                             elem_style=dict(fontSize=14))
                        antd.Typography.Text("编辑以下内容以修改运行中的 MCP Servers",
                                             elem_style=dict(fontSize=12),
                                             type="secondary")
                        with antd.Tooltip(title="目前只支持 SSE 类型的 MCP Server"):
                            with antd.Typography.Text(type="warning"):
                                antd.Icon("InfoCircleOutlined")
                    add_mcp_server_form, add_mcp_server_json_form = AddMcpServerButton(
                    )

                    with antd.Button("重置默认配置",
                                     size="small") as reset_mcp_config_btn:
                        with ms.Slot("icon"):
                            antd.Icon("ReloadOutlined")
                with ms.Div():
                    with antd.Tooltip("在 MCP Inspector 中测试待连接的 MCP Server"):
                        with antd.Button(
                                "前往 MCP Inspector 测试",
                                color="primary",
                                variant="outlined",
                                size="small",
                                href_target="_blank",
                                href=
                                "https://modelscope.cn/studios/modelscope/mcp-inspector"
                        ):
                            with ms.Slot("icon"):
                                antd.Icon("ExportOutlined")
            mcp_config = gr.Code(default_mcp_config,
                                 show_label=False,
                                 container=False,
                                 max_lines=20,
                                 lines=3,
                                 language="json")
        with antd.Form.Item(form_name="model", label="模型"):
            with ms.Slot("extra"):
                with ms.Fragment(visible=False) as thought_tip:
                    antd.Typography.Text("Note: 推理模式在调用工具前，会有较长的思考过程，需耐心等待。",
                                         elem_style=dict(fontSize=12),
                                         type="warning")

                with antd.Flex(align="center",
                               gap=4,
                               elem_style=dict(marginTop=4)):
                    ms.Text("Powered by")
                    with antd.Typography.Link(
                            href=
                            "https://modelscope.cn/docs/model-service/API-Inference/intro",
                            href_target="_blank",
                            elem_style=dict(display="flex",
                                            alignItems="center")):
                        antd.Image(
                            "https://gw.alicdn.com/imgextra/i4/O1CN01dCJ2sA1OHUQJFyCRm_!!6000000001680-2-tps-200-200.png",
                            preview=False,
                            width=20,
                            height=20)
                        ms.Text("ModelScope API-Inference")
            with antd.Select(options=model_options) as model_select:
                with ms.Slot("labelRender",
                             params_mapping="""(option) => {
                                const tag = window.MODEL_OPTIONS_MAP[option.value].tag
                                return {
                                    label: option.label, 
                                    link: { href: `https://modelscope.cn/models/${option.value.split(':')[0]}` },  
                                    tag: tag ? { value: tag.label, style: { display: 'inline-block', color: tag.color } } : undefined
                                }
                             }"""):
                    with antd.Flex(gap="small"):
                        antd.Typography.Text(as_item="label")
                        antd.Tag(elem_style=dict(display="none"),
                                 as_item="tag")
                        antd.Typography.Link("模型链接",
                                             href_target="_blank",
                                             as_item="link")
                with ms.Slot("optionRender",
                             params_mapping="""(option) =>  ({ 
                            label: option.data.label.split(':')[0], 
                            tag: option.data.tag ? { value: option.data.tag.label, style: { display: 'inline-block', color: option.data.tag.color } } : undefined 
                        })"""):

                    with antd.Flex(gap="small"):
                        antd.Typography.Text(as_item="label")
                        antd.Tag(elem_style=dict(display="none"),
                                 as_item="tag")

        with antd.Form.Item(form_name="sys_prompt", label="系统提示"):
            antd.Input.Textarea(auto_size=dict(minRows=2, maxRows=4))

    def add_mcp_server(mcp_config_value, add_mcp_server_form_value):
        if not mcp_config_value:
            mcp_config_value = "{}"
        mcp_config = json.loads(mcp_config_value)
        name = add_mcp_server_form_value["name"]
        url = add_mcp_server_form_value["url"]
        if "mcpServers" not in mcp_config:
            mcp_config["mcpServers"] = {}
        mcp_config["mcpServers"][name] = {"type": "sse", "url": url}

        return gr.update(value=json.dumps(mcp_config, indent=4))

    def add_mcp_server_by_json(mcp_config_value,
                               add_mcp_server_json_form_value):
        if not mcp_config_value:
            mcp_config_value = "{}"
        mcp_config = json.loads(mcp_config_value)
        json_value = add_mcp_server_json_form_value["json"]
        json_config = json.loads(json_value)
        if "mcpServers" not in mcp_config:
            mcp_config["mcpServers"] = {}

        mcp_config["mcpServers"].update(json_config.get("mcpServers", {}))

        return gr.update(value=json.dumps(mcp_config, indent=4))

    def select_model(e: gr.EventData):
        return gr.update(visible=e._data["payload"][1].get("thought", False))

    add_mcp_server_form.finish(fn=add_mcp_server,
                               inputs=[mcp_config, add_mcp_server_form],
                               outputs=[mcp_config],
                               queue=False)
    add_mcp_server_json_form.finish(
        fn=add_mcp_server_by_json,
        inputs=[mcp_config, add_mcp_server_json_form],
        outputs=[mcp_config],
        queue=False)
    model_select.change(fn=select_model, outputs=[thought_tip], queue=False)

    return config_form, mcp_config_confirm_btn, reset_mcp_config_btn, mcp_config