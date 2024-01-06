from .prompt import LengthConstraint, PromptGenerator

MS_DEFAULT_SYSTEM_TEMPLATE = """<|system|>:你是达摩院的ModelScopeGPT（魔搭助手），你是个大语言模型， 是2023年达摩院的工程师训练得到的。\
你有多种能力，可以通过插件集成魔搭社区的模型api来回复用户的问题，还能解答用户使用模型遇到的问题和模型知识相关问答。
"""

MS_DEFAULT_INSTRUCTION_TEMPLATE = """当前对话可以使用的插件信息如下，请自行判断是否需要调用插件来解决当前用户问题。若需要调用插件，则需要将插件调用请求按照json格式给出，必须包含api_name、parameters字段，并在其前后使用<|startofthink|>和<|endofthink|>作为标志。\
然后你需要根据插件API调用结果生成合理的答复； 若无需调用插件，则直接给出对应回复即可。\n\n<tool_list>"""

MS_DEFAULT_USER_TEMPLATE = """<|user|>:<user_input>"""

MS_DEFAULT_EXEC_TEMPLATE = """<|startofexec|><exec_result><|endofexec|>\n"""

MS_DEFAULT_ASSISTANT_TEMPLATE = """<|assistant|>:"""


class MSPromptGenerator(PromptGenerator):

    def __init__(self,
                 system_template=MS_DEFAULT_SYSTEM_TEMPLATE,
                 instruction_template=MS_DEFAULT_INSTRUCTION_TEMPLATE,
                 user_template=MS_DEFAULT_USER_TEMPLATE,
                 exec_template=MS_DEFAULT_EXEC_TEMPLATE,
                 assistant_template=MS_DEFAULT_ASSISTANT_TEMPLATE,
                 sep='\n\n',
                 length_constraint=LengthConstraint(),
                 **kwargs):
        super().__init__(
            system_template=system_template,
            instruction_template=instruction_template,
            user_template=user_template,
            exec_template=exec_template,
            assistant_template=assistant_template,
            sep=sep,
            length_constraint=length_constraint,
            **kwargs)
