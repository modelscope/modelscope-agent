from __future__ import annotations
import base64
import html
import io
import os
import re
from urllib import parse

import json
import markdown
from gradio.components import Chatbot as ChatBotBase
from modelscope_agent.output_parser import MRKLOutputParser
from PIL import Image

ALREADY_CONVERTED_MARK = '<!-- ALREADY CONVERTED BY PARSER. -->'


# 图片本地路径转换为 base64 格式
def covert_image_to_base64(image_path):
    # 获得文件后缀名
    ext = image_path.split('.')[-1]
    if ext not in ['gif', 'jpeg', 'png']:
        ext = 'jpeg'

    with open(image_path, 'rb') as image_file:
        # Read the file
        encoded_string = base64.b64encode(image_file.read())

        # Convert bytes to string
        base64_data = encoded_string.decode('utf-8')

        # 生成base64编码的地址
        base64_url = f'data:image/{ext};base64,{base64_data}'
        return base64_url


def convert_url(text, new_filename):
    # Define the pattern to search for
    # This pattern captures the text inside the square brackets, the path, and the filename
    pattern = r'!\[([^\]]+)\]\(([^)]+)\)'

    # Define the replacement pattern
    # \1 is a backreference to the text captured by the first group ([^\]]+)
    replacement = rf'![\1]({new_filename})'

    # Replace the pattern in the text with the replacement
    return re.sub(pattern, replacement, text)


def format_cover_html(configuration, bot_avatar_path):
    if bot_avatar_path:
        image_src = covert_image_to_base64(bot_avatar_path)
    else:
        image_src = '//img.alicdn.com/imgextra/i3/O1CN01YPqZFO1YNZerQfSBk_!!6000000003047-0-tps-225-225.jpg'
    return f"""
<div class="bot_cover">
    <div class="bot_avatar">
        <img src={image_src} />
    </div>
    <div class="bot_name">{configuration.get("name", "")}</div>
    <div class="bot_desp">{configuration.get("description", "")}</div>
</div>
"""


def format_goto_publish_html(label, zip_url, agent_user_params, disable=False):
    if disable:
        return f"""<div class="publish_link_container">
        <a class="disabled">{label}</a>
    </div>
    """
    else:
        params = {'AGENT_URL': zip_url}
        params.update(agent_user_params)
        template = 'modelscope/agent_template'
        params_str = json.dumps(params)
        link_url = f'https://www.modelscope.cn/studios/fork?target={template}&overwriteEnv={parse.quote(params_str)}'
        return f"""
    <div class="publish_link_container">
        <a href="{link_url}" target="_blank">{label}</a>
    </div>
    """


class ChatBot(ChatBotBase):

    def normalize_markdown(self, bot_message):
        lines = bot_message.split('\n')
        normalized_lines = []
        inside_list = False

        for i, line in enumerate(lines):
            if re.match(r'^(\d+\.|-|\*|\+)\s', line.strip()):
                if not inside_list and i > 0 and lines[i - 1].strip() != '':
                    normalized_lines.append('')
                inside_list = True
                normalized_lines.append(line)
            elif inside_list and line.strip() == '':
                if i < len(lines) - 1 and not re.match(r'^(\d+\.|-|\*|\+)\s',
                                                       lines[i + 1].strip()):
                    normalized_lines.append(line)
                continue
            else:
                inside_list = False
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def convert_markdown(self, bot_message):
        if bot_message.count('```') % 2 != 0:
            bot_message += '\n```'

        bot_message = self.normalize_markdown(bot_message)

        result = markdown.markdown(
            bot_message,
            extensions=[
                'toc', 'extra', 'tables', 'codehilite',
                'markdown_cjk_spacing.cjk_spacing', 'pymdownx.magiclink'
            ],
            extension_configs={
                'markdown_katex': {
                    'no_inline_svg': True,  # fix for WeasyPrint
                    'insert_fonts_css': True,
                },
                'codehilite': {
                    'linenums': False,
                    'guess_lang': True
                },
                'mdx_truly_sane_lists': {
                    'nested_indent': 2,
                    'truly_sane': True,
                }
            })
        result = ''.join(result)
        return result

    @staticmethod
    def prompt_parse(message):
        output = ''
        if 'Thought' in message:
            if 'Action' in message or 'Action Input:' in message:
                re_pattern_thought = re.compile(
                    pattern=r'([\s\S]+)Thought:([\s\S]+)Action:')

                res = re_pattern_thought.search(message)

                if res is None:
                    re_pattern_thought_only = re.compile(
                        pattern=r'Thought:([\s\S]+)Action:')
                    res = re_pattern_thought_only.search(message)
                    llm_result = ''
                else:
                    llm_result = res.group(1).strip()
                action_thought_result = res.group(2).strip()

                re_pattern_action = re.compile(
                    pattern=
                    r'Action:([\s\S]+)Action Input:([\s\S]+)<\|startofexec\|>')
                res = re_pattern_action.search(message)
                if res is None:
                    action, action_parameters = MRKLOutputParser(
                    ).parse_response(message)
                else:
                    action = res.group(1).strip()
                    action_parameters = res.group(2)
                action_result = json.dumps({
                    'api_name': action,
                    'parameters': action_parameters
                })
                output += f'{llm_result}\n{action_thought_result}\n<|startofthink|>\n{action_result}\n<|endofthink|>\n'
            if '<|startofexec|>' in message:
                re_pattern3 = re.compile(
                    pattern=r'<\|startofexec\|>([\s\S]+)<\|endofexec\|>')
                res3 = re_pattern3.search(message)
                observation = res3.group(1).strip()
                output += f'\n<|startofexec|>\n{observation}\n<|endofexec|>\n'
            if 'Final Answer' in message:
                re_pattern2 = re.compile(
                    pattern=r'Thought:([\s\S]+)Final Answer:([\s\S]+)')
                res2 = re_pattern2.search(message)
                # final_thought_result = res2.group(1).strip()
                final_answer_result = res2.group(2).strip()
                output += f'{final_answer_result}\n'

            if output == '':
                return message
            print(output)
            return output
        else:
            return message

    def convert_bot_message(self, bot_message):

        bot_message = ChatBot.prompt_parse(bot_message)
        # print('processed bot message----------')
        # print(bot_message)
        # print('processed bot message done')
        start_pos = 0
        result = ''
        find_json_pattern = re.compile(r'{[\s\S]+}')
        START_OF_THINK_TAG, END_OF_THINK_TAG = '<|startofthink|>', '<|endofthink|>'
        START_OF_EXEC_TAG, END_OF_EXEC_TAG = '<|startofexec|>', '<|endofexec|>'
        while start_pos < len(bot_message):
            try:
                start_of_think_pos = bot_message.index(START_OF_THINK_TAG,
                                                       start_pos)
                end_of_think_pos = bot_message.index(END_OF_THINK_TAG,
                                                     start_pos)
                if start_pos < start_of_think_pos:
                    result += self.convert_markdown(
                        bot_message[start_pos:start_of_think_pos])
                think_content = bot_message[start_of_think_pos
                                            + len(START_OF_THINK_TAG
                                                  ):end_of_think_pos].strip()
                json_content = find_json_pattern.search(think_content)
                think_content = json_content.group(
                ) if json_content else think_content
                try:
                    think_node = json.loads(think_content)
                    plugin_name = think_node.get(
                        'plugin_name',
                        think_node.get('plugin',
                                       think_node.get('api_name', 'unknown')))
                    summary = f'选择插件【{plugin_name}】，调用处理中...'
                    del think_node['url']
                    # think_node.pop('url', None)

                    detail = f'```json\n\n{json.dumps(think_node, indent=3, ensure_ascii=False)}\n\n```'
                except Exception:
                    summary = '思考中...'
                    detail = think_content
                    # traceback.print_exc()
                    # detail += traceback.format_exc()
                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'
                # print(f'detail:{detail}')
                start_pos = end_of_think_pos + len(END_OF_THINK_TAG)
            except Exception:
                # result += traceback.format_exc()
                break
                # continue

            try:
                start_of_exec_pos = bot_message.index(START_OF_EXEC_TAG,
                                                      start_pos)
                end_of_exec_pos = bot_message.index(END_OF_EXEC_TAG, start_pos)
                # print(start_of_exec_pos)
                # print(end_of_exec_pos)
                # print(bot_message[start_of_exec_pos:end_of_exec_pos])
                # print('------------------------')
                if start_pos < start_of_exec_pos:
                    result += self.convert_markdown(
                        bot_message[start_pos:start_of_think_pos])
                exec_content = bot_message[start_of_exec_pos
                                           + len(START_OF_EXEC_TAG
                                                 ):end_of_exec_pos].strip()
                try:
                    summary = '完成插件调用.'
                    detail = f'```json\n\n{exec_content}\n\n```'
                except Exception:
                    pass

                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'

                start_pos = end_of_exec_pos + len(END_OF_EXEC_TAG)
            except Exception:
                # result += traceback.format_exc()
                continue
        if start_pos < len(bot_message):
            result += self.convert_markdown(bot_message[start_pos:])
        result += ALREADY_CONVERTED_MARK
        return result

    def convert_bot_message_for_qwen(self, bot_message):

        start_pos = 0
        result = ''
        find_json_pattern = re.compile(r'{[\s\S]+}')
        ACTION = 'Action:'
        ACTION_INPUT = 'Action Input'
        OBSERVATION = 'Observation'
        RESULT_START = '<result>'
        RESULT_END = '</result>'
        while start_pos < len(bot_message):
            try:
                action_pos = bot_message.index(ACTION, start_pos)
                action_input_pos = bot_message.index(ACTION_INPUT, start_pos)
                result += self.convert_markdown(
                    bot_message[start_pos:action_pos])
                # Action: image_gen
                # Action Input
                # {"text": "金庸武侠 世界", "resolution": "1280x720"}
                # Observation: <result>![IMAGEGEN](https://dashscope-result-sh.oss-cn-shanghai.aliyuncs.com/1d/e9/20231116/723609ee/d046d2d9-0c95-420b-9467-f0e831f5e2b7-1.png?Expires=1700227460&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=R0PlEazQF9uBD%2Fh9tkzOkJMGyg8%3D)<result> # noqa E501
                action_name = bot_message[action_pos
                                          + len(ACTION
                                                ):action_input_pos].strip()
                # action_start action_end 使用 Action Input 到 Observation 之间
                action_input_end = bot_message[action_input_pos:].index(
                    OBSERVATION) - 1
                action_input = bot_message[action_input_pos:action_input_pos
                                           + action_input_end].strip()
                is_json = find_json_pattern.search(action_input)
                if is_json:
                    action_input = is_json.group()
                else:
                    action_input = re.sub(r'^Action Input[:]?[\s]*', '',
                                          action_input)

                summary = f'调用工具 {action_name}'
                if is_json:
                    detail = f'```json\n\n{json.dumps(json.loads(action_input), indent=4, ensure_ascii=False)}\n\n```'
                else:
                    detail = action_input
                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'
                start_pos = action_input_pos + action_input_end + 1
                try:
                    observation_pos = bot_message.index(OBSERVATION, start_pos)
                    idx = observation_pos + len(OBSERVATION)
                    obs_message = bot_message[idx:]
                    observation_start_id = obs_message.index(
                        RESULT_START) + len(RESULT_START)
                    observation_end_idx = obs_message.index(RESULT_END)
                    summary = '完成调用'
                    exec_content = obs_message[
                        observation_start_id:observation_end_idx]
                    detail = f'```\n\n{exec_content}\n\n```'
                    start_pos = idx + observation_end_idx + len(RESULT_END)
                except Exception:
                    summary = '执行中...'
                    detail = ''
                    exec_content = None

                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'
                if exec_content is not None and '[IMAGEGEN]' in exec_content:
                    # convert local file to base64
                    re_pattern = re.compile(pattern=r'!\[[^\]]+\]\(([^)]+)\)')
                    res = re_pattern.search(exec_content)
                    if res:
                        image_path = res.group(1).strip()
                        if os.path.isfile(image_path):
                            exec_content = convert_url(
                                exec_content,
                                covert_image_to_base64(image_path))
                    result += self.convert_markdown(f'{exec_content}')

            except Exception:
                # import traceback; traceback.print_exc()
                result += self.convert_markdown(bot_message[start_pos:])
                start_pos = len(bot_message[start_pos:])
                break

        result += ALREADY_CONVERTED_MARK
        return result

    def postprocess(
        self,
        message_pairs: list[list[str | tuple[str] | tuple[str, str] | None]
                            | tuple],
    ) -> list[list[str | dict | None]]:
        """
        Parameters:
            message_pairs: List of lists representing the message and response pairs.
            Each message and response should be a string, which may be in Markdown format.
            It can also be a tuple whose first element is a string or pathlib.
            Path filepath or URL to an image/video/audio, and second (optional) element is the alt text,
            in which case the media file is displayed. It can also be None, in which case that message is not displayed.
        Returns:
            List of lists representing the message and response. Each message and response will be a string of HTML,
            or a dictionary with media information. Or None if the message is not to be displayed.
        """
        if message_pairs is None:
            return []
        processed_messages = []
        for message_pair in message_pairs:
            assert isinstance(
                message_pair, (tuple, list)
            ), f'Expected a list of lists or list of tuples. Received: {message_pair}'
            assert (
                len(message_pair) == 2
            ), f'Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}'
            if isinstance(message_pair[0], tuple) or isinstance(
                    message_pair[1], tuple):
                processed_messages.append([
                    self._postprocess_chat_messages(message_pair[0]),
                    self._postprocess_chat_messages(message_pair[1]),
                ])
            else:
                # 处理不是元组的情况
                user_message, bot_message = message_pair

                if user_message and not user_message.endswith(
                        ALREADY_CONVERTED_MARK):
                    convert_md = self.convert_markdown(
                        html.escape(user_message))
                    user_message = f'{convert_md}' + ALREADY_CONVERTED_MARK
                if bot_message and not bot_message.endswith(
                        ALREADY_CONVERTED_MARK):
                    # bot_message = self.convert_bot_message(bot_message)
                    bot_message = self.convert_bot_message_for_qwen(
                        bot_message)
                processed_messages.append([
                    user_message,
                    bot_message,
                ])

        return processed_messages
