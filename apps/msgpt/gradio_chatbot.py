from __future__ import annotations
import ast
import html
import re
import traceback
from typing import List, Tuple

import gradio as gr
import json
import markdown
from gradio.components import Chatbot as ChatBotBase

ALREADY_CONVERTED_MARK = '<!-- ALREADY CONVERTED BY PARSER. -->'


class ChatBot(ChatBotBase):

    def normalize_markdown(self, bot_message, remove_media=False):

        if remove_media:
            media_regex = r'(!\[[^\]]*\]\([^)]+\)|<audio[^>]+>.*?</audio>)\n*'
            # 使用正则表达式进行替换
            bot_message = re.sub(media_regex, '', bot_message)

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

    def convert_markdown(self, bot_message, remove_media=False):
        if bot_message.count('```') % 2 != 0:
            bot_message += '\n```'

        bot_message = self.normalize_markdown(bot_message, remove_media)

        result = markdown.markdown(
            bot_message,
            extensions=[
                'toc', 'extra', 'tables', 'markdown_katex', 'codehilite',
                'mdx_truly_sane_lists', 'markdown_cjk_spacing.cjk_spacing',
                'pymdownx.magiclink'
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

    def convert_bot_message(self, bot_message):

        # 兼容老格式
        chunks = bot_message.split('<extra_id_0>')
        if len(chunks) > 1:
            new_bot_message = ''
            for idx, chunk in enumerate(chunks):
                new_bot_message += chunk
                if idx % 2 == 0:
                    if idx != len(chunks) - 1:
                        new_bot_message += '<|startofthink|>'
                else:
                    new_bot_message += '<|endofthink|>'

            bot_message = new_bot_message

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
                    think_node = json.loads(
                        think_content.replace('\n', ''), strict=False)
                    plugin_name = think_node.get(
                        'plugin_name',
                        think_node.get('plugin',
                                       think_node.get('api_name', 'unknown')))
                    summary = f'选择插件【{plugin_name}】'
                    think_node.pop('url', None)
                    detail = f'```json\n\n{json.dumps(think_node,indent=3,ensure_ascii=False)}\n\n```'
                except Exception:
                    traceback.print_exc()
                    summary = '思考中...'
                    detail = think_content
                    # detail += traceback.format_exc()
                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    detail) + '</details>'
                start_pos = end_of_think_pos + len(END_OF_THINK_TAG)
            except Exception:
                # result += traceback.format_exc()
                break

            try:
                start_of_exec_pos = bot_message.index(START_OF_EXEC_TAG,
                                                      start_pos)
                end_of_exec_pos = bot_message.index(END_OF_EXEC_TAG, start_pos)
                if start_pos < start_of_exec_pos:
                    result += self.convert_markdown(
                        bot_message[start_pos:start_of_think_pos])
                exec_content = bot_message[start_of_exec_pos
                                           + len(START_OF_EXEC_TAG
                                                 ):end_of_exec_pos].strip()
                exec_content = self.process_exec_result(exec_content)

                # result += self.convert_markdown(exec_content)
                summary = '执行结果'
                result += '<details> <summary>' + summary + '</summary>' + self.convert_markdown(
                    exec_content) + '</details>'
                start_pos = end_of_exec_pos + len(END_OF_EXEC_TAG)
            except Exception:
                # traceback.print_exc()
                break
        if start_pos < len(bot_message):
            result += self.convert_markdown(
                bot_message[start_pos:], remove_media=True)
        result += ALREADY_CONVERTED_MARK
        return result

    def postprocess(
        self, message_pairs: List[Tuple[str | None, str | None]]
    ) -> List[Tuple[str | None, str | None]]:
        """
        Parameters:
            y: List of tuples representing the message and response pairs.
            Each message and response should be a string, which may be in Markdown format.
        Returns:
            List of tuples representing the message and response. Each message and response will be a string of HTML.
        """
        if not message_pairs:
            return []
        user_message, bot_message = message_pairs[-1]
        if not user_message.endswith(ALREADY_CONVERTED_MARK):
            user_message = f"<p style=\"white-space:pre-wrap;\">{self.convert_markdown(html.escape(user_message))}</p>"\
                + ALREADY_CONVERTED_MARK
        if not bot_message.endswith(ALREADY_CONVERTED_MARK):
            bot_message = self.convert_bot_message(bot_message)
        message_pairs[-1] = (user_message, bot_message)
        return message_pairs

    def process_exec_result(self, exec_result: str):

        exec_result = exec_result.replace("{'result': ", '')
        exec_result = exec_result[:-1]
        exec_result = exec_result.replace("'", "\"")
        try:
            exec_result = json.loads(
                exec_result.replace('\n', ''), strict=False)
            final_result = f'```json\n\n{exec_result}\n\n```'
            return final_result
        except Exception:
            match_image = re.search(r'!\[IMAGEGEN\]\((.*?)\)', exec_result)
            if match_image:
                img_path = match_image.group(1)

                gr_img_path = self.transform_to_gr_file(img_path)
                final_result = exec_result.replace(img_path, gr_img_path)
                return final_result

            match_audio = re.search(
                r'<audio id=audio controls= preload=none> <source id=wav src="(.*?)"> <\/audio>',
                exec_result)
            if match_audio:
                audio_path = match_audio.group(1)
                gr_audio_path = self.transform_to_gr_file(audio_path)

                final_result = exec_result.replace(audio_path, gr_audio_path)
                return final_result

            final_result = exec_result
            return final_result

    def transform_to_gr_file(self, file_path):
        file_manager = gr.File()
        gr_file_path = file_manager.make_temp_copy_if_needed(file_path)

        gr_file_path = f'./file={gr_file_path}'

        return gr_file_path
