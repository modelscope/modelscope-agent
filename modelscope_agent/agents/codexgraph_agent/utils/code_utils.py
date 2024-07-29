import codecs
import re

import json


def extract_and_parse_json(text):
    try:
        start_token = '```json'
        end_token = '```'
        start_index = text.find(start_token)
        end_index = text.rfind(end_token)
        json_str = text[start_index + len(start_token):end_index].strip()
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f'JSON decode error: {e}'


def extract_code_from_file(file_path, start_line, end_line, is_indent=True):
    if start_line < 1:
        start_line = 1
    try:
        with codecs.open(file_path, 'r', encoding='utf-8') as input:
            sourceCode = input.read()
        source_code_lines = sourceCode.split('\n')
        extracted_lines = source_code_lines[start_line - 1:end_line]
    except Exception:
        return ''

    if is_indent:
        first_line_indent = len(extracted_lines[0]) - len(
            extracted_lines[0].lstrip())

        extracted_lines = [
            line[first_line_indent:] if len(line) > first_line_indent else ''
            for line in extracted_lines
        ]

    extracted_code = '\n'.join(extracted_lines)
    return extracted_code


def extract_text_between_markers(text, start_marker, end_marker):
    try:
        # Find the start and end indices of the markers
        start_index = text.find(start_marker)
        end_index = text.find(end_marker, start_index + 1)

        # Check if any of the markers were not found
        if start_index == -1 or end_index == -1:
            return None, 'One or both markers not found'

        # Extract and return the text between the markers
        extracted_text = text[start_index
                              + len(start_marker):end_index].strip()
        if len(extracted_text) == 0:
            return None, 'The text between markers is empty'
        return extracted_text, None

    except Exception as e:
        return None, str(e)


def process_string(input_string, folded_len=10, is_indent=False):
    pattern = re.compile(r'<CODE>(.*?)</CODE>')
    matches = pattern.findall(input_string)

    for match in matches:
        code_dict = json.loads(match)
        file_path = code_dict['F']
        start_line = int(code_dict['S'])
        end_line = int(code_dict['E'])

        code_snippet = extract_code_from_file(
            file_path, start_line, end_line, is_indent=is_indent)

        if len(matches) > 3 and len(code_snippet) > folded_len:
            trimmed_snippet = code_snippet
            folded_snippet = '{0}...(code folded)'.format(
                trimmed_snippet.strip()[:folded_len])
            input_string = input_string.replace(
                '<CODE>{}</CODE>'.format(match), folded_snippet)
        else:
            input_string = input_string.replace(
                '<CODE>{}</CODE>'.format(match), code_snippet)

    return input_string


def process_string_swebench(input_string,
                            env_name,
                            folded_len=20,
                            is_indent=False):
    pattern = re.compile(r'<CODE>(.*?)</CODE>')
    matches = pattern.findall(input_string)
    task_repo, task_index = env_name.rsplit('__', 1)

    for match in matches:
        code_dict = json.loads(match)
        file_path = code_dict['F'].replace(task_repo,
                                           task_repo + '__' + task_index)
        start_line = int(code_dict['S'])
        end_line = int(code_dict['E'])

        code_snippet = extract_code_from_file(
            file_path, start_line, end_line, is_indent=is_indent)

        if len(matches) > 3 and len(code_snippet) > folded_len:
            trimmed_snippet = code_snippet
            folded_snippet = '{0}...(code folded)'.format(
                trimmed_snippet.strip()[:folded_len])
            input_string = input_string.replace(
                '<CODE>{}</CODE>'.format(match), folded_snippet)
        else:
            input_string = input_string.replace(
                '<CODE>{}</CODE>'.format(match), code_snippet)

    return input_string
