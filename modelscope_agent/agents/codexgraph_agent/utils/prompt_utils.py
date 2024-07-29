import os
from string import Template


def response_to_msg(thought, action, action_input):
    msg = f"""
## thought
{thought}
## action
{action}
## action_input
{action_input}
    """
    return msg


def build_system_prompt(folder_path, schema_path, language='python'):
    primary_system_prompt_path = os.path.join(folder_path, language,
                                              'system_prompt_primary.txt')
    cypher_system_prompt_path = os.path.join(folder_path, language,
                                             'system_prompt_cypher.txt')

    with open(primary_system_prompt_path, 'r') as f:
        primary_system_prompt = f.read()
    with open(cypher_system_prompt_path, 'r') as f:
        cypher_system_prompt = f.read()

    if language == 'python':
        db_schema_path = os.path.join(schema_path, 'python', 'schema.txt')
        with open(db_schema_path, 'r') as f:
            db_schema = f.read()
        primary_system_prompt = primary_system_prompt.replace(
            '{{python_db_schema}}', db_schema)
        cypher_system_prompt = cypher_system_prompt.replace(
            '{{python_db_schema}}', db_schema)

    return primary_system_prompt, cypher_system_prompt


def load_prompt_template(file_path, prompt_file, language='python'):
    prompt_file_path = os.path.join(file_path, language, prompt_file)
    with open(prompt_file_path, 'r') as f:
        user_prompt = f.read()
    return Template(user_prompt)


def replace_system_prompt(messages, prompt: str):
    """
    Replace the system prompt in the message thread.
    This is because the main agent system prompt main invole tool_calls info, which
    should not be known to task agents.
    """
    messages[0]['content'] = prompt
    return messages
