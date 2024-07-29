import re


def add_label_to_nodes(cypher_query, new_label):
    split_query = cypher_query.split('RETURN', 1)

    if len(split_query) < 2:
        match_part = split_query[0]
        return_part = ''
    else:
        match_part = split_query[0]
        return_part = 'RETURN' + split_query[1]

    pattern = r'(\(\s*\w*\s*)(:[^{}()]*)?(\{[^{}()]*\})?\)'

    def replace_label(match):
        before_label = match.group(1)
        existing_labels = match.group(2) if match.group(2) else ''
        properties = match.group(3) if match.group(3) else ''

        if new_label not in existing_labels:
            if existing_labels:
                new_labels = f':{new_label}{existing_labels}'
            else:
                new_labels = f':{new_label}'
        else:
            new_labels = existing_labels

        return f'{before_label}{new_labels}{properties})'

    updated_match_part = re.sub(pattern, replace_label, match_part)
    updated_query = updated_match_part + return_part

    return updated_query


def extract_cypher_queries(given_text):
    pattern = re.compile(r'```cypher(.*?)```', re.DOTALL)
    return [match.strip() for match in pattern.findall(given_text)]
