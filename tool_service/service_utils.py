from typing import Dict

import json


def create_success_msg(output: Dict, request_id: str, message: str = ''):
    response_body = {
        'status_code': 200,
        'request_id': request_id,
        'message': '',
        'output': output
    }
    return response_body


def create_error_msg(message: str, request_id: str, code: int = 400):
    response_body = {
        'status_code': code,
        'request_id': request_id,
        'message': message
    }
    return response_body


def parse_service_response(response):
    try:
        # Assuming the response is a JSON string
        response_data = response.json()

        # Extract the 'output' field from the response
        output_data = response_data.get('output', {})
        return output_data
    except json.JSONDecodeError:
        # Handle the case where response is not JSON or cannot be decoded
        return None
