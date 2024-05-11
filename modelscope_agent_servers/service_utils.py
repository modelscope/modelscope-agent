from typing import Dict, Union

import json
from fastapi.responses import JSONResponse
from requests import Response


def create_success_msg(output: Union[Dict, str],
                       request_id: str,
                       message: str = '',
                       **kwargs):
    content = {'request_id': request_id, 'message': message, 'output': output}
    content.update(kwargs)
    return JSONResponse(content=content)


def create_error_msg(message: str, request_id: str, status_code: int = 400):
    return JSONResponse(
        content={
            'request_id': request_id,
            'message': message
        },
        status_code=status_code)


def parse_service_response(response: Response):
    try:
        # Assuming the response is a JSON string
        response_data = response.json()

        # Extract the 'output' field from the response
        output_data = response_data.get('output', {})
        return output_data
    except json.JSONDecodeError:
        # Handle the case where response is not JSON or cannot be decoded
        return None
