import json
from websocket import create_connection

from .base import LLM


class ModelScopeGPT(LLM):
    name = 'ms_gpt'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.url = self.cfg.get('url', '')
        self.token = self.cfg.get('token', '')
        self.generate_cfg = self.cfg.get('generate_cfg', {})

    def generate(self, prompt):

        params = {'input': {'prompt': prompt}}

        params.update(**self.generate_cfg)

        conn = create_connection(
            self.url, timeout=30, header={'Authorization': self.token})

        conn.send(json.dumps(params))

        total_response = ''
        while True:
            result = conn.recv()
            if len(result) <= 0:
                continue
            result = json.loads(result)
            is_final = result['header']['finished']
            new_response = result['payload']['output']['text']
            # print(f'new_response: {new_response}')

            total_response = new_response
            if '<|endofthink|>' in new_response[-25:]:
                conn.close()
                break
            if is_final:
                conn.close()
                break

        conn.close()

        return total_response

    def stream_generate(self, prompt):

        params = {'input': {'prompt': prompt}}

        params.update(**self.generate_cfg)

        conn = create_connection(
            self.url, timeout=30, header={'Authorization': self.token})

        conn.send(json.dumps(params))

        total_response = ''
        while True:
            result = conn.recv()
            if len(result) <= 0:
                continue
            result = json.loads(result)
            is_final = result['header']['finished']
            new_response = result['payload']['output']['text']
            frame_text = new_response[len(total_response):]
            # print(f'new_response: {new_response}')

            if '<|endofthink|>' in new_response[-25:]:
                yield frame_text
                conn.close()
                break
            yield frame_text
            total_response = new_response
            if is_final:
                conn.close()
                break

        conn.close()
