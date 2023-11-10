class CustomOutputWrapper:

    @staticmethod
    def handle_message_chat_completion(response):
        message = {'content': ''}
        try:
            message = response['choices'][0]['message']
        except Exception as e:
            print(f'input: {response}, original error: {str(e)}')
            return message

    @staticmethod
    def handle_message_chat_completion_chunk(response):
        message = {}
        try:
            return response['choices'][0]['delta']['content']
        except Exception as e:
            print(f'input: {response}, original error: {str(e)}')
            return message

    @staticmethod
    def handle_message_text_completion(response):
        message = ''
        try:
            message = response['output']['text']
            return message
        except Exception as e:
            print(f'input: {response}, original error: {str(e)}')
            return message


DEFAULT_MESSAGE = {
    'role': 'user',
    'content': 'No entry from user - please suggest something to enter'
}
