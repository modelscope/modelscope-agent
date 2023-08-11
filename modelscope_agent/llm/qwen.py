from modelscope import GenerationConfig
from .local_llm import LocalLLM


class QWen(LocalLLM):

    def __init__(self, cfg):
        self.name = 'qwen-7b'
        super().__init__(self.name, cfg)
        self.model.generation_config = GenerationConfig.from_pretrained(
            'qwen/Qwen-7B-Chat', revision='v1.0.1', trust_remote_code=True)
        self.model.generation_config.stop_words_ids.append([37763, 367, 25])

        # self.setup()

    def generate(self, prompt):
        response = self.model.chat(
            self.tokenizer, prompt, history=[], stream=False)[0]
        # 去除observation
        idx = response.find('Observation')
        if idx != -1:
            response = response[:idx]
        return response
