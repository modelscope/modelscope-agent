from .base import register_llm
from .modelscope import ModelScopeLLM


@register_llm('chatglm_modelscope')
class ModelScopeChatGLM(ModelScopeLLM):

    def _inference(self, prompt: str) -> str:
        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command('<|user|>'),
            self.tokenizer.get_command('<|observation|>')
        ]
        result = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_cfg,
            eos_token_id=eos_token_id)

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)
        # 遇到生成'<', '|', 'user', '|', '>'的case
        response = response.split('<|user|>')[0].split('<|observation|>')[0]

        return response
