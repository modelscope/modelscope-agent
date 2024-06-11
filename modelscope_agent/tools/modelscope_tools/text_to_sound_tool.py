from typing import Dict, Optional

import torch
from diffusers import AudioLDM2Pipeline
from modelscope_agent.tools.base import BaseTool


class TexttoSoundTool(BaseTool):
    default_model: str = ''
    task: str = ''
    model_revision = None
    name: str = 'text-to-sound'
    description: str = '根据给定的描述，生成符合描述的音效'

    parameters: list = [{
        'name': 'prompts',
        'description': '用户输入的音效描述列表',
        'required': True,
        'type': 'list'
    }, {
        'name': 'seed',
        'description': '随机数种子',
        'required': False,
        'type': 'int'
    }, {
        'name': 'n_candidate_per_text',
        'description': '每个音效描述生成备选音效的数量（生成多个，筛选其中最好的）',
        'required': False,
        'type': 'string'
    }, {
        'name': 'guidance_scale',
        'description': 'CFG 的引导比例',
        'required': False,
        'type': 'float'
    }]

    def __init__(self, cfg: Optional[Dict] = {}):
        super().__init__(cfg)
        self.is_initialized = False

    def setup(self):
        if self.is_initialized:
            return

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = AudioLDM2Pipeline.from_pretrained(
            self.cfg.get("audioldm2_path", 'cvssp/audioldm2')
        )
        self.pipeline = self.pipeline.to(self.device)
        self.is_initialized = True

    def call(self, params: str, **kwargs):
        params: dict = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        try:
            self.setup()
            prompts = params['prompts']
            seed = params.get('seed', 0)
            n_candidate_per_text = params.get('n_candidate_per_text', 3)
            guidance_scale = params.get('guidance_scale', 3.5)
            n_steps = params.get('n_steps', 200)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            audios = self.pipeline(
                prompts,
                num_inference_steps=n_steps,
                audio_length_in_s=10.0,
                guidance_scale=guidance_scale,
                num_waveforms_per_prompt=n_candidate_per_text,
                generator=generator).audios
            audios = audios[::n_candidate_per_text]
            return audios

        except RuntimeError as e:
            import traceback
            raise RuntimeError(
                f'Call failed with error: {e}, with detail {traceback.format_exc()}'
            )
