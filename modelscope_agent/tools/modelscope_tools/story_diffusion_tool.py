import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from modelscope_agent.tools.base import BaseTool, register_tool


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cal_attn_mask_xl(total_length,
                     id_length,
                     sa32,
                     sa64,
                     height,
                     width,
                     device='cuda',
                     dtype=torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand(
        (1, total_length * nums_1024), device=device, dtype=dtype) < sa32
    bool_matrix4096 = torch.rand(
        (1, total_length * nums_4096), device=device, dtype=dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length, 1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length, 1)
    for i in range(total_length):
        bool_matrix1024[i:i + 1, id_length * nums_1024:] = False
        bool_matrix4096[i:i + 1, id_length * nums_4096:] = False
        bool_matrix1024[i:i + 1, i * nums_1024:(i + 1) * nums_1024] = True
        bool_matrix4096[i:i + 1, i * nums_4096:(i + 1) * nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1, nums_1024, 1).reshape(
        -1, total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1, nums_4096, 1).reshape(
        -1, total_length * nums_4096)
    return mask1024, mask4096


class AttnProcessor(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError(
                'AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.'
            )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None else encoder_hidden_states.shape)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(
        self,
        global_attn_args,
        hidden_size=None,
        cross_attention_dim=None,
        id_length=4,
        device='cuda',
        dtype=torch.float16,
        height=1280,
        width=720,
        sa32=0.5,
        sa64=0.5,
    ):
        super().__init__()
        if not hasattr(F, 'scaled_dot_product_attention'):
            raise ImportError(
                'AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.'
            )
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}
        self.height = height
        self.width = width
        self.sa32 = sa32
        self.sa64 = sa64
        self.write = True

        self.global_attn_args = global_attn_args

    def __call__(self,
                 attn,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 temb=None):
        total_count = self.global_attn_args['total_count']
        attn_count = self.global_attn_args['attn_count']
        cur_step = self.global_attn_args['cur_step']
        mask1024 = self.global_attn_args['mask1024']
        mask4096 = self.global_attn_args['mask4096']

        if self.write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [
                hidden_states[:self.id_length], hidden_states[self.id_length:]
            ]
        else:
            encoder_hidden_states = torch.cat(
                (self.id_bank[cur_step][0].to(self.device), hidden_states[:1],
                 self.id_bank[cur_step][1].to(self.device), hidden_states[1:]))
        # skip in early step
        if cur_step < 5:
            hidden_states = self.__call2__(attn, hidden_states,
                                           encoder_hidden_states,
                                           attention_mask, temb)
        else:  # 256 1024 4096
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not self.write:
                    if hidden_states.shape[1] == (self.height
                                                  // 32) * (self.width // 32):
                        attention_mask = mask1024[mask1024.shape[0]
                                                  // self.total_length
                                                  * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0]
                                                  // self.total_length
                                                  * self.id_length:]
                else:
                    if hidden_states.shape[1] == (self.height
                                                  // 32) * (self.width // 32):
                        attention_mask = mask1024[:mask1024.shape[0]
                                                  // self.total_length * self.
                                                  id_length, :mask1024.shape[0]
                                                  // self.total_length
                                                  * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0]
                                                  // self.total_length * self.
                                                  id_length, :mask4096.shape[0]
                                                  // self.total_length
                                                  * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states,
                                               encoder_hidden_states,
                                               attention_mask, temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states, None,
                                               attention_mask, temb)
        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(
                self.total_length,
                self.id_length,
                self.sa32,
                self.sa64,
                self.height,
                self.width,
                device=self.device,
                dtype=self.dtype)
            self.global_attn_args['mask1024'] = mask1024
            self.global_attn_args['mask4096'] = mask4096

        self.global_attn_args['attn_count'] = attn_count
        self.global_attn_args['cur_step'] = cur_step

        return hidden_states

    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel,
                                               height * width).transpose(1, 2)
        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token,
                                           channel).reshape(
                                               -1, img_nums * nums_token,
                                               channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, nums_token,
                channel).reshape(-1, (self.id_length + 1) * nums_token,
                                 channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states

    def __call2__(self,
                  attn,
                  hidden_states,
                  encoder_hidden_states=None,
                  attention_mask=None,
                  temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel,
                                               height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (hidden_states.shape)
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(
                -1, self.id_length + 1, sequence_length,
                channel).reshape(-1, (self.id_length + 1) * sequence_length,
                                 channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StoryDiffusionTool(BaseTool):
    default_model: str = ''
    task: str = ''
    model_revision = None
    name: str = 'story-diffusion'
    description: str = '根据给定的描述，返回主角描绘一致的故事绘本漫画'

    parameters: list = [{
        'name': 'prompts',
        'description': '用户输入的画面描述列表，每一项为一张画面的描述',
        'required': True,
        'type': 'list'
    }, {
        'name': 'seed',
        'description': '随机数种子',
        'required': False,
        'type': 'int'
    }, {
        'name': 'style_name',
        'description': '画图风格名称',
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

        self.attn_args = {
            'attn_count': 0,
            'cur_step': 0,
            'total_count': 0,
        }
        self.sa32 = 0.5
        self.sa64 = 0.5
        self.id_length = self.cfg.get('id_length', 4)
        self.total_length = self.cfg['num_pages']
        self.id_length = min(self.id_length, self.total_length)
        self.height = self.cfg.get('height', 768)
        self.width = self.cfg.get('width', 768)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16
        self.num_steps = self.cfg.get('num_steps', 50)
        self.styles = {
            '(No style)': ('{prompt}', ''),
            'Japanese Anime':
            ('anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. '
             'best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, '
             'line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, '
             'Linear compositions, Scalable artwork, Digital art, High Contrast Shadows',
             'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, '
             'cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, '
             'username, blurry'),
            'Digital/Oil Painting':
            ('{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, '
             '8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, '
             'Sharp focus, humorous illustration, big depth of field',
             'anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, '
             'ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, '
             'fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, '
             'watermark, username, blurry'),
            'Pixar/Disney Character':
            ('Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, '
             'filled with vivid colors and a sense of wonder.',
             'lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, '
             'extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, '
             'signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, '
             'ultra textured, photo'),
            'Photographic':
            ('cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, '
             'realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, '
             'soft focus',
             'drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, '
             'lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, '
             'worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry'
             ),
            'Comic book':
            ('comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed',
             'photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, '
             'error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, '
             'normal quality, jpeg artifacts, signature, watermark, username, blurry'
             ),
            'Line art':
            ('line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, '
             'vector graphics',
             'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, '
             'cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, '
             'expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, '
             'extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, '
             'signature, watermark, username, blurry'),
            'Black and White Film Noir':
            ('{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, '
             'subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k',
             'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, '
             'cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, '
             'expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, '
             'extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, '
             'signature, watermark, username, blurry'),
            'Isometric Rooms':
            ('Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, '
             '3d blender render',
             'anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, '
             'cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, '
             'expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, '
             'extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, '
             'signature, watermark, username, blurry')
        }

        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.cfg.get("real_vis_path", 'SG161222/RealVisXL_V4.0'),
            torch_dtype=torch.float16,
            use_safetensors=False)

        pipe = pipe.to(self.device)

        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(self.num_steps)
        unet = pipe.unet

        attn_procs = {}
        # Insert PairedAttention
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith(
                'attn1.processor') else unet.config.cross_attention_dim
            # if name.startswith("mid_block"):
            #     hidden_size = unet.config.block_out_channels[-1]
            # elif name.startswith("up_blocks"):
            #     block_id = int(name[len("up_blocks.")])
            #     hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            # elif name.startswith("down_blocks"):
            #     block_id = int(name[len("down_blocks.")])
            #     hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None and (name.startswith('up_blocks')):
                attn_procs[name] = SpatialAttnProcessor2_0(
                    id_length=self.id_length,
                    device=self.device,
                    height=self.height,
                    width=self.width,
                    sa32=self.sa32,
                    sa64=self.sa64,
                    global_attn_args=self.attn_args)
                self.attn_args['total_count'] += 1
            else:
                attn_procs[name] = AttnProcessor()
        print('successsfully load consistent self-attention')
        print(f"number of the processor : {self.attn_args['total_count']}")
        # unet.set_attn_processor(copy.deepcopy(attn_procs))
        unet.set_attn_processor(attn_procs)
        mask1024, mask4096 = cal_attn_mask_xl(
            self.total_length,
            self.id_length,
            self.sa32,
            self.sa64,
            self.height,
            self.width,
            device=self.device,
            dtype=torch.float16,
        )

        self.attn_args.update({'mask1024': mask1024, 'mask4096': mask4096})

        self.pipe = pipe
        self.negative_prompt = 'naked, deformed, bad anatomy, disfigured, poorly drawn face, ' \
                               'mutation, extra limb, ugly, disgusting, poorly drawn hands, ' \
                               'missing limb, floating limbs, disconnected limbs, blurry, ' \
                               'watermarks, oversaturated, distorted hands, amputation'

        self.is_initialized = True

    def set_attn_write(self, value: bool):
        unet = self.pipe.unet
        for name, processor in unet.attn_processors.items():
            cross_attention_dim = None if name.endswith(
                'attn1.processor') else unet.config.cross_attention_dim
            if cross_attention_dim is None:
                if name.startswith('up_blocks'):
                    assert isinstance(processor, SpatialAttnProcessor2_0)
                    processor.write = value

    def apply_style(self,
                    style_name: str,
                    positives: list,
                    negative: str = ''):
        p, n = self.styles.get(style_name, self.styles['(No style)'])
        return [p.replace('{prompt}', positive)
                for positive in positives], n + ' ' + negative

    def apply_style_positive(self, style_name: str, positive: str):
        p, n = self.styles.get(style_name, self.styles['(No style)'])
        return p.replace('{prompt}', positive)

    def call(self, params: str, **kwargs):
        params: dict = self._verify_args(params)
        if isinstance(params, str):
            return 'Parameter Error'

        try:
            self.setup()
            prompts = params['prompts']
            seed = params.get('seed', 0)
            style_name = params.get('style_name', 'Japanese Anime')
            guidance_scale = params.get('guidance_scale', 5.0)
            assert len(
                prompts
            ) == self.total_length, 'The number of prompts should be equal to the number of pages.'
            setup_seed(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            torch.cuda.empty_cache()

            id_prompts = prompts[:self.id_length]
            real_prompts = prompts[self.id_length:]
            self.set_attn_write(True)
            self.attn_args.update({'cur_step': 0, 'attn_count': 0})
            id_prompts, negative_prompt = self.apply_style(
                style_name, id_prompts, self.negative_prompt)
            id_images = self.pipe(
                id_prompts,
                num_inference_steps=self.num_steps,
                guidance_scale=guidance_scale,
                height=self.height,
                width=self.width,
                negative_prompt=negative_prompt,
                generator=generator).images

            self.set_attn_write(False)
            real_images = []
            for real_prompt in real_prompts:
                self.attn_args['cur_step'] = 0
                real_prompt = self.apply_style_positive(
                    style_name, real_prompt)
                real_images.append(
                    self.pipe(
                        real_prompt,
                        num_inference_steps=self.num_steps,
                        guidance_scale=guidance_scale,
                        height=self.height,
                        width=self.width,
                        negative_prompt=negative_prompt,
                        generator=generator).images[0])

            images = id_images + real_images
            return images
        except RuntimeError as e:
            import traceback
            raise RuntimeError(
                f'Call failed with error: {e}, with detail {traceback.format_exc()}'
            )
