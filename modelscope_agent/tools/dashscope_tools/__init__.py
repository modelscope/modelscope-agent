import sys

from modelscope_agent.utils import _LazyModule

_import_structure = {
    'image_enhancement': ['ImageEnhancement'],
    'image_generation': ['TextToImageTool'],
    'image_generation_lite': ['TextToImageLiteTool'],
    'qwen_vl': ['QWenVL'],
    'style_repaint': ['StyleRepaint'],
    'wordart_tool': ['WordArtTexture'],
    'sambert_tts_tool': ['SambertTtsTool'],
    'paraformer_asr_tool': ['ParaformerAsrTool']
}

sys.modules[__name__] = _LazyModule(
    __name__,
    globals()['__file__'],
    _import_structure,
    module_spec=__spec__,
)
