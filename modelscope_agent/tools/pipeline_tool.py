from modelscope.pipelines import pipeline
from .tool import Tool


class ModelscopePipelineTool(Tool):

    default_model: str = ''
    task: str = ''
    model_revision = None

    def __init__(self, cfg):

        super().__init__(cfg)
        self.model = self.cfg.get('model', None) or self.default_model
        self.model_revision = self.cfg.get('model_revision',
                                           None) or self.model_revision

        self.pipeline = None
        self.is_initialized = False

    def setup(self):

        # only initialize when this tool is really called to save memory
        if not self.is_initialized:
            self.pipeline = pipeline(
                task=self.task,
                model=self.model,
                model_revision=self.model_revision)
        self.is_initialized = True

    def _local_call(self, *args, **kwargs):

        self.setup()

        parsed_args, parsed_kwargs = self._local_parse_input(*args, **kwargs)
        origin_result = self.pipeline(*parsed_args, **parsed_kwargs)
        final_result = self._parse_output(origin_result, remote=False)
        return final_result
