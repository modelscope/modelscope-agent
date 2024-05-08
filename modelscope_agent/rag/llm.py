# yapf: disable
from typing import Any, Callable, Dict, Optional, Sequence

import dashscope
from llama_index.core.base.llms.types import (ChatMessage, ChatResponse,
                                              ChatResponseAsyncGen,
                                              ChatResponseGen,
                                              CompletionResponse,
                                              CompletionResponseAsyncGen,
                                              CompletionResponseGen,
                                              LLMMetadata, MessageRole)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import (llm_chat_callback,
                                             llm_completion_callback)
from llama_index.core.llms.llm import LLM
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from modelscope_agent.llm.dashscope import DashScopeLLM


class MSAgentLLM(LLM):
    """dashscope LLM.

    Examples:
        `pip install llama-index-llms-cohere`

        ```python
        from llama_index.llms.cohere import Cohere

        llm = Cohere(model="command", api_key=api_key)
        resp = llm.complete("Paul Graham is ")
        print(resp)
        ```
    """

    model: str = Field(description='The dashscope model to use.')
    temperature: float = Field(
        description='The temperature to use for sampling.')
    max_retries: int = Field(
        default=10, description='The maximum number of API retries.')
    max_tokens: int = Field(
        description='The maximum number of tokens to generate.')
    # llm: DashScopeLLM = Field(description="The dashscope model to use.")
    _llm: Any = PrivateAttr()

    def __init__(
        self,
        llm: DashScopeLLM,
        model: str = 'qwen_max',
        temperature: float = 0.5,
        max_tokens: int = 2000,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]],
                                              str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.
        DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        # additional_kwargs = additional_kwargs or {}
        callback_manager = callback_manager or CallbackManager([])
        self._llm = llm

        super().__init__(
            llm=llm,
            temperature=temperature,
            additional_kwargs=additional_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            model=model,
            callback_manager=callback_manager,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return 'MS_Agent_LLM'

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            # num_output=self.max_tokens,
            is_chat_model=True,
            # model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            'model': self.model,
            'temperature': self.temperature,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            **self._model_kwargs,
            **kwargs,
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage],
             **kwargs: Any) -> ChatResponse:
        messages = [{
            'role': msg.role,
            'content': msg.content
        } for msg in messages]
        response = self._llm._chat_no_stream(messages, **kwargs)

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response),
            raw={'text': response},
        )

    @llm_completion_callback()
    def complete(self,
                 prompt: str,
                 formatted: bool = False,
                 **kwargs: Any) -> CompletionResponse:
        pass
        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = completion_with_retry(
            client=self._client,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.generations[0].text,
            raw=response.__dict__,
        )
    """

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage],
                    **kwargs: Any) -> ChatResponseGen:
        messages = [{
            'role': msg.role,
            'content': msg.content
        } for msg in messages]
        response = self._llm._chat_stream(messages, **kwargs)

        def gen() -> ChatResponseGen:
            content = ''
            role = MessageRole.ASSISTANT
            for r in response:
                content += r
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=r,
                    raw=r,
                )

        print(f'response: {response}')
        return gen()

    @llm_completion_callback()
    def stream_complete(self,
                        prompt: str,
                        formatted: bool = False,
                        **kwargs: Any) -> CompletionResponseGen:
        pass
        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True

        response = completion_with_retry(
            client=self._client,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        def gen() -> CompletionResponseGen:
            content = ""
            for r in response:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r._asdict()
                )

        return gen()
        """

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage],
                    **kwargs: Any) -> ChatResponse:
        pass
        """
        history = messages_to_cohere_history(messages[:-1])
        prompt = messages[-1].content
        all_kwargs = self._get_all_kwargs(**kwargs)
        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=True,
            message=prompt,
            chat_history=history,
            **all_kwargs,
        )

        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=response.text),
            raw=response.__dict__,
        )
        """

    @llm_completion_callback()
    async def acomplete(self,
                        prompt: str,
                        formatted: bool = False,
                        **kwargs: Any) -> CompletionResponse:
        pass
        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        if "stream" in all_kwargs:
            warnings.warn(
                "Parameter `stream` is not supported by the `chat` method."
                "Use the `stream_chat` method instead"
            )

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        return CompletionResponse(
            text=response.generations[0].text,
            raw=response.__dict__,
        )
        """

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage],
                           **kwargs: Any) -> ChatResponseAsyncGen:
        pass
        """
        history = messages_to_cohere_history(messages[:-1])
        prompt = messages[-1].content
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True
        if all_kwargs["model"] not in CHAT_MODELS:
            raise ValueError(f"{all_kwargs['model']} not supported for chat")
        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=True,
            message=prompt,
            chat_history=history,
            **all_kwargs,
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            role = MessageRole.ASSISTANT
            async for r in response:
                if "text" in r.__dict__:
                    content_delta = r.text
                else:
                    content_delta = ""
                content += content_delta
                yield ChatResponse(
                    message=ChatMessage(role=role, content=content),
                    delta=content_delta,
                    raw=r.__dict__,
                )

        return gen()
        """

    @llm_completion_callback()
    async def astream_complete(self,
                               prompt: str,
                               formatted: bool = False,
                               **kwargs: Any) -> CompletionResponseAsyncGen:
        pass
        """
        all_kwargs = self._get_all_kwargs(**kwargs)
        all_kwargs["stream"] = True

        response = await acompletion_with_retry(
            aclient=self._aclient,
            max_retries=self.max_retries,
            chat=False,
            prompt=prompt,
            **all_kwargs,
        )

        async def gen() -> CompletionResponseAsyncGen:
            content = ""
            async for r in response:
                content_delta = r.text
                content += content_delta
                yield CompletionResponse(
                    text=content, delta=content_delta, raw=r._asdict()
                )

        return gen()
    """
