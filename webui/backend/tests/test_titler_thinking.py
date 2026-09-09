"""The titler builds its own request instead of going through the SDK, so the
compatibility lessons the main path learned have to be repeated here — and were
not. Three separate live failures came out of that (probed 2026-08-18)."""
import httpx
import pytest

from app.backends.ms_agent import titler


def test_no_hardcoded_temperature_anywhere():
    """Kimi answers `400 invalid temperature: only 1 is allowed for this model`,
    and the allowed value changes with thinking mode (`only 0.6` there), so no
    fixed number works. Titles do not need the knob; the main path already
    stopped sending it unless the user asks."""
    source = (titler.__file__).replace(".pyc", ".py")
    with open(source, encoding="utf-8") as fh:
        assert "temperature" not in fh.read()


@pytest.mark.parametrize("base_url,expected", [
    ("https://dashscope.aliyuncs.com/compatible-mode/v1",
     {"enable_thinking": False}),
    ("https://api-inference.modelscope.cn/v1", {"enable_thinking": False}),
    ("https://open.bigmodel.cn/api/paas/v4",
     {"thinking": {"type": "disabled"}}),
    ("https://api.moonshot.cn/v1", {"reasoning_effort": "none"}),
    ("https://openrouter.ai/api/v1", {"reasoning": {"enabled": False}}),
    ("https://api.deepseek.com", {"thinking": {"type": "disabled"}}),
])
def test_thinking_is_switched_off_in_each_endpoints_own_dialect(base_url,
                                                                expected):
    """It used to send the Qwen spelling everywhere, which every other vendor
    silently ignores — so the title call kept paying for reasoning it thought
    it had disabled. `extra_body` is flattened because this is a raw JSON body,
    not an OpenAI-client call."""
    assert titler._thinking_off(base_url) == expected


def test_a_model_that_cannot_stop_thinking_is_detected():
    """OpenRouter's Grok family: asking for no reasoning is a 400. The titler
    has to notice and stop asking, rather than retry the same request."""
    body = ('{"error":{"message":"Reasoning is mandatory for this endpoint '
            'and cannot be disabled.","code":400}}')
    resp = httpx.Response(400, text=body,
                          request=httpx.Request("POST", "https://x/v1"))
    assert titler._is_mandatory_thinking(resp) is True

    ok = httpx.Response(200, text="{}",
                        request=httpx.Request("POST", "https://x/v1"))
    assert titler._is_mandatory_thinking(ok) is False
    other = httpx.Response(400, text='{"error":{"message":"bad model"}}',
                           request=httpx.Request("POST", "https://x/v1"))
    assert titler._is_mandatory_thinking(other) is False


def test_dropping_thinking_leaves_the_rest_of_the_payload_alone():
    payload = {"model": "m", "messages": [], "max_tokens": 600,
               "reasoning": {"enabled": False}, "enable_thinking": False}
    assert titler._without_thinking(payload) == {
        "model": "m", "messages": [], "max_tokens": 600}


def test_the_budget_is_not_tight_enough_to_be_eaten_by_reasoning():
    """grok-4.5 spends ~70 reasoning tokens before writing anything; at the old
    160-token cap it never got to the title and answered the user's question
    instead."""
    assert titler._MAX_TOKENS >= 600
    assert titler._MAX_TOKENS_THINKING > titler._MAX_TOKENS
