import asyncio
import time

import pytest

from ms_agent.llm.utils import Message
from ms_agent.permission.config import PermissionConfig
from ms_agent.permission.enforcer import PermissionDecision, PermissionEnforcer
from ms_agent.permission.handler import PermissionAction, PermissionResponse
from ms_agent.permission.memory import PermissionMemory
from ms_agent.permission.provider import (
    AgentDecisionProvider,
    LlmDecisionProvider,
    ProviderDecision,
    request_provider_decision,
)


class _Provider:
    def __init__(self, result=None, error=None, delay=0):
        self.result = result
        self.error = error
        self.delay = delay
        self.calls = 0

    async def decide(self, tool_name, tool_args, context, suggestions):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class _Handler:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def ask(self, tool_name, tool_args, context, suggestions=None,
                  call_id=''):
        self.calls += 1
        return self.response


def _config(**extra):
    return PermissionConfig.from_dict({
        'mode': 'delegate',
        'decision_provider': 'llm',
        **extra,
    })


@pytest.mark.parametrize('timeout', ['nan', 'inf', '-inf', 0, -1])
def test_provider_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError):
        _config(provider_timeout=timeout)


@pytest.mark.asyncio
async def test_provider_allow_once_allows_without_human(tmp_path):
    provider = _Provider(ProviderDecision('allow_once', 'low risk'))
    enforcer = PermissionEnforcer(
        _config(),
        provider=provider,
        memory=PermissionMemory(project_path=tmp_path),
    )

    result = await enforcer.check('custom---tool', {'value': 1})

    assert result.action == 'allow'
    assert provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_result', [
    {'action': 'allow_always'},
    {'action': 'unexpected'},
    'not-json',
])
async def test_invalid_or_persistent_provider_result_denies_without_human(
        tmp_path, bad_result):
    enforcer = PermissionEnforcer(
        _config(),
        provider=_Provider(bad_result),
        memory=PermissionMemory(project_path=tmp_path),
    )

    result = await enforcer.check('custom---tool', {})

    assert result.action == 'deny'
    assert 'uncertain' in result.reason.lower()


@pytest.mark.asyncio
async def test_provider_exception_and_timeout_become_uncertain_then_human(
        tmp_path):
    handler = _Handler(PermissionResponse(PermissionAction.ALLOW_ONCE))
    for provider in (
            _Provider(error=RuntimeError('boom')),
            _Provider(ProviderDecision('allow_once'), delay=.05)):
        enforcer = PermissionEnforcer(
            _config(human_approval_available=True, provider_timeout=0.001),
            handler=handler,
            provider=provider,
            memory=PermissionMemory(project_path=tmp_path),
        )
        assert (await enforcer.check('custom---tool', {})).action == 'allow'

    assert handler.calls == 2


@pytest.mark.asyncio
async def test_human_flag_without_real_handler_does_not_auto_allow(tmp_path):
    enforcer = PermissionEnforcer(
        _config(human_approval_available=True),
        provider=_Provider(ProviderDecision('uncertain')),
        memory=PermissionMemory(project_path=tmp_path),
    )

    assert (await enforcer.check('custom---tool', {})).action == 'deny'


@pytest.mark.asyncio
async def test_delegate_handler_without_human_flag_does_not_escalate(tmp_path):
    handler = _Handler(PermissionResponse(PermissionAction.ALLOW_ONCE))
    enforcer = PermissionEnforcer(
        _config(human_approval_available=False),
        handler=handler,
        provider=_Provider(ProviderDecision('uncertain')),
        memory=PermissionMemory(project_path=tmp_path),
    )

    result = await enforcer.check('custom---tool', {})

    assert result.action == 'deny'
    assert handler.calls == 0
    assert 'uncertain' in result.reason.lower()


@pytest.mark.asyncio
async def test_interactive_mode_without_real_handler_fails_closed(tmp_path):
    enforcer = PermissionEnforcer(
        PermissionConfig(mode='interactive'),
        memory=PermissionMemory(project_path=tmp_path),
    )

    result = await enforcer.check('custom---tool', {})

    assert result.action == 'deny'
    assert 'human' in result.reason.lower()


@pytest.mark.asyncio
async def test_synchronous_provider_is_also_bounded_by_timeout(tmp_path):
    class SlowSyncProvider:
        def decide(self, tool_name, tool_args, context, suggestions):
            time.sleep(.05)
            return ProviderDecision('allow_once')

    enforcer = PermissionEnforcer(
        _config(provider_timeout=.001),
        provider=SlowSyncProvider(),
        memory=PermissionMemory(project_path=tmp_path),
    )

    result = await enforcer.check('custom---tool', {})

    assert result.action == 'deny'
    assert 'timed out' in result.reason


@pytest.mark.asyncio
async def test_blacklist_and_force_safety_cannot_be_bypassed(tmp_path):
    provider = _Provider(ProviderDecision('allow_once'))
    handler = _Handler(PermissionResponse(PermissionAction.DENY))
    enforcer = PermissionEnforcer(
        _config(
            blacklist=['custom---danger'],
            human_approval_available=True,
        ),
        handler=handler,
        provider=provider,
        memory=PermissionMemory(project_path=tmp_path),
    )

    blocked = await enforcer.check('custom---danger', {})
    forced = await enforcer.check(
        'custom---safe',
        {},
        force_decision=PermissionDecision('ask', 'safety requires approval'),
    )

    assert blocked.action == 'deny'
    assert forced.action == 'deny'
    assert provider.calls == 0
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_allow_always_honors_global_scope(tmp_path):
    global_file = tmp_path / 'global' / 'permissions.json'
    memory = PermissionMemory(
        project_path=tmp_path / 'project', global_path=global_file)
    handler = _Handler(PermissionResponse(
        PermissionAction.ALLOW_ALWAYS,
        pattern='custom---tool',
        scope='global',
    ))
    enforcer = PermissionEnforcer(
        PermissionConfig(mode='interactive'),
        handler=handler,
        memory=memory,
    )

    assert (await enforcer.check('custom---tool', {})).action == 'allow'
    assert memory.list(scope='project') == []
    assert [entry.pattern for entry in memory.list(scope='global')] == [
        'custom---tool'
    ]


@pytest.mark.asyncio
async def test_llm_provider_uses_no_tools_and_redacts_sensitive_args():
    class Llm:
        calls = []

        def generate(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return Message(
                role='assistant',
                content='{"action":"allow_once","reason":"read-only"}',
            )

    llm = Llm()
    provider = LlmDecisionProvider(llm)

    decision = await provider.decide(
        'web---fetch',
        {
            'url': 'https://example.com',
            'authorization': 'Bearer secret',
        },
        'network request',
        ['domain:example.com'],
    )

    assert decision.action == 'allow_once'
    messages, kwargs = llm.calls[0]
    assert kwargs['tools'] == []
    assert 'Bearer secret' not in messages[-1].content
    assert '[REDACTED]' in messages[-1].content


@pytest.mark.asyncio
async def test_agent_provider_rejects_self_approval_and_malformed_output():
    calls = []

    async def approve(payload):
        calls.append(payload)
        return {'action': 'allow_once'}

    self_provider = AgentDecisionProvider(
        approve, approver_id='worker-1', requester_id='worker-1')
    assert (await self_provider.decide('tool', {}, '', [])).action == 'uncertain'
    assert calls == []

    provider = AgentDecisionProvider(
        approve, approver_id='approver-1', requester_id='worker-1')
    assert (await provider.decide('tool', {}, '', [])).action == 'allow_once'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_sync_agent_callback_does_not_block_provider_timeout(tmp_path):
    def blocking(_payload):
        time.sleep(.05)
        return {'action': 'allow_once'}

    provider = AgentDecisionProvider(
        blocking, approver_id='approver-1', requester_id='worker-1')
    enforcer = PermissionEnforcer(
        _config(provider_timeout=.001),
        provider=provider,
        memory=PermissionMemory(project_path=tmp_path),
    )
    started = time.monotonic()

    result = await enforcer.check('custom---tool', {})

    assert result.action == 'deny'
    assert time.monotonic() - started < .04


@pytest.mark.asyncio
async def test_sync_provider_timeout_uses_dedicated_executor(monkeypatch):
    seen = []
    original = asyncio.BaseEventLoop.run_in_executor

    def wrapped(self, executor, func, *args):
        seen.append(executor)
        return original(self, executor, func, *args)

    monkeypatch.setattr(asyncio.BaseEventLoop, 'run_in_executor', wrapped)

    class Provider:
        def decide(self, tool_name, tool_args, context, suggestions):
            return ProviderDecision('deny', 'blocked')

    result = await request_provider_decision(
        Provider(),
        tool_name='custom---tool',
        tool_args={},
        context='',
        suggestions=[],
        timeout=1,
    )

    assert result.action == 'deny'
    assert seen
    assert seen[0] is not None


@pytest.mark.asyncio
async def test_provider_redacts_embedded_credentials_and_suggestions():
    class Llm:
        prompt = ''

        def generate(self, messages, **kwargs):
            self.prompt = messages[-1].content
            return Message(
                role='assistant',
                content='{"action":"deny","reason":"credential"}',
            )

    llm = Llm()
    provider = LlmDecisionProvider(llm)
    await provider.decide(
        'code_executor---shell_executor',
        {
            'command': 'curl -H "Authorization: Bearer abc123" '
                       '"https://example.com?access_token=query-secret"',
            'headers': {'X-Api-Key': 'header-secret'},
        },
        'token=reason-secret',
        ['tool:https://user:pass@example.com?api_key=suggestion-secret'],
    )

    for secret in (
        'abc123', 'query-secret', 'header-secret', 'reason-secret',
        'pass', 'suggestion-secret',
    ):
        assert secret not in llm.prompt


@pytest.mark.asyncio
async def test_provider_redacts_access_keys_pats_and_basic_auth():
    class Llm:
        prompt = ''

        def generate(self, messages, **kwargs):
            self.prompt = messages[-1].content
            return Message(role='assistant', content='{"action":"deny"}')

    llm = Llm()
    provider = LlmDecisionProvider(llm)
    await provider.decide(
        'code_executor---shell_executor',
        {
            'AWS_ACCESS_KEY_ID': 'AKIAEXAMPLE',
            'GITHUB_PAT': 'ghp_example',
            'command': 'curl -H "Authorization: Basic dXNlcjpwYXNz" '
                       'https://example.com',
        },
        'quoted password="multi word secret"',
        [],
    )

    for secret in ('AKIAEXAMPLE', 'ghp_example', 'dXNlcjpwYXNz',
                   'multi word secret'):
        assert secret not in llm.prompt

