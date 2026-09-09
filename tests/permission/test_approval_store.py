import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from ms_agent.permission.approval import (
    ApprovalConflictError,
    ApprovalRequest,
    FileApprovalStore,
    MemoryApprovalStore,
)
from ms_agent.permission.handler import (
    PermissionAction,
    PermissionResponse,
    WebPermissionHandler,
)


@pytest.mark.parametrize('store_factory', [
    lambda tmp_path: MemoryApprovalStore(),
    lambda tmp_path: FileApprovalStore(tmp_path / 'approvals.json'),
])
def test_store_cas_state_machine_and_one_time_token(tmp_path, store_factory):
    store = store_factory(tmp_path)
    request = ApprovalRequest.create(
        tool_name='web---fetch',
        tool_args={'url': 'https://example.com/a'},
        call_id='call-1',
    )
    store.create(request)

    saved = store.get(request.id)
    assert saved is not None
    assert saved.fingerprint == request.fingerprint
    assert saved.version == 1

    approved = store.transition(
        request.id, 'approved', expected_version=1,
        token=request.token, fingerprint=request.fingerprint)
    assert approved.version == 2
    with pytest.raises(ApprovalConflictError):
        store.transition(request.id, 'resume_queued', expected_version=1)
    with pytest.raises(ApprovalConflictError):
        store.transition(
            request.id, 'resume_queued', expected_version=2,
            token=request.token)

    queued = store.transition(
        request.id, 'resume_queued', expected_version=2)
    resumed = store.transition(
        request.id,
        'resumed',
        expected_version=queued.version,
        token=queued.continuation_token,
        fingerprint=request.fingerprint,
    )
    assert resumed.state == 'resumed'
    assert resumed.continuation_used


def test_invalid_transition_is_rejected():
    store = MemoryApprovalStore()
    request = ApprovalRequest.create('tool', {})
    store.create(request)
    with pytest.raises(ValueError):
        store.transition(request.id, 'resumed', expected_version=1)


@pytest.mark.parametrize('kwargs', [
    {},
    {'token': 'provided-later'},
    {'fingerprint': 'provided-later'},
])
def test_external_decision_requires_token_and_fingerprint(kwargs):
    store = MemoryApprovalStore()
    request = ApprovalRequest.create('tool', {})
    store.create(request)
    supplied = {
        key: request.token if key == 'token' else request.fingerprint
        for key in kwargs
    }

    with pytest.raises(ApprovalConflictError):
        store.transition(
            request.id, 'approved', expected_version=1, **supplied)


@pytest.mark.parametrize(
    'decision_state', ['approved', 'denied', 'expired', 'cancelled'])
def test_every_decision_outcome_can_queue_resume(decision_state):
    store = MemoryApprovalStore()
    request = ApprovalRequest.create('tool', {})
    store.create(request)
    decided = store.transition(
        request.id,
        decision_state,
        expected_version=1,
        token=request.token,
        fingerprint=request.fingerprint,
    )

    queued = store.transition(
        request.id, 'resume_queued', expected_version=decided.version)

    assert queued.state == 'resume_queued'


def test_file_store_survives_reload(tmp_path):
    path = tmp_path / 'approvals.json'
    first = FileApprovalStore(path)
    request = ApprovalRequest.create('tool', {'unicode': '例子'})
    first.create(request)

    second = FileApprovalStore(path)
    assert second.get(request.id) == first.get(request.id)
    assert second.list()[0].tool_args == {'unicode': '例子'}
    assert path.stat().st_mode & 0o777 == 0o600
    assert request.token not in path.read_text(encoding='utf-8')


def test_legacy_plaintext_approval_file_is_rewritten_on_open(tmp_path):
    path = tmp_path / 'approvals.json'
    request = ApprovalRequest.create('tool', {'authorization': 'secret'})
    path.write_text(
        json.dumps({
            'version': 1,
            'requests': [{
                **{
                    key: value
                    for key, value in {
                        'id': request.id,
                        'tool_name': request.tool_name,
                        'tool_args': request.tool_args,
                        'call_id': request.call_id,
                        'context': request.context,
                        'suggestions': list(request.suggestions),
                        'state': 'pending',
                        'version': 1,
                        'fingerprint': request.fingerprint,
                        'token': request.token,
                        'token_used': False,
                        'created_at': request.created_at,
                        'updated_at': request.updated_at,
                    }.items()
                }
            }],
        }),
        encoding='utf-8',
    )
    os.chmod(path, 0o644)

    store = FileApprovalStore(path)
    saved = store.get(request.id)
    assert saved is not None
    assert path.stat().st_mode & 0o777 == 0o600
    text = path.read_text(encoding='utf-8')
    assert request.token not in text
    approved = store.transition(
        request.id,
        'approved',
        expected_version=saved.version,
        token=request.token,
        fingerprint=request.fingerprint,
    )
    assert approved.state == 'approved'


def test_file_store_reloads_before_cas_write(tmp_path):
    path = tmp_path / 'approvals.json'
    first = FileApprovalStore(path)
    request = ApprovalRequest.create('tool', {})
    first.create(request)
    stale = FileApprovalStore(path)

    first.transition(
        request.id,
        'approved',
        expected_version=1,
        token=request.token,
        fingerprint=request.fingerprint,
    )
    with pytest.raises(ApprovalConflictError):
        stale.transition(
            request.id,
            'denied',
            expected_version=1,
            token=request.token,
            fingerprint=request.fingerprint,
        )


def test_web_handler_persists_before_emit_and_resolve_is_idempotent():
    async def run():
        store = MemoryApprovalStore()

        class Emitter:
            events = []

            def emit(self, event):
                persisted = store.get(event['request_id'])
                assert persisted is not None
                self.events.append(event)

        emitter = Emitter()
        handler = WebPermissionHandler(emitter, timeout=1, store=store)
        task = asyncio.create_task(handler.ask('tool', {'x': 1}, 'context'))
        await asyncio.sleep(0)
        event = emitter.events[0]
        response = PermissionResponse(
            PermissionAction.ALLOW_ALWAYS,
            pattern='tool:*',
            scope='global',
        )

        assert handler.resolve(
            event['request_id'],
            response,
            token=event['approval_token'],
            fingerprint=event['fingerprint'],
        )
        assert handler.resolve(
            event['request_id'],
            response,
            token=event['approval_token'],
            fingerprint=event['fingerprint'],
        )
        assert not handler.resolve(
            event['request_id'],
            PermissionResponse(PermissionAction.DENY),
            token=event['approval_token'],
            fingerprint=event['fingerprint'],
        )
        assert (await task).action == PermissionAction.ALLOW_ALWAYS
        persisted = store.get(event['request_id'])
        assert persisted.state == 'approved'
        assert persisted.decision == 'allow_always'
        assert persisted.pattern == 'tool:*'
        assert persisted.scope == 'global'

    asyncio.run(run())


def test_request_persists_redacted_args_but_fingerprints_original():
    request = ApprovalRequest.create(
        'web---fetch',
        {'url': 'https://example.com', 'authorization': 'Bearer secret'},
    )

    assert request.tool_args['authorization'] == '[REDACTED]'
    assert request.fingerprint != ApprovalRequest.create(
        'web---fetch',
        {'url': 'https://example.com', 'authorization': 'different'},
    ).fingerprint


def test_resolve_without_live_waiter_queues_durable_resume(tmp_path):
    store = FileApprovalStore(tmp_path / 'approvals.json')
    request = ApprovalRequest.create('tool', {'x': 1})
    store.create(request)

    class Emitter:
        def emit(self, event):
            pass

    restarted = WebPermissionHandler(Emitter(), store=store)
    assert restarted.resolve(
        request.id,
        PermissionResponse(PermissionAction.ALLOW_ONCE),
        token=request.token,
        fingerprint=request.fingerprint,
    )

    persisted = store.get(request.id)
    assert persisted is not None
    assert persisted.state == 'resume_queued'


def test_expired_pending_approval_cannot_be_approved():
    store = MemoryApprovalStore()
    request = ApprovalRequest.create(
        'tool',
        {},
        expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)
                    ).isoformat(),
    )
    store.create(request)
    with pytest.raises(ApprovalConflictError):
        store.transition(
            request.id,
            'approved',
            expected_version=1,
            token=request.token,
            fingerprint=request.fingerprint,
        )
    assert store.get(request.id).state == 'expired'


def test_malformed_record_does_not_drop_the_store(tmp_path):
    path = tmp_path / 'approvals.json'
    store = FileApprovalStore(path)
    request = ApprovalRequest.create('tool', {'keep': True})
    store.create(request)
    payload = json.loads(path.read_text(encoding='utf-8'))
    payload['requests'].append({'not': 'an-approval'})
    payload['requests'].append({
        **payload['requests'][0],
        'id': 'compat-1',
        'unknown_future_field': True,
    })
    path.write_text(json.dumps(payload), encoding='utf-8')

    reloaded = FileApprovalStore(path)
    ids = {item.id for item in reloaded.list()}
    assert request.id in ids
    assert 'compat-1' in ids


def test_continuation_token_can_be_reissued_after_file_reload(tmp_path):
    path = tmp_path / 'approvals.json'
    store = FileApprovalStore(path)
    request = ApprovalRequest.create('tool', {'x': 1})
    store.create(request)
    approved = store.transition(
        request.id,
        'approved',
        expected_version=1,
        token=request.token,
        fingerprint=request.fingerprint,
        decision='allow_once',
    )
    queued = store.transition(
        request.id,
        'resume_queued',
        expected_version=approved.version,
    )
    assert queued.continuation_token

    restarted = FileApprovalStore(path)
    lost = restarted.get(request.id)
    assert lost is not None
    assert lost.continuation_token == ''
    assert lost.continuation_token_hash

    reissued = restarted.reissue_continuation(
        request.id,
        fingerprint=request.fingerprint,
        expected_version=lost.version,
    )
    assert reissued.state == 'resume_queued'
    assert reissued.continuation_token
    assert reissued.continuation_token != queued.continuation_token

    with pytest.raises(ApprovalConflictError):
        restarted.transition(
            request.id,
            'resumed',
            expected_version=reissued.version,
            token=queued.continuation_token,
            fingerprint=request.fingerprint,
        )

    resumed = restarted.transition(
        request.id,
        'resumed',
        expected_version=reissued.version,
        token=reissued.continuation_token,
        fingerprint=request.fingerprint,
    )
    assert resumed.state == 'resumed'
    assert resumed.continuation_used


def test_timeout_after_decision_queues_resume():
    async def run():
        store = MemoryApprovalStore()

        class Emitter:
            def emit(self, event):
                pass

        handler = WebPermissionHandler(Emitter(), timeout=0.01, store=store)
        task = asyncio.create_task(handler.ask('tool', {}, ''))
        await asyncio.sleep(0)
        request = store.list()[0]
        store.transition(
            request.id,
            'approved',
            expected_version=request.version,
            token=request.token,
            fingerprint=request.fingerprint,
            decision='allow_once',
        )
        response = await task
        persisted = store.get(request.id)
        assert response.action == PermissionAction.DENY
        assert persisted.state == 'resume_queued'

    asyncio.run(run())

