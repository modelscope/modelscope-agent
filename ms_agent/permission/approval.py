"""Durable approval requests with CAS transitions and resume tracking."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from .provider import sanitize_sensitive_text, sanitize_tool_args

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

ApprovalState = Literal[
    'pending',
    'approved',
    'denied',
    'expired',
    'cancelled',
    'resume_queued',
    'resumed',
    'resume_failed',
]

_TRANSITIONS: dict[str, frozenset[str]] = {
    'pending': frozenset({'approved', 'denied', 'expired', 'cancelled'}),
    'approved': frozenset({'resume_queued'}),
    'resume_queued': frozenset({'resumed', 'resume_failed'}),
    'denied': frozenset({'resume_queued'}),
    'expired': frozenset({'resume_queued'}),
    'cancelled': frozenset({'resume_queued'}),
    'resumed': frozenset(),
    'resume_failed': frozenset(),
}


class ApprovalConflictError(RuntimeError):
    """A CAS version, token, fingerprint, or duplicate-create conflict."""


def approval_fingerprint(
    tool_name: str,
    tool_args: dict[str, Any],
    call_id: str = '',
) -> str:
    canonical = json.dumps(
        {
            'tool_name': tool_name,
            'tool_args': tool_args,
            'call_id': call_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _is_expired(expires_at: str) -> bool:
    if not expires_at:
        return False
    try:
        expiry = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expiry


@dataclass(frozen=True)
class ApprovalRequest:
    id: str
    tool_name: str
    tool_args: dict[str, Any]
    call_id: str = ''
    context: str = ''
    suggestions: tuple[str, ...] = ()
    state: ApprovalState = 'pending'
    version: int = 1
    fingerprint: str = ''
    token: str = ''
    token_hash: str = ''
    token_used: bool = False
    created_at: str = ''
    updated_at: str = ''
    feedback: str = ''
    decision: str = ''
    pattern: str = ''
    scope: Literal['project', 'global'] = 'project'
    decided_by: str = ''
    project_id: str = ''
    session_id: str = ''
    dispatch_id: str = ''
    runtime_id: str = ''
    requester: str = ''
    decision_provider: str = ''
    expires_at: str = ''
    decision_args: dict[str, Any] | None = None
    decision_fingerprint: str = ''
    resume_strategy: str = 'reattach_or_continue'
    resume_error: str = ''
    continuation_token: str = ''
    continuation_token_hash: str = ''
    continuation_used: bool = False

    @property
    def request_id(self) -> str:
        return self.id

    @property
    def status(self) -> ApprovalState:
        """Compatibility alias for APIs that call the state ``status``."""
        return self.state

    @classmethod
    def create(
        cls,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        call_id: str = '',
        context: str = '',
        suggestions: list[str] | tuple[str, ...] = (),
        request_id: str | None = None,
        project_id: str = '',
        session_id: str = '',
        dispatch_id: str = '',
        runtime_id: str = '',
        requester: str = '',
        decision_provider: str = '',
        expires_at: str = '',
        resume_strategy: str = 'reattach_or_continue',
    ) -> 'ApprovalRequest':
        now = _now()
        token = uuid4().hex
        return cls(
            id=request_id or uuid4().hex,
            tool_name=tool_name,
            tool_args=sanitize_tool_args(tool_args),
            call_id=call_id,
            context=sanitize_sensitive_text(context),
            suggestions=tuple(sanitize_tool_args(list(suggestions))),
            fingerprint=approval_fingerprint(tool_name, tool_args, call_id),
            token=token,
            token_hash=_token_digest(token),
            created_at=now,
            updated_at=now,
            project_id=project_id,
            session_id=session_id,
            dispatch_id=dispatch_id,
            runtime_id=runtime_id,
            requester=requester,
            decision_provider=decision_provider,
            expires_at=expires_at,
            resume_strategy=resume_strategy,
        )


class ApprovalStore(Protocol):

    def create(self, request: ApprovalRequest) -> ApprovalRequest: ...

    def get(self, request_id: str) -> ApprovalRequest | None: ...

    def list(self, state: ApprovalState | None = None
             ) -> list[ApprovalRequest]: ...

    def update(
        self,
        request: ApprovalRequest,
        *,
        expected_version: int,
    ) -> ApprovalRequest: ...

    def transition(
        self,
        request_id: str,
        state: ApprovalState,
        *,
        expected_version: int,
        token: str | None = None,
        fingerprint: str | None = None,
        feedback: str = '',
        decision: str = '',
        pattern: str = '',
        scope: Literal['project', 'global'] | None = None,
        decided_by: str = '',
        decision_args: dict[str, Any] | None = None,
    ) -> ApprovalRequest: ...

    def compare_and_set(
        self,
        request_id: str,
        *,
        expected_version: int,
        state: ApprovalState,
        token: str | None = None,
        fingerprint: str | None = None,
        feedback: str = '',
        decision: str = '',
        pattern: str = '',
        scope: Literal['project', 'global'] | None = None,
        decided_by: str = '',
        decision_args: dict[str, Any] | None = None,
    ) -> ApprovalRequest: ...

    def reissue_continuation(
        self,
        request_id: str,
        *,
        fingerprint: str,
        expected_version: int | None = None,
    ) -> ApprovalRequest: ...


class MemoryApprovalStore:
    """Thread-safe in-memory ApprovalStore."""

    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = threading.RLock()

    def create(self, request: ApprovalRequest) -> ApprovalRequest:
        with self._lock:
            if request.id in self._requests:
                raise ApprovalConflictError(
                    f'Approval request already exists: {request.id}')
            stored = deepcopy(request)
            self._requests[request.id] = stored
            self._after_change()
            return deepcopy(stored)

    def get(self, request_id: str) -> ApprovalRequest | None:
        with self._lock:
            request = self._requests.get(request_id)
            return deepcopy(request) if request is not None else None

    def list(
        self,
        state: ApprovalState | None = None,
    ) -> list[ApprovalRequest]:
        with self._lock:
            values = sorted(
                self._requests.values(), key=lambda item: item.created_at)
            return deepcopy([
                item for item in values
                if state is None or item.state == state
            ])

    def update(
        self,
        request: ApprovalRequest,
        *,
        expected_version: int,
    ) -> ApprovalRequest:
        with self._lock:
            current = self._require(request.id)
            self._check_version(current, expected_version)
            immutable = (
                'id', 'tool_name', 'tool_args', 'call_id', 'state',
                'fingerprint', 'token', 'token_hash', 'token_used',
                'created_at', 'project_id', 'session_id', 'dispatch_id',
                'runtime_id', 'requester', 'decision_provider',
                'decision', 'pattern', 'scope', 'decided_by',
                'decision_args', 'decision_fingerprint',
                'continuation_token', 'continuation_token_hash',
                'continuation_used',
            )
            if any(
                    getattr(request, field) != getattr(current, field)
                    for field in immutable):
                raise ValueError(
                    'Approval identity and state must be changed via '
                    'transition()')
            updated = replace(
                request,
                version=current.version + 1,
                updated_at=_now(),
            )
            self._requests[request.id] = deepcopy(updated)
            self._after_change()
            return deepcopy(updated)

    def reissue_continuation(
        self,
        request_id: str,
        *,
        fingerprint: str,
        expected_version: int | None = None,
    ) -> ApprovalRequest:
        """Rotate a lost continuation token without executing the tool call."""
        with self._lock:
            current = self._require(request_id)
            if expected_version is not None:
                self._check_version(current, expected_version)
            if current.state not in ('resume_queued', 'resume_failed'):
                raise ApprovalConflictError(
                    'Continuation can only be reissued from '
                    'resume_queued or resume_failed')
            if current.continuation_used:
                raise ApprovalConflictError(
                    'Continuation token was already used')
            expected_fingerprint = (
                current.decision_fingerprint or current.fingerprint)
            if not hmac.compare_digest(fingerprint, expected_fingerprint):
                raise ApprovalConflictError(
                    'Continuation fingerprint mismatch')
            token = uuid4().hex
            updated = replace(
                current,
                state='resume_queued',
                version=current.version + 1,
                updated_at=_now(),
                continuation_token=token,
                continuation_token_hash=_token_digest(token),
                continuation_used=False,
            )
            self._requests[request_id] = deepcopy(updated)
            self._after_change()
            return deepcopy(updated)

    def compare_and_set(
        self,
        request_id: str,
        *,
        expected_version: int,
        state: ApprovalState,
        token: str | None = None,
        fingerprint: str | None = None,
        feedback: str = '',
        decision: str = '',
        pattern: str = '',
        scope: Literal['project', 'global'] | None = None,
        decided_by: str = '',
        decision_args: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Explicit CAS alias for transition-oriented callers."""
        return self.transition(
            request_id,
            state,
            expected_version=expected_version,
            token=token,
            fingerprint=fingerprint,
            feedback=feedback,
            decision=decision,
            pattern=pattern,
            scope=scope,
            decided_by=decided_by,
            decision_args=decision_args,
        )

    def transition(
        self,
        request_id: str,
        state: ApprovalState,
        *,
        expected_version: int,
        token: str | None = None,
        fingerprint: str | None = None,
        feedback: str = '',
        decision: str = '',
        pattern: str = '',
        scope: Literal['project', 'global'] | None = None,
        decided_by: str = '',
        decision_args: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        with self._lock:
            current = self._require(request_id)
            self._check_version(current, expected_version)
            if state not in _TRANSITIONS.get(current.state, frozenset()):
                raise ValueError(
                    f'Invalid approval transition: {current.state} -> {state}')
            if fingerprint is not None and not hmac.compare_digest(
                    fingerprint, current.fingerprint):
                raise ApprovalConflictError('Approval fingerprint mismatch')

            consumes_token = current.state == 'pending'
            if consumes_token and _is_expired(current.expires_at):
                updated = replace(
                    current,
                    state='expired',
                    version=current.version + 1,
                    updated_at=_now(),
                    feedback=feedback or current.feedback or 'Approval expired',
                )
                self._requests[request_id] = deepcopy(updated)
                self._after_change()
                raise ApprovalConflictError('Approval request has expired')
            if consumes_token:
                if current.token_used:
                    raise ApprovalConflictError('Approval token was already used')
                requires_token = state in ('approved', 'denied')
                if requires_token and token is None:
                    raise ApprovalConflictError('Approval token is required')
                if requires_token and fingerprint is None:
                    raise ApprovalConflictError(
                        'Approval fingerprint is required')
                expected_hash = (
                    current.token_hash or _token_digest(current.token))
                if token is not None and not hmac.compare_digest(
                        _token_digest(token), expected_hash):
                    raise ApprovalConflictError('Invalid approval token')
            elif current.state == 'resume_queued' and state == 'resumed':
                if token is None or fingerprint is None:
                    raise ApprovalConflictError(
                        'Continuation token and fingerprint are required')
                expected_hash = current.continuation_token_hash
                if (
                    current.continuation_used
                    or not expected_hash
                    or not hmac.compare_digest(
                        _token_digest(token), expected_hash)
                ):
                    raise ApprovalConflictError(
                        'Invalid or consumed continuation token')
                expected_fingerprint = (
                    current.decision_fingerprint or current.fingerprint)
                if not hmac.compare_digest(
                        fingerprint, expected_fingerprint):
                    raise ApprovalConflictError(
                        'Continuation fingerprint mismatch')
            elif token is not None:
                raise ApprovalConflictError(
                    'A token is not valid for this transition')

            continuation_token = current.continuation_token
            continuation_token_hash = current.continuation_token_hash
            continuation_used = current.continuation_used
            if state == 'resume_queued' and not continuation_token_hash:
                continuation_token = uuid4().hex
                continuation_token_hash = _token_digest(continuation_token)
            if current.state == 'resume_queued' and state == 'resumed':
                continuation_used = True

            updated = replace(
                current,
                state=state,
                version=current.version + 1,
                token_used=current.token_used or consumes_token,
                updated_at=_now(),
                feedback=feedback or current.feedback,
                decision=decision or current.decision,
                pattern=pattern or current.pattern,
                scope=scope or current.scope,
                decided_by=decided_by or current.decided_by,
                decision_args=(
                    sanitize_tool_args(decision_args)
                    if decision_args is not None else current.decision_args),
                decision_fingerprint=(
                    approval_fingerprint(
                        current.tool_name,
                        decision_args,
                        current.call_id,
                    )
                    if decision_args is not None
                    else current.decision_fingerprint),
                continuation_token=continuation_token,
                continuation_token_hash=continuation_token_hash,
                continuation_used=continuation_used,
            )
            self._requests[request_id] = deepcopy(updated)
            self._after_change()
            return deepcopy(updated)

    def _require(self, request_id: str) -> ApprovalRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(request_id)
        return request

    @staticmethod
    def _check_version(
        request: ApprovalRequest,
        expected_version: int,
    ) -> None:
        if request.version != expected_version:
            raise ApprovalConflictError(
                f'Approval version conflict: expected {expected_version}, '
                f'found {request.version}')

    def _after_change(self) -> None:
        pass


class FileApprovalStore(MemoryApprovalStore):
    """JSON-file ApprovalStore using atomic replacement on every mutation."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock_path = self._path.with_suffix(f'{self._path.suffix}.lock')
        super().__init__()
        self._load()

    def _load(self) -> None:
        self._requests = {}
        if not self._path.exists():
            return
        known = {item.name for item in fields(ApprovalRequest)}
        dirty = (self._path.stat().st_mode & 0o777) != 0o600
        try:
            raw = json.loads(self._path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            return
        entries = raw.get('requests', raw) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return
        for item in entries:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item['suggestions'] = tuple(item.get('suggestions', ()))
            legacy_token = str(item.get('token') or '')
            if legacy_token:
                dirty = True
            item.setdefault(
                'token_hash',
                _token_digest(legacy_token) if legacy_token else '',
            )
            item['token'] = ''
            legacy_continuation = str(item.get('continuation_token') or '')
            if legacy_continuation:
                dirty = True
            item.setdefault(
                'continuation_token_hash',
                _token_digest(legacy_continuation)
                if legacy_continuation else '',
            )
            item['continuation_token'] = ''
            filtered = {key: value for key, value in item.items()
                        if key in known}
            try:
                request = ApprovalRequest(**filtered)
            except (TypeError, ValueError, KeyError):
                continue
            self._requests[request.id] = request
        if dirty:
            self._after_change()

    @contextmanager
    def _file_transaction(self):
        with self._lock:
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(
                self._lock_path,
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            os.chmod(self._lock_path, 0o600)
            with os.fdopen(lock_fd, 'a+', encoding='utf-8') as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    lock_file.seek(0, os.SEEK_END)
                    if lock_file.tell() == 0:
                        lock_file.write('\0')
                        lock_file.flush()
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    self._load()
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    elif msvcrt is not None:  # pragma: no cover - Windows
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)

    def create(self, request: ApprovalRequest) -> ApprovalRequest:
        with self._file_transaction():
            return super().create(request)

    def get(self, request_id: str) -> ApprovalRequest | None:
        with self._file_transaction():
            return super().get(request_id)

    def list(
        self,
        state: ApprovalState | None = None,
    ) -> list[ApprovalRequest]:
        with self._file_transaction():
            return super().list(state)

    def update(
        self,
        request: ApprovalRequest,
        *,
        expected_version: int,
    ) -> ApprovalRequest:
        with self._file_transaction():
            return super().update(
                request, expected_version=expected_version)

    def reissue_continuation(
        self,
        request_id: str,
        *,
        fingerprint: str,
        expected_version: int | None = None,
    ) -> ApprovalRequest:
        with self._file_transaction():
            return super().reissue_continuation(
                request_id,
                fingerprint=fingerprint,
                expected_version=expected_version,
            )

    def transition(
        self,
        request_id: str,
        state: ApprovalState,
        *,
        expected_version: int,
        token: str | None = None,
        fingerprint: str | None = None,
        feedback: str = '',
        decision: str = '',
        pattern: str = '',
        scope: Literal['project', 'global'] | None = None,
        decided_by: str = '',
        decision_args: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        with self._file_transaction():
            return super().transition(
                request_id,
                state,
                expected_version=expected_version,
                token=token,
                fingerprint=fingerprint,
                feedback=feedback,
                decision=decision,
                pattern=pattern,
                scope=scope,
                decided_by=decided_by,
                decision_args=decision_args,
            )

    def _after_change(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        requests = sorted(
            self._requests.values(), key=lambda item: item.created_at)
        persisted = []
        for item in requests:
            raw = asdict(item)
            raw['token'] = ''
            raw['continuation_token'] = ''
            persisted.append(raw)
        payload = {
            'version': 1,
            'requests': persisted,
        }
        temp = self._path.with_name(
            f'.{self._path.name}.{uuid4().hex}.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, self._path)
        os.chmod(self._path, 0o600)

