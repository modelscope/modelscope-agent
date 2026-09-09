"""PermissionHandler protocol and implementations.

Three implementations:
  - AutoPermissionHandler: always allow (fallback).
  - CLIPermissionHandler: interactive terminal menu.
  - WebPermissionHandler: Future-based async with event emitter.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Literal, Protocol

from .approval import (ApprovalConflictError, ApprovalRequest, ApprovalStore,
                       MemoryApprovalStore)
from .provider import sanitize_sensitive_text, sanitize_tool_args


class PermissionAction(str, Enum):
    ALLOW_ONCE = 'allow_once'
    ALLOW_SESSION = 'allow_session'
    ALLOW_ALWAYS = 'allow_always'
    DENY = 'deny'
    MODIFY = 'modify'


@dataclass(frozen=True)
class PermissionResponse:
    action: PermissionAction
    updated_args: dict[str, Any] | None = None
    pattern: str | None = None
    feedback: str | None = None
    scope: Literal['project', 'global'] = 'project'


class PermissionHandler(Protocol):
    """Confirmation UI for a tool call the policy can't decide on its own.

    Optional duck-typed attribute ``supports_concurrent_asks`` (default
    ``False`` when absent) declares whether several asks may be in flight at
    once. It is False for anything bound to the one terminal — N prompts
    fighting over a single stdin/menu deadlock — so ``PermissionEnforcer``
    serializes those. A handler that keys pending asks by id and renders them
    independently (``WebPermissionHandler``) sets it True, so a round's
    parallel tool calls all surface for decision at the same time instead of
    one-at-a-time behind whoever the user answers first.
    """

    async def ask(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str,
        suggestions: list[str] | None = None,
        call_id: str = '',
        workspace_root: str = '',
    ) -> PermissionResponse:
        ...


class AutoPermissionHandler:
    """Always allows — used as fallback or in auto mode."""

    # Never blocks on anything, so it has no reason to be serialized.
    supports_concurrent_asks = True

    async def ask(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str,
        suggestions: list[str] | None = None,
        call_id: str = '',
        workspace_root: str = '',
    ) -> PermissionResponse:
        return PermissionResponse(action=PermissionAction.ALLOW_ONCE)


def _args_preview(tool_args: dict[str, Any]) -> str:
    args_display = json.dumps(
        sanitize_tool_args(tool_args), ensure_ascii=False, indent=2)
    if len(args_display) > 500:
        args_display = args_display[:500] + '...'
    return args_display


class CLIPermissionHandler:
    """One-layer CLI permission prompt (scene options, single stdin read)."""

    async def ask(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str,
        suggestions: list[str] | None = None,
        call_id: str = '',
        workspace_root: str = '',
    ) -> PermissionResponse:
        from .ask_options import (build_ask_options, format_ask_menu,
                                  parse_ask_choice)
        options = build_ask_options(
            tool_name,
            tool_args,
            workspace_root=workspace_root or None,
            suggestions=suggestions,
        )
        menu = format_ask_menu(
            options,
            tool_name=tool_name,
            args_preview=_args_preview(tool_args),
            context=sanitize_sensitive_text(context) if context else '',
        )
        print(f'\n{menu}', file=sys.stderr)
        print('choice: ', end='', file=sys.stderr, flush=True)
        loop = asyncio.get_running_loop()
        try:
            raw = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            return PermissionResponse(action=PermissionAction.DENY)
        if raw == '':
            return PermissionResponse(action=PermissionAction.DENY)
        return parse_ask_choice(raw, options)


class EventEmitter(Protocol):
    """Protocol for pushing events to the frontend."""

    def emit(self, event: dict[str, Any]) -> None:
        ...


@dataclass
class _PendingAsk:
    """One card the user has not answered yet, and what it was about."""
    future: 'asyncio.Future[PermissionResponse]'
    loop: asyncio.AbstractEventLoop
    tool_name: str
    tool_args: dict
    forced: bool = False


class WebPermissionHandler:
    """Async handler that suspends on a Future until the frontend responds."""

    # Pending asks are keyed by request_id and each renders as its own card, so
    # a round's parallel tool calls can all wait for a decision simultaneously.
    # Serializing them instead would show one card at a time while the untouched
    # siblings sat there looking like they were already running.
    supports_concurrent_asks = True

    def __init__(
        self,
        event_emitter: EventEmitter,
        timeout: float | None = None,
        store: ApprovalStore | None = None,
    ) -> None:
        """``timeout=None`` waits indefinitely for an answer.

        That is the right default for a handler whose whole purpose is to ask
        a person something: expiring the question answers it on their behalf,
        with the one answer they cannot undo. The host sets a bound where one
        makes sense — a full-access session, where the human may not be at the
        keyboard at all.
        """
        self._pending: dict[str, _PendingAsk] = {}
        self._pending_lock = threading.RLock()
        self._event_emitter = event_emitter
        self._timeout = timeout
        self._store = store or MemoryApprovalStore()

    async def ask(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        context: str,
        suggestions: list[str] | None = None,
        call_id: str = '',
        workspace_root: str = '',
        forced: bool = False,
    ) -> PermissionResponse:
        from .ask_options import build_ask_options
        ask_options = build_ask_options(
            tool_name,
            tool_args,
            workspace_root=workspace_root or None,
            suggestions=suggestions,
        )
        expires_at = ''
        if self._timeout is not None:
            expires_at = (
                datetime.now(timezone.utc)
                + timedelta(seconds=self._timeout)
            ).isoformat()
        request = ApprovalRequest.create(
            tool_name,
            tool_args,
            call_id=call_id,
            context=context,
            suggestions=suggestions or [],
            expires_at=expires_at,
        )
        request_id = request.id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PermissionResponse] = loop.create_future()

        # Durability comes before visibility: consumers can immediately fetch
        # every request they observe from the emitted event.
        try:
            self._store.create(request)
            with self._pending_lock:
                self._pending[request_id] = _PendingAsk(
                    future=future,
                    loop=loop,
                    tool_name=tool_name,
                    tool_args=dict(tool_args or {}),
                    forced=forced,
                )
            try:
                self._event_emitter.emit({
                    'type': 'permission_request',
                    'request_id': request_id,
                    'call_id': call_id,
                    'tool_name': tool_name,
                    'tool_args': request.tool_args,
                    'context': sanitize_sensitive_text(context),
                    'suggestions': list(request.suggestions),
                    'options': [opt.key for opt in ask_options],
                    'ask_options': [
                        {
                            'key': opt.key,
                            'label': opt.label,
                            'action': opt.action.value,
                            'pattern': opt.pattern,
                            'editable': opt.editable,
                            'edit_value': opt.edit_value,
                        }
                        for opt in ask_options
                    ],
                    'approval_token': request.token,
                    'fingerprint': request.fingerprint,
                    'version': request.version,
                })
            except Exception as exc:
                self._cancel_pending(request_id, request.version)
                return PermissionResponse(
                    action=PermissionAction.DENY,
                    feedback=(
                        'Permission request could not be delivered: '
                        f'{type(exc).__name__}'),
                )
            if self._timeout is None:
                return await future
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._timeout)
        except asyncio.TimeoutError:
            current = self._store.get(request_id)
            if current is not None and current.state in ('approved', 'denied'):
                try:
                    self._store.transition(
                        request_id,
                        'resume_queued',
                        expected_version=current.version,
                    )
                except (ApprovalConflictError, ValueError):
                    pass
            elif current is not None and current.state == 'pending':
                try:
                    self._store.transition(
                        request_id,
                        'expired',
                        expected_version=current.version,
                    )
                except (ApprovalConflictError, ValueError):
                    pass
            return PermissionResponse(
                action=PermissionAction.DENY,
                feedback=(
                    f'No response within {self._timeout:.0f}s, so this call '
                    'was not run. This is a TIMEOUT, not a refusal by the '
                    'user — nobody saw the request. Do not re-request the '
                    'same approval; finish what you can without it and say '
                    'plainly what is left waiting on approval.'),
            )
        except asyncio.CancelledError:
            current = self._store.get(request_id)
            if current is not None and current.state in ('approved', 'denied'):
                try:
                    self._store.transition(
                        request_id,
                        'resume_queued',
                        expected_version=current.version,
                    )
                except (ApprovalConflictError, ValueError):
                    pass
            elif current is not None and current.state == 'pending':
                self._cancel_pending(request_id, current.version)
            raise
        except Exception as exc:
            return PermissionResponse(
                action=PermissionAction.DENY,
                feedback=(
                    'Permission request could not be persisted: '
                    f'{type(exc).__name__}'),
            )
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def awaiting_request_ids(self) -> set:
        """Every request still open for an answer.

        A host replaying a reconnected turn needs this to tell a card that is
        still live from one that was already decided.
        """
        with self._pending_lock:
            return {
                request_id
                for request_id, pending in self._pending.items()
                if not pending.future.done()
            }

    def is_awaiting(self, request_id: str) -> bool:
        """Whether this request is still open for an answer.

        Public because a host has to ask before routing a click, and reaching
        into ``_pending`` to ask makes the host's code depend on how pending
        asks happen to be stored — which is how adding a field to that record
        turned every approval click into a 500.
        """
        with self._pending_lock:
            pending = self._pending.get(request_id)
        return pending is not None and not pending.future.done()

    def _cancel_pending(self, request_id: str, version: int) -> None:
        try:
            self._store.transition(
                request_id,
                'cancelled',
                expected_version=version,
            )
        except (ApprovalConflictError, KeyError, ValueError):
            pass

    @staticmethod
    def _set_future_result(
        future: asyncio.Future[PermissionResponse],
        response: PermissionResponse,
    ) -> None:
        if not future.done():
            future.set_result(response)

    def resolve(
        self,
        request_id: str,
        response: PermissionResponse,
        *,
        token: str | None = None,
        fingerprint: str | None = None,
        decided_by: str = '',
    ) -> bool:
        request = self._store.get(request_id)
        if request is None:
            with self._pending_lock:
                pending = self._pending.get(request_id)
            if pending is None or pending.future.done():
                return False
            pending.loop.call_soon_threadsafe(
                self._set_future_result, pending.future, response)
            return True
        # In-process resolve (TUI tests, same-handler click) may omit the
        # one-time token; an external WebUI must still present it.
        if token is None and request.state == 'pending':
            token = request.token
            fingerprint = fingerprint or request.fingerprint
        state = (
            'denied'
            if response.action == PermissionAction.DENY else 'approved')
        if request.state != 'pending':
            return (
                request.state in (state, 'resume_queued', 'resumed')
                and request.decision == response.action.value
                and request.pattern == (response.pattern or '')
                and request.scope == response.scope
                and request.decision_args == (
                    sanitize_tool_args(response.updated_args)
                    if response.updated_args is not None else None)
            )
        try:
            decided = self._store.transition(
                request_id,
                state,
                expected_version=request.version,
                token=token,
                fingerprint=fingerprint,
                feedback=response.feedback or '',
                decision=response.action.value,
                pattern=response.pattern or '',
                scope=response.scope,
                decided_by=decided_by,
                decision_args=response.updated_args,
            )
        except (ApprovalConflictError, ValueError):
            return False
        with self._pending_lock:
            pending = self._pending.get(request_id)
        if pending is not None:
            if not pending.future.done():
                pending.loop.call_soon_threadsafe(
                    self._set_future_result, pending.future, response)
                return True
        try:
            queued = self._store.transition(
                request_id,
                'resume_queued',
                expected_version=decided.version,
            )
        except (ApprovalConflictError, ValueError):
            return False
        try:
            self._event_emitter.emit({
                'type': 'permission_resume_queued',
                'request_id': request_id,
                'decision': response.action.value,
                'updated_args': queued.decision_args,
                'continuation_token': queued.continuation_token,
                'fingerprint': (
                    queued.decision_fingerprint or queued.fingerprint),
                'version': queued.version,
            })
        except Exception as exc:
            try:
                self._store.transition(
                    request_id,
                    'resume_failed',
                    expected_version=queued.version,
                    feedback=f'Resume delivery failed: {type(exc).__name__}',
                )
            except (ApprovalConflictError, ValueError):
                pass
            return False
        return True

    def resolve_matching(
        self,
        covers: Callable[[str, dict[str, Any]], bool],
        response: PermissionResponse,
    ) -> int:
        """Answer the still-open asks that a decision just made unnecessary.

        A round can put several cards up at once, and answering one of them
        with "always allow" is a statement about a PATTERN, not about that one
        call. Leaving its siblings up asks the user the question they just
        answered — and since a wait has no deadline, an unanswered sibling
        holds the turn open indefinitely rather than being quietly denied.

        Safety confirmations are skipped: those exist precisely so a remembered
        answer cannot stand in for looking at this one.
        """
        resolved = 0
        with self._pending_lock:
            items = list(self._pending.items())
        for request_id, pending in items:
            if pending.forced or pending.future.done():
                continue
            if not covers(pending.tool_name, pending.tool_args):
                continue
            pending.loop.call_soon_threadsafe(
                self._set_future_result, pending.future, response)
            with self._pending_lock:
                self._pending.pop(request_id, None)
            resolved += 1
        return resolved

    def cancel_pending(self, feedback: str = 'Session closed') -> int:
        """Answer every outstanding ask so nothing is left waiting on a person
        who has gone. Returns how many were resolved.

        Needed once waits can be unbounded: a suspended ask holds its turn, and
        a held turn is exempt from idle reclamation, so an abandoned prompt
        would otherwise pin its session for the life of the process.
        """
        resolved = 0
        deny = PermissionResponse(
            action=PermissionAction.DENY, feedback=feedback)
        with self._pending_lock:
            items = list(self._pending.items())
        for request_id, pending in items:
            if pending.future.done():
                continue
            pending.loop.call_soon_threadsafe(
                self._set_future_result, pending.future, deny)
            resolved += 1
            with self._pending_lock:
                self._pending.pop(request_id, None)
        return resolved
