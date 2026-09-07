"""PermissionEnforcer: outer-layer user-intent permission control.

Checks blacklist/whitelist, session/persistent memory, and falls back to
the PermissionHandler for interactive user confirmation.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Literal

from .config import PermissionConfig
from .handler import (AutoPermissionHandler, PermissionAction,
                      PermissionHandler, PermissionResponse)
from .matcher import CONTENT_SEP, PermissionMatcher
from .memory import PermissionMemory
from .provider import (PermissionDecisionProvider, ProviderDecision,
                       request_provider_decision)
from .suggestions import generate_suggestions
from ms_agent.utils import get_logger

logger = get_logger()


@dataclass(frozen=True)
class PermissionDecision:
    action: Literal['allow', 'deny', 'ask']
    reason: str
    updated_args: dict[str, Any] | None = None
    #: Whether a standing answer may satisfy this confirmation next time.
    #:
    #: Safety confirmations default to False — a remembered answer must not
    #: stand in for looking at THIS call. But that is not true of all of them
    #: equally. "This command runs code I cannot analyse" is a thing a user can
    #: reasonably decide once for a project, the way they decide about `git`;
    #: "this reads a private key" is not. Marking the first kind rememberable
    #: is what keeps the confirmation useful — an ask that reappears no matter
    #: how the user answers it does not make anyone safer, it just teaches them
    #: to turn confirmations off entirely.
    rememberable: bool = False


class PermissionEnforcer:
    """Outer-layer permission enforcement based on user intent and configuration."""

    def __init__(
        self,
        config: PermissionConfig,
        handler: PermissionHandler | None = None,
        memory: PermissionMemory | None = None,
        provider: PermissionDecisionProvider | None = None,
    ) -> None:
        self._config = config
        self._handler = handler or AutoPermissionHandler()
        self._memory = memory or PermissionMemory()
        self._provider = provider
        self._matcher = PermissionMatcher()
        # Parallel tool calls (asyncio.gather in ToolManager.parallel_call_tool)
        # reach the handler concurrently. Whether that is safe is the HANDLER's
        # property, not a blanket rule: a terminal-bound one (CLI prompt / TUI
        # menu) deadlocks with N prompts fighting over one stdin, while a
        # request_id-keyed UI wants them all at once. Handlers opt in with
        # ``supports_concurrent_asks``; everyone else is serialized with a lock
        # created lazily per running loop (the per-turn TUI uses a fresh loop
        # each turn, so a single init-time Lock would bind to the wrong one).
        self._ask_lock: asyncio.Lock | None = None
        self._ask_lock_loop = None
        self._ask_thread_lock = threading.RLock()

    def _ask_lock_for_loop(self) -> 'asyncio.Lock':
        loop = asyncio.get_running_loop()
        with self._ask_thread_lock:
            if self._ask_lock is None or self._ask_lock_loop is not loop:
                self._ask_lock = asyncio.Lock()
                self._ask_lock_loop = loop
            return self._ask_lock

    async def _ask_user(self,
                        *,
                        forced: bool = False,
                        **kwargs) -> PermissionResponse | None:
        """Put one ask in front of the user, serialized unless the handler
        declares it can service several at once.

        Returns ``None`` when a queued ask turned out to be unnecessary: while
        it waited for the lock, an earlier ask in the same round was answered
        with allow_session / allow_always covering this call too, so prompting
        again would ask the user something they just answered. ``forced`` asks
        (a SafetyGuard confirmation) skip that shortcut — memory must never
        bypass a safety ask.
        """
        # ``call_id`` / ``workspace_root`` are newer, optional kwargs. A handler
        # that predates them — or a lightweight test double — need not accept
        # them; drop them for such handlers so their fixed signature keeps
        # working.
        if 'call_id' in kwargs and not self._handler_accepts('call_id'):
            kwargs.pop('call_id')
        if (
            'workspace_root' in kwargs
            and not self._handler_accepts('workspace_root')
        ):
            kwargs.pop('workspace_root')
        if getattr(self._handler, 'supports_concurrent_asks', False):
            # A handler that services asks concurrently needs to know which of
            # them are safety confirmations, so a remembered answer is never
            # applied to one. Older handlers with a fixed signature don't.
            if self._handler_accepts('forced'):
                return await self._handler.ask(forced=forced, **kwargs)
            return await self._handler.ask(**kwargs)
        async with self._ask_lock_for_loop():
            if not forced and self._memory.matches(kwargs['tool_name'],
                                                   kwargs['tool_args']):
                return None
            return await self._handler.ask(**kwargs)

    def _can_ask_human(self) -> bool:
        """Whether a real person can actually answer a prompt right now.

        ``AutoPermissionHandler`` is the stand-in used headlessly and it always
        answers "allow", so treating it as an asker would turn every ask rule
        into a no-op.
        """
        return not isinstance(self._handler, AutoPermissionHandler)

    def _handler_accepts(self, param: str) -> bool:
        try:
            sig = inspect.signature(self._handler.ask)
        except (TypeError, ValueError):
            return True  # can't introspect — assume it takes it, don't strip
        params = sig.parameters.values()
        return (any(p.name == param for p in params)
                or any(p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in params))

    async def check(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        force_decision: PermissionDecision | None = None,
        call_id: str = '',
    ) -> PermissionDecision:
        # ``call_id`` (the tool_call this ask is gating) is threaded to the
        # handler so a UI can record/correlate the decision against the exact
        # call — important when a round fires several identical tool calls in
        # parallel. Empty when the LLM adapter didn't assign an id yet;
        # handlers must tolerate that.
        # 1. Blacklist → deny (not overridable in any mode)
        for pattern in self._config.blacklist:
            if self._matcher.match_with_content(pattern, tool_name, tool_args):
                return PermissionDecision(
                    action='deny',
                    reason=f'Denied by blacklist rule: {pattern}',
                )

        if force_decision and force_decision.action == 'deny':
            return force_decision

        if force_decision and force_decision.action == 'ask':
            rememberable = getattr(force_decision, 'rememberable', False)
            if rememberable and self._memory.matches(tool_name, tool_args):
                return PermissionDecision(
                    action='allow',
                    reason='Allowed by remembered permission',
                )
            if not self._human_approval_available():
                return PermissionDecision(
                    action='deny',
                    reason=(
                        'Safety approval requires a human, but no human '
                        'approval handler is available'),
                )
            suggestions = generate_suggestions(tool_name, tool_args)
            response = await self._ask_user(
                forced=not rememberable,
                tool_name=tool_name,
                tool_args=tool_args,
                context=force_decision.reason or '',
                suggestions=suggestions,
                call_id=call_id,
                workspace_root=self._workspace_root(),
            )
            return self._process_response(response, tool_name, tool_args)

        # 1b. Ask rules → confirm, in EVERY mode. Until now this config existed
        # but only the hook path consulted it, so an ask rule was silently
        # inert on the ordinary route. It outranks the mode and the whitelist —
        # that is the whole point of "ask even under full access" — but not the
        # user's own remembered answer below, so consenting once still sticks.
        ask_rule = next(
            (p for p in self._config.ask_rules
             if self._matcher.match_with_content(p, tool_name, tool_args)),
            None,
        )
        if ask_rule and not self._can_ask_human():
            # Headless (AutoPermissionHandler allows everything): there is
            # nobody to confirm, and silently running the thing an ask rule was
            # written to gate would be worse than refusing.
            return PermissionDecision(
                action='deny',
                reason=(f'Ask rule matched: {ask_rule}; no interactive '
                        'handler is attached to confirm it'),
            )

        # 2. Auto / strict / full-access → allow (safety handled by SafetyGuard
        # + ask_resolver). Ask rules still confirm, including under full-access.
        if self._config.mode in ('auto', 'strict', 'full_access') and not ask_rule:
            return PermissionDecision(
                action='allow',
                reason=f'{self._config.mode.capitalize()} mode')

        # 3. Whitelist → allow
        if not ask_rule:
            for pattern in self._config.whitelist:
                if self._matcher.match_with_content(pattern, tool_name,
                                                    tool_args):
                    return PermissionDecision(
                        action='allow',
                        reason=f'Allowed by whitelist rule: {pattern}',
                    )

        # 4. Memory (session + persistent) → allow
        if self._memory.matches(tool_name, tool_args):
            return PermissionDecision(
                action='allow',
                reason='Allowed by remembered permission',
            )

        # 5. Delegate unknown calls to an injected automated provider.
        # Ask rules outrank the mode, so a matching network command still
        # needs a human rather than the LLM classifier.
        if self._config.mode == 'delegate' and not ask_rule:
            return await self._delegate(tool_name, tool_args, call_id=call_id)

        # 6. Ask user via handler (serialized unless it opts into concurrency)
        if not self._can_ask_human():
            return PermissionDecision(
                action='deny',
                reason='Interactive approval requires a human handler',
            )
        suggestions = generate_suggestions(tool_name, tool_args)
        response = await self._ask_user(
            tool_name=tool_name,
            tool_args=tool_args,
            context='',
            suggestions=suggestions,
            call_id=call_id,
            workspace_root=self._workspace_root(),
        )

        return self._process_response(response, tool_name, tool_args)

    def _remember_pattern(self, response: PermissionResponse, tool_name: str,
                          tool_args: dict[str, Any]) -> str:
        """What to remember when the caller named no pattern of its own.

        The bare tool name means "allow this TOOL" — for the shell that is
        every future command, so approving ``ls -la`` once silently handed over
        unrestricted shell access. Prefer instead the most specific generated
        suggestion that is no broader than the tool itself (``<tool>:ls *``);
        a suggestion that WIDENS the scope (``<server>---*``) is not a fallback
        anyone asked for.
        """
        if response.pattern:
            return response.pattern
        # Prefer a tool-scoped glob (``<tool>:ls *``) over an exact snapshot of
        # this one invocation. The exact suggestion is listed first so a UI can
        # offer it, but a patternless "always allow" means the command family.
        fallback = tool_name
        for s in generate_suggestions(tool_name, tool_args):
            if not (s == tool_name
                    or s.startswith(f'{tool_name}{CONTENT_SEP}')):
                continue
            if '*' in s or '?' in s:
                return s
            if fallback == tool_name:
                fallback = s
        return fallback

    def _release_asks_covered_by_memory(self, pattern: str) -> int:
        """Apply a just-remembered answer to the other cards still on screen.

        Only reaches handlers that show several cards at once. There, "always
        allow" is a statement about a pattern, so re-asking about a sibling the
        pattern covers asks a question the user has answered. It also stops
        mattering only in one direction: since a wait has no deadline, an
        unanswered sibling now holds the turn open instead of being denied
        after a couple of minutes, so leaving them up turns a mis-set
        expectation into a stuck conversation.
        """
        resolver = getattr(self._handler, 'resolve_matching', None)
        if resolver is None:
            return 0
        released = resolver(
            lambda name, args: self._memory.matches(name, args),
            PermissionResponse(action=PermissionAction.ALLOW_ONCE),
        )
        if released:
            logger.info(
                'permission pattern %r also released %d waiting request(s)',
                pattern, released)
        return released

    def _workspace_root(self) -> str:
        root = getattr(self._memory, 'project_root', None)
        return str(root) if root else ''

    def _human_approval_available(self) -> bool:
        # AutoPermissionHandler cannot prompt — ignore the YAML flag.
        # Otherwise honor ``human_approval_available``, and treat interactive
        # mode as a person at the terminal even if the constructor defaulted
        # the flag to False.
        if isinstance(self._handler, AutoPermissionHandler):
            return False
        if self._config.human_approval_available:
            return True
        return self._config.mode == 'interactive'

    async def _delegate(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        *,
        call_id: str,
    ) -> PermissionDecision:
        suggestions = generate_suggestions(tool_name, tool_args)
        if self._provider is None:
            provider_decision = ProviderDecision(
                'uncertain', 'No permission decision provider is configured')
        else:
            provider_decision = await request_provider_decision(
                self._provider,
                tool_name=tool_name,
                tool_args=tool_args,
                context='',
                suggestions=suggestions,
                timeout=self._config.provider_timeout,
            )

        if provider_decision.action == 'allow_once':
            return PermissionDecision(
                action='allow',
                reason=provider_decision.reason or 'Delegated provider allowed once',
            )
        if provider_decision.action == 'deny':
            return PermissionDecision(
                action='deny',
                reason=(
                    provider_decision.feedback
                    or provider_decision.reason
                    or 'Delegated provider denied'
                ),
            )

        context = (
            provider_decision.reason
            or 'Delegated provider was uncertain')
        if self._human_approval_available():
            response = await self._ask_user(
                tool_name=tool_name,
                tool_args=tool_args,
                context=context,
                suggestions=suggestions,
                call_id=call_id,
                workspace_root=self._workspace_root(),
            )
            return self._process_response(response, tool_name, tool_args)
        return PermissionDecision(
            action='deny',
            reason=f'Delegated provider uncertain: {context}',
        )

    def _process_response(
        self,
        response: PermissionResponse | None,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> PermissionDecision:
        if response is None:
            # The ask was skipped: memory started covering this call while it
            # was queued behind another one (see _ask_user).
            return PermissionDecision(
                action='allow',
                reason='Allowed by remembered permission',
            )

        if response.action == PermissionAction.ALLOW_ONCE:
            return PermissionDecision(
                action='allow', reason='User allowed once')

        if response.action == PermissionAction.ALLOW_SESSION:
            pattern = self._remember_pattern(response, tool_name, tool_args)
            self._memory.add_session(pattern)
            self._release_asks_covered_by_memory(pattern)
            return PermissionDecision(
                action='allow',
                reason=f'User allowed for session (pattern: {pattern})',
            )

        if response.action == PermissionAction.ALLOW_ALWAYS:
            pattern = self._remember_pattern(response, tool_name, tool_args)
            content = (
                pattern.split(CONTENT_SEP, 1)[1]
                if CONTENT_SEP in pattern else pattern)
            if '|' in content:
                return PermissionDecision(
                    action='deny',
                    reason='Refusing to persist a rule that uses | alternatives',
                )
            self._memory.add(pattern, scope=response.scope, source='user')
            self._release_asks_covered_by_memory(pattern)
            return PermissionDecision(
                action='allow',
                reason=(
                    f'User allowed always '
                    f'(scope: {response.scope}, pattern: {pattern})'
                ),
            )

        if response.action == PermissionAction.MODIFY:
            updated = response.updated_args or tool_args
            for pattern in self._config.blacklist:
                if self._matcher.match_with_content(
                        pattern, tool_name, updated):
                    return PermissionDecision(
                        action='deny',
                        reason=f'Denied by blacklist after edit: {pattern}',
                    )
            return PermissionDecision(
                action='allow',
                reason='User modified args',
                updated_args=updated,
            )

        if response.action == PermissionAction.DENY:
            return PermissionDecision(
                action='deny',
                reason=response.feedback or 'User denied',
            )

        return PermissionDecision(action='deny', reason='Unknown action')
