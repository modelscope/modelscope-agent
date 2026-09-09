# Copyright (c) ModelScope Contributors. All rights reserved.
"""Permission handler for the TUI — one-layer scene-specific confirmation.

Matches Claude Code: a compact header plus a single Select whose rows are
complete decisions (Yes / persist this scoped rule / No). Persist-row edits
happen in place; there is no nested Always-allow wizard.

This handler does NOT declare ``supports_concurrent_asks``, so the enforcer
serializes its asks — one terminal, one menu at a time. That alone is no longer
enough for a quiet screen: the menu runs on the same event loop the renderer
draws from, and a sibling tool call approved a moment earlier can finish while
this menu is up. So ``ask()`` also holds the renderer's output for its duration
(``RichEventSink.hold_output``).
"""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from rich.console import Console
from typing import Any, Optional

from ms_agent.permission.ask_options import (build_ask_options,
                                             format_ask_menu,
                                             parse_ask_choice)
from ms_agent.permission.handler import PermissionAction, PermissionResponse
from ms_agent.permission.provider import (sanitize_sensitive_text,
                                           sanitize_tool_args)
from ms_agent.tui.select import SelectItem, select_async
from ms_agent.tui.theme import DEFAULT_THEME, Theme
from ms_agent.tui.tool_view import tool_header


class TUIPermissionHandler:

    def __init__(self,
                 console: Optional[Console] = None,
                 io: Any = None,
                 theme: Theme = DEFAULT_THEME,
                 renderer: Any = None,
                 pause_live=None) -> None:
        self._console = console or Console()
        self._theme = theme
        # The event renderer, so its draws can be held while this menu owns the
        # terminal. A sibling tool call finishing mid-menu would otherwise print
        # its result line straight through the prompt_toolkit app the user is
        # reading (tool completions arrive per call now, so that overlap is
        # reachable whenever one call is approved while another is still asked).
        self._renderer = renderer
        self._pause_live = pause_live

    @contextmanager
    def _own_screen(self):
        hold = getattr(self._renderer, 'hold_output', None)
        if hold is None:
            yield  # no renderer wired (tests, embedders) — nothing to hold
            return
        with hold():
            yield

    async def ask(
        self,
        tool_name,
        tool_args,
        context,
        suggestions=None,
        call_id='',
        workspace_root='',
    ):
        with self._own_screen():
            return await self._ask(
                tool_name,
                tool_args,
                context,
                suggestions=suggestions,
                workspace_root=workspace_root,
            )

    async def _ask(
        self,
        tool_name,
        tool_args,
        context,
        suggestions=None,
        workspace_root='',
    ):
        options = build_ask_options(
            tool_name,
            tool_args,
            workspace_root=workspace_root or None,
            suggestions=suggestions,
        )
        args_preview = self._format_args(sanitize_tool_args(tool_args))
        context_text = sanitize_sensitive_text(context) if context else ''
        if not sys.stdin.isatty():
            print(
                format_ask_menu(
                    options,
                    tool_name=tool_name,
                    args_preview=args_preview,
                    context=context_text,
                ),
                file=sys.stderr)
            print('choice: ', end='', file=sys.stderr, flush=True)
            loop = asyncio.get_running_loop()
            try:
                raw = await loop.run_in_executor(None, sys.stdin.readline)
            except (EOFError, KeyboardInterrupt):
                return PermissionResponse(action=PermissionAction.DENY)
            if raw == '':
                return PermissionResponse(action=PermissionAction.DENY)
            return parse_ask_choice(raw, options)

        items = [
            SelectItem(
                label=opt.label,
                editable=opt.editable,
                initial=opt.edit_value,
            ) for opt in options
        ]
        if self._pause_live is not None:
            self._pause_live()
        # Compact header: the renderer already printed "• Run ssh …".
        # Dumping pretty JSON here blew past the select window height and
        # overwrote the transcript.
        compact = tool_header(tool_name, tool_args)
        header = f'⚠ Allow this?  {compact}' if compact else '⚠ Allow this tool call?'
        result = await select_async(
            items,
            default=0,
            header=header,
        )
        if result is None:
            return PermissionResponse(action=PermissionAction.DENY)
        option = options[result.index]
        if result.value and option.editable:
            option = option.with_edit(result.value)
        return option.to_response()

    def _format_args(self, tool_args) -> str:
        try:
            s = json.dumps(tool_args, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            s = str(tool_args)
        lines = s.splitlines()
        if len(lines) > 8:
            s = '\n'.join(lines[:8]) + '\n  …'
        elif len(s) > 400:
            s = s[:400] + '…'
        return s
