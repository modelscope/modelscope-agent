# Copyright (c) ModelScope Contributors. All rights reserved.
"""Inline arrow-key selection menu (prompt_toolkit).

A small, reusable single-choice picker rendered *in the scroll flow* (not a
full-screen dialog) — the pattern Claude Code / Qoder / hermes use for
permission prompts and model pickers: a ``❯`` cursor on the highlighted row,
↑/↓ (or Ctrl-P/N) to move, number keys to jump, Enter to confirm, Esc/Ctrl-C
to cancel. The menu erases itself on exit so only the outcome remains.

Editable rows (permission "don't ask again" prefixes) keep the user on the
same layer: highlight the row, type a new prefix, Enter.

Runs on the current event loop via ``run_async`` (the agent's permission ask
executes in the main loop, so this composes cleanly). Callers must guard for a
TTY; there is a plain fallback for pipes/CI.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class SelectItem:
    label: str
    editable: bool = False
    initial: str = ''


@dataclass(frozen=True)
class SelectResult:
    index: int
    value: str | None = None


SelectOption = Union[str, SelectItem]


def format_editable_label(label: str, initial: str, current: str) -> str:
    """Swap the editable token in-place; never append a second copy.

    ``ssh *`` → ``ssh ecs`` must stay ``… don't ask again for ssh ecs``,
    not ``… ssh *: ssh ecs``. ``current`` may be empty while backspacing.
    """
    if initial and initial in label:
        return label.replace(initial, current, 1)
    marker = "don't ask again for "
    if marker in label:
        return label.split(marker, 1)[0] + marker + current
    if not current:
        return label
    return f'{label} {current}'


def _as_item(option: SelectOption) -> SelectItem:
    if isinstance(option, SelectItem):
        return option
    return SelectItem(label=str(option))


async def select_async(options: Sequence[SelectOption],
                       *,
                       default: int = 0,
                       header: Optional[str] = None) -> Optional[SelectResult]:
    """Show an inline menu; return the choice, or None if cancelled.

    ``header`` (optional, may be multi-line) renders above the options *inside*
    the transient menu — its first line bold, the rest dim — so context (e.g. a
    tool name + args) shows during the decision and is erased with the menu,
    leaving nothing behind.

    Falls back to a single blocking line read when stdin is not a TTY (accepts
    a 1-based number, ``y``/``n``, or ``2=custom prefix``).
    """
    if not options:
        return None
    items = [_as_item(opt) for opt in options]
    if not sys.stdin.isatty():
        return await _fallback_numeric(items)
    return await _menu_async(items, default, header)


async def _menu_async(options: Sequence[SelectOption],
                      default: int,
                      header: Optional[str] = None) -> Optional[SelectResult]:
    """The prompt_toolkit menu itself (no TTY guard, so tests can drive it via
    a pipe input + AppSession)."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    options = [_as_item(opt) for opt in options]

    sel = [max(0, min(default, len(options) - 1))]
    buffers = [
        opt.initial if opt.editable else '' for opt in options
    ]
    kb = KeyBindings()

    def _label(i: int) -> str:
        opt = options[i]
        if not opt.editable:
            return opt.label
        return format_editable_label(opt.label, opt.initial, buffers[i])

    @kb.add('up')
    @kb.add('c-p')
    def _up(event) -> None:
        sel[0] = (sel[0] - 1) % len(options)

    @kb.add('down')
    @kb.add('c-n')
    @kb.add('tab')
    def _down(event) -> None:
        sel[0] = (sel[0] + 1) % len(options)

    @kb.add('enter')
    def _accept(event) -> None:
        idx = sel[0]
        value = buffers[idx] if options[idx].editable else None
        event.app.exit(result=SelectResult(idx, value or None))

    @kb.add('escape')
    @kb.add('c-c')
    def _cancel(event) -> None:
        event.app.exit(result=None)

    @kb.add('backspace')
    def _backspace(event) -> None:
        idx = sel[0]
        if options[idx].editable and buffers[idx]:
            buffers[idx] = buffers[idx][:-1]

    @kb.add('<any>')
    def _type(event) -> None:
        key = event.key_sequence[0].key
        if not isinstance(key, str) or len(key) != 1:
            return
        idx = sel[0]
        if options[idx].editable and key.isprintable():
            buffers[idx] += key

    for _i in range(min(len(options), 9)):

        @kb.add(str(_i + 1))
        def _pick(event, i=_i) -> None:
            if options[sel[0]].editable:
                buf = buffers[sel[0]] or options[sel[0]].initial
                buffers[sel[0]] = buf + str(i + 1)
                return
            event.app.exit(result=SelectResult(i, None))

    header_lines = header.splitlines() if header else []

    def _render():
        frags = []
        for j, hl in enumerate(header_lines):
            frags.append(
                ('class:head' if j == 0 else 'class:headdim', hl + '\n'))
        for i, opt in enumerate(options):
            mark = '❯' if i == sel[0] else ' '
            line = f'{mark} {i + 1}. {_label(i)}'
            if opt.editable and i == sel[0]:
                line += '█'
            style = 'class:sel' if i == sel[0] else 'class:opt'
            frags.append((style, line + '\n'))
        frags.append((
            'class:hint',
            '↑/↓ · enter · type to edit persist · esc',
        ))
        return frags

    control = FormattedTextControl(_render, focusable=True, show_cursor=False)
    style = Style.from_dict({
        'sel': 'bold ansicyan',
        'opt': '',
        'head': 'bold ansiyellow',
        'headdim': 'ansibrightblack',
        'hint': 'italic ansibrightblack',
    })
    preview_lines = list(header_lines)
    preview_lines.extend(
        f'❯ {i + 1}. {_label(i)}█' for i in range(len(options)))
    preview_lines.append('↑/↓ · enter · type to edit persist · esc')
    app = Application(
        layout=Layout(
            HSplit(
                [Window(control,
                        wrap_lines=True,
                        height=_wrapped_height(preview_lines))])),
        key_bindings=kb,
        style=style,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    from ms_agent.tui.tty import restore_cooked_tty
    try:
        return await app.run_async()
    finally:
        restore_cooked_tty()


def _wrapped_height(lines: Sequence[str]) -> int:
    """Rows needed at the current terminal width (long labels wrap)."""
    width = max(20, shutil.get_terminal_size((80, 24)).columns)
    total = 0
    for line in lines:
        total += max(1, (max(len(line), 1) + width - 1) // width)
    return max(total, 1)


async def _fallback_numeric(
    options: Sequence[SelectItem],
) -> Optional[SelectResult]:
    """Non-TTY fallback: one line. ``2=prefix`` edits the persist row."""
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        raw = (await loop.run_in_executor(None, input, 'choice: ')).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in ('n', 'no', 'esc'):
        return SelectResult(len(options) - 1, None)
    if lowered in ('y', 'yes'):
        return SelectResult(0, None)
    extra = ''
    if raw[0].isdigit():
        i = 1
        while i < len(raw) and raw[i].isdigit():
            i += 1
        try:
            idx = int(raw[:i]) - 1
        except ValueError:
            return None
        extra = raw[i:].lstrip('= ').strip()
        if 0 <= idx < len(options):
            value = extra if extra and options[idx].editable else None
            return SelectResult(idx, value)
    return None
