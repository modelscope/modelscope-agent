# Copyright (c) ModelScope Contributors. All rights reserved.
"""Inline selection menu key handling (driven via a prompt_toolkit pipe).

Validates the arrow-key / number-key / enter / cancel logic without a real
terminal, so the permission menu's behavior is regression-guarded.
"""
import asyncio

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from ms_agent.tui.select import (
    SelectItem,
    _menu_async,
    format_editable_label,
)

OPTS = ['Yes', "Yes, and don't ask again for echo *", 'No']


def _run(keys: str, default: int = 0):
    async def go():
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                inp.send_text(keys)
                return await _menu_async(OPTS, default)

    return asyncio.run(go())


def test_enter_picks_default():
    assert _run('\r').index == 0


def test_enter_picks_given_default():
    assert _run('\r', default=2).index == 2


def test_down_then_enter():
    assert _run('\x1b[B\r').index == 1  # ↓, Enter


def test_down_wraps_from_last_to_first():
    assert _run('\x1b[B\x1b[B\x1b[B\r').index == 0


def test_up_then_enter_wraps():
    assert _run('\x1b[A\r').index == len(OPTS) - 1


def test_tab_moves_down():
    assert _run('\t\r').index == 1


def test_number_key_jumps_and_selects():
    assert _run('3').index == 2


def test_ctrl_c_cancels_to_none():
    assert _run('\x03') is None


def _run_with_header(keys: str, header: str):
    async def go():
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                inp.send_text(keys)
                return await _menu_async(OPTS, 0, header)

    return asyncio.run(go())


def test_wrapped_height_counts_rows():
    from ms_agent.tui.select import _wrapped_height
    assert _wrapped_height(['a', 'b', 'c']) == 3
    long = 'x' * 200
    assert _wrapped_height([long]) >= 2


def test_format_editable_label_replaces_token():
    label = "Yes, and don't ask again for ssh *"
    assert format_editable_label(label, 'ssh *', 'ssh *') == label
    assert (
        format_editable_label(label, 'ssh *', 'ssh ecs')
        == "Yes, and don't ask again for ssh ecs")
    assert format_editable_label(label, 'ssh *', 'ssh ') == (
        "Yes, and don't ask again for ssh ")
    assert format_editable_label(label, 'ssh *', '') == (
        "Yes, and don't ask again for ")
    assert ':' not in format_editable_label(label, 'ssh *', 'ssh ecs')


EDITABLE = [
    SelectItem('Yes'),
    SelectItem(
        "Yes, and don't ask again for ssh *",
        editable=True,
        initial='ssh *',
    ),
    SelectItem('No'),
]


def _run_editable(keys: str):
    async def go():
        with create_pipe_input() as inp:
            with create_app_session(input=inp, output=DummyOutput()):
                inp.send_text(keys)
                return await _menu_async(EDITABLE, 0)

    return asyncio.run(go())


def test_editable_row_backspace_then_type_keeps_one_token():
    # ↓, backspace (*), type ecs, Enter → persist prefix is ssh ecs
    result = _run_editable('\x1b[B\x08ecs\r')
    assert result.index == 1
    assert result.value == 'ssh ecs'


def test_editable_row_clear_then_type_does_not_restore_initial():
    result = _run_editable('\x1b[B' + '\x08' * 6 + 'git *\r')
    assert result.index == 1
    assert result.value == 'git *'
