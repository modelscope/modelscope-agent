"""One-layer permission option builder and choice parser."""

from ms_agent.permission.ask_options import (
    build_ask_options,
    format_ask_menu,
    parse_ask_choice,
)
from ms_agent.permission.handler import PermissionAction


def test_shell_persist_is_prefix_and_editable():
    options = build_ask_options(
        'code_executor---shell_executor',
        {'command': 'echo permission-e2e-ok'},
    )
    assert [o.key for o in options] == ['yes', 'persist', 'no']
    persist = options[1]
    assert persist.action == PermissionAction.ALLOW_ALWAYS
    assert persist.pattern == 'code_executor---shell_executor:echo *'
    assert persist.editable
    assert "don't ask again for echo *" in persist.label
    menu = format_ask_menu(options, tool_name='code_executor---shell_executor')
    assert 'Always allow' not in menu
    assert 'Allow for this session' not in menu


def test_compound_shell_has_no_persist_row():
    options = build_ask_options(
        'code_executor---shell_executor',
        {'command': 'cd src && npm test'},
    )
    assert [o.key for o in options] == ['yes', 'no']


def test_url_persist_is_domain():
    options = build_ask_options(
        'web---fetch',
        {'url': 'https://example.com/path'},
    )
    persist = options[1]
    assert persist.pattern == 'web---fetch:domain:example.com'
    assert 'example.com' in persist.label
    assert persist.action == PermissionAction.ALLOW_ALWAYS


def test_private_url_has_no_persist_row():
    options = build_ask_options(
        'web---fetch',
        {'url': 'http://127.0.0.1/admin'},
    )
    assert [o.key for o in options] == ['yes', 'no']


def test_file_inside_workspace_is_session():
    options = build_ask_options(
        'file_system---write_file',
        {'path': '/proj/notes.txt'},
        workspace_root='/proj',
    )
    persist = options[1]
    assert persist.action == PermissionAction.ALLOW_SESSION
    assert persist.pattern == 'file_system---write_file|file_system---edit_file'
    assert 'this project this session' in persist.label


def test_file_outside_workspace_is_session_directory():
    options = build_ask_options(
        'file_system---write_file',
        {'path': '/tmp/outside/a.txt'},
        workspace_root='/proj',
    )
    persist = options[1]
    assert persist.action == PermissionAction.ALLOW_SESSION
    assert '/tmp/outside/' in persist.label


def test_parse_choice_one_layer_edit():
    options = build_ask_options(
        'code_executor---shell_executor',
        {'command': 'echo hi'},
    )
    once = parse_ask_choice('1', options)
    assert once.action == PermissionAction.ALLOW_ONCE
    persist = parse_ask_choice('2', options)
    assert persist.action == PermissionAction.ALLOW_ALWAYS
    assert persist.pattern == 'code_executor---shell_executor:echo *'
    edited = parse_ask_choice('2=echo permission-e2e*', options)
    assert edited.pattern == 'code_executor---shell_executor:echo permission-e2e*'
    deny = parse_ask_choice('3', options)
    assert deny.action == PermissionAction.DENY
    assert parse_ask_choice(None, options).action == PermissionAction.DENY
    assert parse_ask_choice('', options).action == PermissionAction.DENY
    assert parse_ask_choice('2*', options).action == PermissionAction.DENY
    assert parse_ask_choice('2=echo *|rm *', options).action == PermissionAction.DENY
