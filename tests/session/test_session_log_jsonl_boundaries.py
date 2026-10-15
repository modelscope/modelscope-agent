# Copyright (c) ModelScope Contributors. All rights reserved.
"""JSONL records are separated by LF, not Unicode characters in JSON strings."""
import pytest

from ms_agent.session.session_log import SessionLog


@pytest.mark.parametrize('separator', ['\u0085', '\u2028', '\u2029'])
@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
@pytest.mark.parametrize('final_newline', [True, False])
def test_messages_restore_unicode_content(tmp_path, separator, newline,
                                          final_newline):
    content = f'before{separator}after'
    log = SessionLog(tmp_path, 'unicode')
    log.append({'role': 'user', 'content': content})
    path = tmp_path / 'unicode.jsonl'
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    if not final_newline:
        raw = raw.rstrip(b'\n')
    path.write_bytes(raw.replace(b'\n', newline))

    restored = SessionLog(tmp_path, 'unicode').get_all_messages()

    assert len(restored) == 1
    assert restored[0]['content'] == content
    assert restored[0]['seq'] == 0


@pytest.mark.parametrize('separator', ['\u0085', '\u2028', '\u2029'])
def test_sequence_continues_after_unicode_message(tmp_path, separator):
    log = SessionLog(tmp_path, 'unicode')
    log.append({'role': 'user', 'content': f'before{separator}after'})

    reopened = SessionLog(tmp_path, 'unicode')

    assert reopened.append({'role': 'assistant', 'content': 'reply'}) == 1


@pytest.mark.parametrize('separator', ['\u0085', '\u2028', '\u2029'])
@pytest.mark.parametrize('writer,reader,key', [
    ('record_compaction', 'get_compaction_events', 'summary'),
    ('record_error', 'get_errors', 'message'),
    ('record_permission', 'get_permissions', 'tool_name'),
    ('record_loop_end', 'get_loop_ends', 'message'),
    ('record_skill_invocation', 'get_skill_invocations', 'original_text'),
])
def test_event_readers_preserve_unicode(tmp_path, separator, writer, reader,
                                        key):
    content = f'before{separator}after'
    log = SessionLog(tmp_path, 'unicode')
    getattr(log, writer)({key: content})

    reopened = SessionLog(tmp_path, 'unicode')
    events = getattr(reopened, reader)()

    assert len(events) == 1
    assert events[0][key] == content
    assert reopened.get_all_messages() == []
    assert reopened.append({'role': 'user', 'content': 'next'}) == 1
