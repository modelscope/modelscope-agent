"""Tests for PermissionMemory."""

import json

import pytest

from ms_agent.permission.memory import PermissionMemory


@pytest.fixture
def memory(tmp_path):
    project_path = tmp_path / 'project'
    project_path.mkdir()
    global_path = tmp_path / 'global' / 'permission_memory.json'
    return PermissionMemory(project_path=project_path, global_path=global_path)


class TestAdd:
    def test_add_project(self, memory, tmp_path):
        memory.add('file_system---read_*', scope='project')
        entries = memory.list_all()
        assert len(entries) == 1
        assert entries[0].pattern == 'file_system---read_*'
        assert entries[0].scope == 'project'

    def test_add_global(self, memory):
        memory.add('web_search---*', scope='global')
        entries = memory.list_all()
        assert len(entries) == 1
        assert entries[0].scope == 'global'


class TestMatches:
    def test_match_persistent(self, memory):
        memory.add('file_system---read_*', scope='project')
        assert memory.matches('file_system---read_file', {})
        assert not memory.matches('file_system---write_file', {})

    def test_match_session(self, memory):
        memory.add_session('code_executor---shell_executor:ls *')
        assert memory.matches('code_executor---shell_executor', {'command': 'ls -la'})
        assert not memory.matches('code_executor---shell_executor', {'command': 'rm file'})

    def test_match_content_pattern(self, memory):
        memory.add('code_executor---shell_executor:pip *', scope='project')
        assert memory.matches('code_executor---shell_executor', {'command': 'pip install requests'})
        assert not memory.matches('code_executor---shell_executor', {'command': 'npm install'})


class TestRevoke:
    def test_revoke(self, memory):
        memory.add('file_system---*', scope='project')
        assert memory.matches('file_system---read_file', {})
        count = memory.revoke('file_system---*')
        assert count == 1
        assert not memory.matches('file_system---read_file', {})

    def test_revoke_nonexistent(self, memory):
        count = memory.revoke('nonexistent')
        assert count == 0


class TestPersistence:
    def test_reload(self, tmp_path):
        project_path = tmp_path / 'project'
        project_path.mkdir()
        global_path = tmp_path / 'global' / 'permission_memory.json'

        mem1 = PermissionMemory(project_path=project_path, global_path=global_path)
        mem1.add('file_system---*', scope='project')
        mem1.add('web_search---*', scope='global')

        mem2 = PermissionMemory(project_path=project_path, global_path=global_path)
        assert mem2.matches('file_system---read_file', {})
        assert mem2.matches('web_search---fetch_page', {})

    def test_session_not_persisted(self, tmp_path):
        project_path = tmp_path / 'project'
        project_path.mkdir()

        mem1 = PermissionMemory(project_path=project_path)
        mem1.add_session('temp_pattern')

        mem2 = PermissionMemory(project_path=project_path)
        assert not mem2.matches('temp_pattern', {})

    def test_legacy_json_gets_stable_id(self, tmp_path):
        project_path = tmp_path / 'project'
        memory_file = project_path / '.ms_agent' / 'permission_memory.json'
        memory_file.parent.mkdir(parents=True)
        memory_file.write_text(json.dumps([{
            'pattern': 'legacy---tool',
            'scope': 'project',
            'source': 'user',
            'created_at': '2025-01-01T00:00:00+00:00',
        }]))

        first = PermissionMemory(project_path=project_path)
        second = PermissionMemory(project_path=project_path)

        assert first.list_all()[0].id
        assert first.list_all()[0].id == second.list_all()[0].id

    def test_two_instances_do_not_overwrite_each_other(self, tmp_path):
        project_path = tmp_path / 'project'
        project_path.mkdir()
        global_path = tmp_path / 'global' / 'permission_memory.json'
        first = PermissionMemory(
            project_path=project_path, global_path=global_path)
        stale = PermissionMemory(
            project_path=project_path, global_path=global_path)

        first.add('first---tool')
        stale.add('second---tool')

        reloaded = PermissionMemory(
            project_path=project_path, global_path=global_path)
        assert {entry.pattern for entry in reloaded.list()} == {
            'first---tool',
            'second---tool',
        }

    def test_pathless_memory_keeps_prior_session_rules(self):
        memory = PermissionMemory(project_path=None, global_path=None)
        memory.add('first---tool', scope='project')
        memory.add('second---tool', scope='project')
        assert memory.matches('first---tool', {})
        assert memory.matches('second---tool', {})

    def test_matches_sees_revocation_from_another_instance(self, tmp_path):
        project_path = tmp_path / 'project'
        project_path.mkdir()
        global_path = tmp_path / 'global' / 'permission_memory.json'
        live = PermissionMemory(
            project_path=project_path, global_path=global_path)
        live.add('live---tool')
        other = PermissionMemory(
            project_path=project_path, global_path=global_path)
        other.revoke('live---tool')
        assert not live.matches('live---tool', {})


class TestCrud:
    def test_list_update_delete_by_stable_id(self, memory):
        created = memory.add('old---tool', scope='project')

        assert memory.list(scope='project') == [created]
        assert created.kind == 'tool'
        updated = memory.update(created.id, pattern='new---tool')
        assert updated.id == created.id
        assert updated.pattern == 'new---tool'
        assert not memory.matches('old---tool', {})
        assert memory.matches('new---tool', {})
        assert memory.delete(created.id)
        assert not memory.delete(created.id)
        assert memory.list() == []

    def test_rule_kind_is_inferred_and_editable(self, memory):
        domain = memory.add('web---fetch:domain:example.com')
        shell = memory.add('code_executor---shell_executor:git *')

        assert domain.kind == 'domain'
        assert shell.kind == 'shell'
        assert memory.update(domain.id, kind='tool').kind == 'tool'

    def test_duplicate_update_does_not_remove_original(self, memory):
        first = memory.add('first---tool')
        memory.add('second---tool')

        with pytest.raises(ValueError):
            memory.update(first.id, pattern='second---tool')

        assert memory.matches('first---tool', {})


class TestEdgeCases:
    def test_no_project_path(self, tmp_path):
        global_path = tmp_path / 'global' / 'permission_memory.json'
        mem = PermissionMemory(project_path=None, global_path=global_path)
        mem.add('test', scope='global')
        assert mem.matches('test', {})

    def test_corrupt_file(self, tmp_path):
        project_path = tmp_path / 'project'
        project_path.mkdir()
        mem_file = project_path / '.ms_agent' / 'permission_memory.json'
        mem_file.parent.mkdir(parents=True)
        mem_file.write_text('not json')

        mem = PermissionMemory(project_path=project_path)
        assert mem.list_all() == []

    def test_cross_scope_move_rolls_back_if_second_save_fails(
            self, tmp_path, monkeypatch):
        project_path = tmp_path / 'project'
        project_path.mkdir()
        global_path = tmp_path / 'global' / 'permission_memory.json'
        memory = PermissionMemory(
            project_path=project_path, global_path=global_path)
        entry = memory.add('move---tool', scope='project')

        original = PermissionMemory._save

        def flaky(self, scope):
            if scope == 'global':
                raise OSError('disk full')
            return original(self, scope)

        monkeypatch.setattr(PermissionMemory, '_save', flaky)
        with pytest.raises(OSError):
            memory.update(entry.id, scope='global')

        assert memory.matches('move---tool', {})
        assert [item.scope for item in memory.list_all()
                if item.pattern == 'move---tool'] == ['project']

        reloaded = PermissionMemory(
            project_path=project_path, global_path=global_path)
        assert [item.scope for item in reloaded.list_all()
                if item.pattern == 'move---tool'] == ['project']
        if global_path.exists():
            assert 'move---tool' not in global_path.read_text(
                encoding='utf-8')
