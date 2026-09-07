# Copyright (c) ModelScope Contributors. All rights reserved.
"""Release guards must reject version drift and overwriting published bytes."""
import importlib.util
import pytest
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.dev_scripts/webui'))
import check_webui_release as check  # noqa: E402
import publish_webui_release as publish  # noqa: E402
import webui_packaging as packaging  # noqa: E402


def test_resource_selection_omits_undeclared_local_files(tmp_path, monkeypatch):
    webui = tmp_path / 'webui'
    for name in ('backend/app/main.py', 'backend/.env',
                 '.claude/skills/local/SKILL.md', 'frontend/local-notes.md'):
        file = webui / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('local content')
    inputs = tmp_path / 'resource-files.txt'
    inputs.write_text('# Package inputs\nbackend/app/main.py\n')
    monkeypatch.setattr(packaging, 'WEBUI', webui)
    monkeypatch.setattr(packaging, 'RESOURCE_LIST', inputs)
    assert packaging.resource_paths(include_build=False) == [
        'backend/app/main.py'
    ]


@pytest.mark.parametrize('name', [
    '../secret', '/tmp/secret', 'backend/../../secret',
    'backend\\..\\..\\secret'
])
def test_resource_list_rejects_paths_outside_webui(tmp_path, monkeypatch, name):
    inputs = tmp_path / 'resource-files.txt'
    inputs.write_text(name + '\n')
    monkeypatch.setattr(packaging, 'WEBUI', tmp_path / 'webui')
    monkeypatch.setattr(packaging, 'RESOURCE_LIST', inputs)
    with pytest.raises(RuntimeError, match='Invalid WebUI resource path'):
        packaging.resource_paths(include_build=False)


def test_resource_list_rejects_symlinks(tmp_path, monkeypatch):
    webui = tmp_path / 'webui'
    webui.mkdir()
    private = tmp_path / 'private'
    private.write_text('private content')
    (webui / 'config').symlink_to(private)
    inputs = tmp_path / 'resource-files.txt'
    inputs.write_text('config\n')
    monkeypatch.setattr(packaging, 'WEBUI', webui)
    monkeypatch.setattr(packaging, 'RESOURCE_LIST', inputs)
    with pytest.raises(RuntimeError, match='Invalid WebUI resource path'):
        packaging.resource_paths(include_build=False)


@pytest.mark.parametrize('version', ['1.7.0rc0', '1.7.0rc1', '1.7.0'])
def test_rc_and_stable_use_the_same_tag_rule(version):
    check.check_tag(version, 'v' + version)


@pytest.mark.parametrize('version,tag', [('1.7.0rc0', 'v1.7.0'),
                                         ('1.7.0', 'v1.7.0rc1'),
                                         ('1.7.0', '1.7.0'),
                                         ('1.7.0-rc0', 'v1.7.0-rc0')])
def test_mismatched_or_ambiguous_tags_fail(version, tag):
    with pytest.raises(ValueError):
        check.check_tag(version, tag)


def test_httpx_requirement_is_not_mistaken_for_a_url(tmp_path):
    spec = importlib.util.spec_from_file_location('sdk_setup',
                                                  ROOT / 'setup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requirements = tmp_path / 'requirements.txt'
    requirements.write_text(
        'httpx>=0.28.1\nhttps://example.invalid/tool.whl\n')
    assert module.parse_requirements(str(requirements))[0] == ['httpx>=0.28.1']


def test_pypi_retry_uploads_only_missing_identical_release_files():
    release = {
        'files': {
            'a.whl': 'wheel-hash',
            'a.tar.gz': 'sdist-hash',
            'runtime-requirements.txt': 'lock-hash'
        }
    }
    remote = [{'filename': 'a.whl', 'digests': {'sha256': 'wheel-hash'}}]
    assert publish.pending_packages(release, remote) == ['a.tar.gz']
    remote.append({
        'filename': 'a.tar.gz',
        'digests': {
            'sha256': 'sdist-hash'
        }
    })
    assert publish.pending_packages(release, remote) == []


@pytest.mark.parametrize('filename,digest', [('a.whl', 'other-content'),
                                             ('other.whl', 'wheel-hash')])
def test_pypi_existing_different_content_is_never_skipped(filename, digest):
    with pytest.raises(ValueError):
        publish.pending_packages({'files': {
            'a.whl': 'wheel-hash'
        }}, [{
            'filename': filename,
            'digests': {
                'sha256': digest
            }
        }])


@pytest.mark.parametrize('message', [
    'unauthorized: authentication required', 'dial tcp: i/o timeout',
    'permission denied'
])
def test_acr_connection_and_auth_errors_are_not_absent_tags(
        monkeypatch, message):
    monkeypatch.setattr(
        publish.subprocess, 'run',
        lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', message))
    with pytest.raises(RuntimeError):
        publish.remote_image_exists('example.invalid/image:1.7.0',
                                    'sha256:tested')


def test_acr_cannot_overwrite_a_different_image(monkeypatch):
    monkeypatch.setattr(
        publish.subprocess, 'run',
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, '', ''))
    monkeypatch.setattr(publish.subprocess, 'check_output',
                        lambda *a, **kw: 'sha256:other\n')
    with pytest.raises(ValueError, match='do not overwrite'):
        publish.remote_image_exists('example.invalid/image:1.7.0',
                                    'sha256:tested')
