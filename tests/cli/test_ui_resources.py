import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ms_agent.cli import ui_resources as resources
from ms_agent.version import __version__


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'installed'
    files = {
        'backend/app/main.py': '# backend',
        'frontend/server.js': '// server',
        'frontend/package.json': json.dumps({'packageManager': 'pnpm@10.17.1'}),
        'frontend/pnpm-lock.yaml': 'lockfileVersion: 9',
    }
    manifest = {'format': 1, 'sdk_version': __version__, 'files': {}}
    for rel, data in files.items():
        file = root / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(data)
        manifest['files'][rel] = hashlib.sha256(file.read_bytes()).hexdigest()
    (root / 'RESOURCE-MANIFEST.json').write_text(json.dumps(manifest))
    return root, manifest


def test_materialization_uses_only_listed_files(bundle, tmp_path):
    root, manifest = bundle
    (root / '.env').write_text('EXAMPLE_SECRET=do-not-copy')
    destination = tmp_path / 'cache'
    resources.materialize(root, manifest, destination)
    assert not (destination / '.env').exists()
    assert (destination / 'frontend/server.js').read_text() == '// server'
    assert not (root / 'frontend/node_modules').exists()


def test_cache_copy_is_writable_when_installed_file_is_readonly(bundle, tmp_path):
    root, manifest = bundle
    (root / 'frontend/server.js').chmod(0o444)
    destination = tmp_path / 'cache'
    resources.materialize(root, manifest, destination)
    assert (destination / 'frontend/server.js').stat().st_mode & 0o200
    assert not (root / 'frontend/server.js').stat().st_mode & 0o200


def test_reusing_resources_preserves_prepared_node_dependencies(bundle, tmp_path):
    root, manifest = bundle
    destination = tmp_path / 'cache'
    resources.materialize(root, manifest, destination)
    dependency = destination / 'frontend/node_modules/prepared.txt'
    dependency.parent.mkdir()
    dependency.write_text('already prepared')
    resources.materialize(root, manifest, destination)
    assert dependency.read_text() == 'already prepared'


def test_modified_resources_fail_before_use(bundle, tmp_path):
    root, manifest = bundle
    (root / 'frontend/server.js').write_text('changed')
    with pytest.raises(resources.UIError, match='changed'):
        resources.materialize(root, manifest, tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


def test_resource_path_cannot_escape_root(bundle, tmp_path):
    root, manifest = bundle
    sibling = tmp_path / 'outside'
    sibling.write_text('outside')
    manifest['files']['../outside'] = resources.file_digest(sibling)
    with pytest.raises(resources.UIError):
        resources.materialize(root, manifest, tmp_path / 'cache')


def test_parallel_starts_prepare_node_dependencies_once(bundle, tmp_path, monkeypatch):
    root, _manifest = bundle
    monkeypatch.setenv('MS_AGENT_WEBUI_CACHE', str(tmp_path / 'runtime'))
    monkeypatch.setattr(resources, 'check_backend_dependencies', lambda: None)
    common = SimpleNamespace(validate_build=lambda directory: None)
    calls = []

    def install(frontend, production):
        assert production
        calls.append(frontend)
        time.sleep(0.05)
        (frontend / 'node_modules').mkdir()

    def prepare():
        return resources.prepare_installed(root, common, (22, 23, 2),
                                           skip_install=False, install_node=install)

    with ThreadPoolExecutor(max_workers=2) as pool:
        prepared = list(pool.map(lambda _: prepare(), range(2)))
    assert prepared[0] == prepared[1]
    assert len(calls) == 1
    reused = resources.prepare_installed(root, common, (22, 23, 2), skip_install=True,
                                         install_node=lambda *a, **kw: pytest.fail('must not install'))
    assert reused == prepared[0]
    with pytest.raises(resources.UIError, match='need preparation'):
        resources.prepare_installed(root, common, (22, 24, 0), skip_install=True,
                                    install_node=lambda *a, **kw: pytest.fail('must not install'))


def test_missing_backend_dependencies_names_matching_extra(monkeypatch):
    monkeypatch.setattr(resources.importlib.util, 'find_spec', lambda _: None)
    with pytest.raises(resources.UIError, match=r'ms-agent\[webui\]=='):
        resources.check_backend_dependencies()
