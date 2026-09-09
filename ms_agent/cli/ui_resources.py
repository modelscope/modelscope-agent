# Copyright (c) ModelScope Contributors. All rights reserved.
"""Locate WebUI resources and prepare writable caches for installed wheels."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class UIError(RuntimeError):
    pass


def find_webui():
    package = Path(__file__).resolve().parents[1]
    bundled = package / 'webui'
    if (bundled / 'RESOURCE-MANIFEST.json').is_file():
        return bundled, True
    checkout = package.parent / 'webui'
    if (package.parent / 'setup.py').is_file() and (
            checkout / 'backend/app/launcher.py').is_file():
        return checkout, False
    raise UIError(
        'WebUI resources are missing. Reinstall the matching ms-agent wheel, '
        'or use a complete SDK source checkout.')


def load_common(webui):
    backend = str(webui / 'backend')
    if backend not in sys.path:
        sys.path.insert(0, backend)
    return importlib.import_module('app.launcher')


def cache_root():
    override = os.environ.get('MS_AGENT_WEBUI_CACHE')
    if override:
        return Path(override).expanduser().resolve()
    if os.name == 'nt':
        base = Path(
            os.environ.get('LOCALAPPDATA',
                           Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library/Caches'
    else:
        base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return base / 'ms-agent/webui'


def file_digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def verify_resources(root, manifest):
    if manifest.get('format') != 1 or not manifest.get('files'):
        raise UIError('Invalid WebUI resource manifest; reinstall ms-agent')
    for rel, digest in manifest['files'].items():
        file = root / rel
        if (file.is_symlink() or root.resolve() not in file.resolve().parents
                or not file.is_file() or file_digest(file) != digest):
            raise UIError(f'WebUI resource is missing or changed: {file}')


@contextmanager
def preparation_lock(root, name, timeout=300):
    """Serialize copies/installs; the OS releases the lock if a setup process dies."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ('.' + name + '.lock')).open('a+b') as stream:
        stream.seek(0)
        stream.write(b'0')
        stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt

                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise UIError(
                        'Timed out waiting for another WebUI setup process')
                time.sleep(0.1)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def materialize(bundled, manifest, destination):
    verify_resources(bundled, manifest)
    if destination.exists():
        verify_resources(destination, manifest)
        return
    temporary = Path(
        tempfile.mkdtemp(prefix='.webui-copy-', dir=destination.parent))
    try:
        for rel in list(manifest['files']) + ['RESOURCE-MANIFEST.json']:
            target = temporary / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled / rel, target)
        # Read-only installed files must not make the runtime cache read-only.
        for file in temporary.rglob('*'):
            file.chmod(0o700 if file.is_dir() else 0o600)
        verify_resources(temporary, manifest)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def node_stamp(frontend, node_version, *, production=True):
    return {
        'node':
        list(node_version),
        'platform':
        platform.system(),
        'architecture':
        platform.machine(),
        'lock':
        file_digest(frontend / 'pnpm-lock.yaml'),
        'package_manager':
        json.loads((frontend / 'package.json').read_text())['packageManager'],
        'production':
        production,
    }


def check_backend_dependencies():
    modules = [
        'anthropic', 'exa_py', 'fastapi', 'httpx',
        'loguru', 'mem0', 'pydantic_settings', 'socksio',
        'sse_starlette', 'uvicorn', 'watchfiles'
    ]
    missing = [
        module for module in modules
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        from ms_agent.version import __version__

        raise UIError(
            'Missing WebUI Python dependencies: ' + ', '.join(missing) +
            f'. Run `{sys.executable} -m pip install "ms-agent[webui]=={__version__}"`.'
        )


def prepare_installed(bundled, common, node_version, *, skip_install,
                      install_node, build_frontend=None):
    from ms_agent.version import __version__

    manifest_file = bundled / 'RESOURCE-MANIFEST.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest['sdk_version'] != __version__:
        raise UIError(
            'The WebUI resources do not match the installed SDK version')
    root = cache_root()
    name = __version__ + '-' + file_digest(manifest_file)[:16]
    destination = root / name
    check_backend_dependencies()
    with preparation_lock(root, name):
        materialize(bundled, manifest, destination)
        frontend = destination / 'frontend'
        prebuilt = manifest.get('prebuilt', True)
        if not isinstance(prebuilt, bool):
            raise UIError('Invalid WebUI build mode; reinstall ms-agent')
        if prebuilt:
            common.validate_build(frontend)
        marker = frontend / '.node-dependencies.json'
        expected = node_stamp(frontend, node_version, production=prebuilt)
        try:
            ready = json.loads(marker.read_text()) == expected and (
                frontend / 'node_modules').is_dir()
        except (OSError, ValueError):
            ready = False
        if not ready:
            if skip_install:
                raise UIError(
                    'WebUI Node dependencies need preparation. Run once without --skip-install.'
                )
            marker.unlink(missing_ok=True)
            install_node(frontend, production=prebuilt)
            marker.write_text(json.dumps(expected, sort_keys=True) + '\n')
        if not prebuilt:
            try:
                common.validate_build(frontend)
            except common.BuildError:
                if skip_install:
                    raise UIError(
                        'WebUI sources need a frontend build. Run once without --skip-install.'
                    ) from None
                if build_frontend is None:
                    raise UIError('No WebUI frontend builder is available')
                build_frontend(frontend)
                common.validate_build(frontend)
    return destination
