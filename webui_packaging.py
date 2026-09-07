# Copyright (c) ModelScope Contributors. All rights reserved.
"""Explicit WebUI release inputs, shared by setuptools and release preparation."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEBUI = ROOT / 'webui'
MANIFEST = 'RESOURCE-MANIFEST.json'


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def version():
    namespace = {}
    filename = ROOT / 'ms_agent/version.py'
    exec(
        compile(filename.read_text(encoding='utf-8'), str(filename), 'exec'),
        namespace)
    return namespace['__version__']


def frontend_validator():
    spec = importlib.util.spec_from_file_location(
        '_webui_frontend_build', WEBUI / 'backend/app/frontend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resource_paths(include_build=True):
    """Use the committed snapshot list, never an unrestricted directory copy."""
    source = json.loads((WEBUI / 'SOURCE.json').read_text(encoding='utf-8'))
    selected = {'SOURCE.json', 'README.md', 'README_ZH.md'}
    for rel in source['source_files']:
        if rel in {
                'backend/pyproject.toml', 'backend/uv.lock',
                'backend/.env.example'
        }:
            selected.add(rel)
        elif rel.startswith(('backend/app/', 'frontend/')):
            parts = Path(rel).parts
            ignored = {'node_modules', '__pycache__', 'tests', 'build'}
            if any(part.startswith('.') or part in ignored for part in parts):
                continue
            if not rel.endswith(('.pyc', '.pyo')):
                selected.add(rel)
    build_manifest = WEBUI / 'frontend/build/webui-build.json'
    if include_build and build_manifest.is_file():
        build = json.loads(build_manifest.read_text(encoding='utf-8'))
        for rel in build['outputs']:
            if not rel.startswith('build/') or '..' in Path(rel).parts:
                raise RuntimeError('Invalid path in frontend build manifest')
            selected.add('frontend/' + rel)
        selected.add('frontend/build/webui-build.json')
    for rel in selected:
        file = WEBUI / rel
        if file.is_symlink() or WEBUI.resolve() not in file.resolve().parents:
            raise RuntimeError('Invalid WebUI resource path: ' + rel)
    return sorted(selected)


def validate_frontend():
    frontend_validator().validate_build(WEBUI / 'frontend')
    build = json.loads(
        (WEBUI
         / 'frontend/build/webui-build.json').read_text(encoding='utf-8'))
    allowed = set(resource_paths(include_build=False))
    unexpected = {'frontend/' + rel for rel in build['inputs']} - allowed
    if unexpected:
        raise RuntimeError('Frontend inputs absent from SOURCE.json: '
                           + ', '.join(sorted(unexpected)))


def write_manifest(sdk_commit):
    validate_frontend()
    data = {
        'format': 1,
        'sdk_version': version(),
        'sdk_commit': sdk_commit,
        'files': {rel: digest(WEBUI / rel)
                  for rel in resource_paths()},
    }
    (WEBUI / MANIFEST).write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return data


def validate_release():
    try:
        validate_frontend()
        data = json.loads((WEBUI / MANIFEST).read_text(encoding='utf-8'))
        actual = {rel: digest(WEBUI / rel) for rel in resource_paths()}
        if data.get('format') != 1 or data.get(
                'sdk_version') != version() or data.get('files') != actual:
            raise RuntimeError(
                'WebUI release inputs changed after preparation')
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(
            'WebUI release resources are missing or stale. Run '
            '`python scripts/prepare_webui.py` before building wheel/sdist. '
            'Editable installs do not require a frontend build. Details: '
            + str(exc)) from exc


def copy_resources(destination):
    validate_release()
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    for rel in resource_paths() + [MANIFEST]:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEBUI / rel, target)
