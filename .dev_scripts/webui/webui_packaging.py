# Copyright (c) ModelScope Contributors. All rights reserved.
"""WebUI package resources, shared by setuptools and release preparation."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    """Collect backend resources and frontend inputs without local caches."""
    selected = {
        'README.md', 'README_ZH.md', 'backend/.env.example',
        'backend/pyproject.toml', 'backend/uv.lock'
    }
    for file in (WEBUI / 'backend/app').rglob('*'):
        rel = file.relative_to(WEBUI)
        if (any(
                part.startswith('.')
                or part in {'__pycache__', 'node_modules'}
                for part in rel.parts) or file.suffix in {'.pyc', '.pyo'}):
            continue
        if file.is_file() or file.is_symlink():
            selected.add(rel.as_posix())
    selected.update(
        file.relative_to(WEBUI).as_posix()
        for file in frontend_validator().source_files(WEBUI / 'frontend'))
    build_manifest = WEBUI / 'frontend/build/webui-build.json'
    if include_build and build_manifest.is_file():
        build = json.loads(build_manifest.read_text(encoding='utf-8'))
        for rel in build['outputs']:
            if not rel.startswith('build/') or '..' in Path(rel).parts:
                raise RuntimeError('Invalid path in frontend build manifest')
            selected.add('frontend/' + rel)
        selected.add('frontend/build/webui-build.json')
    for rel in selected:
        if (Path(rel).is_absolute() or '\\' in rel
                or any(part in {'', '.', '..'} for part in rel.split('/'))):
            raise RuntimeError('Invalid WebUI resource path: ' + rel)
        file = WEBUI / rel
        if file.is_symlink() or WEBUI.resolve() not in file.resolve().parents:
            raise RuntimeError('Invalid WebUI resource path: ' + rel)
    return sorted(selected)


def validate_frontend():
    frontend_validator().validate_build(WEBUI / 'frontend')


def write_manifest(sdk_commit):
    validate_frontend()
    data = {
        'format': 1,
        'sdk_version': version(),
        'sdk_commit': sdk_commit,
        'prebuilt': True,
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
        if data.get('prebuilt', True) is not True or data.get('format') != 1 or data.get(
                'sdk_version') != version() or data.get('files') != actual:
            raise RuntimeError(
                'WebUI release inputs changed after preparation')
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(
            'WebUI release resources are missing or stale. Run '
            '`python .dev_scripts/webui/prepare_webui.py` before building wheel/sdist. '
            'Editable installs do not require a frontend build. Details: '
            + str(exc)) from exc


def copy_resources(destination):
    # A Git dependency is built before any frontend preparation. Preserve the
    # source so SDK-only installs need no Node toolchain; the UI can build it on
    # first use. An existing release manifest must still validate in full.
    prebuilt = (WEBUI / MANIFEST).is_file()
    if prebuilt:
        validate_release()
        manifest = json.loads((WEBUI / MANIFEST).read_text(encoding='utf-8'))
    else:
        manifest = {
            'format': 1,
            'sdk_version': version(),
            'sdk_commit': None,
            'prebuilt': False,
            'files': {rel: digest(WEBUI / rel)
                      for rel in resource_paths(include_build=False)},
        }
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    for rel in manifest['files']:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEBUI / rel, target)
    (destination / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
