#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Publish verified artifacts; retries must reuse the original workflow artifacts."""
import argparse
import base64
import contextlib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REGISTRY_IMAGE = 'mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent'


def load_inputs(directory):
    release = json.loads((directory / 'release.json').read_text())
    for name, digest in release['files'].items():
        file = directory / name
        if file.parent != directory or file.is_symlink() or hashlib.sha256(
                file.read_bytes()).hexdigest() != digest:
            raise ValueError('Release input changed: ' + name)
    return release


def pending_packages(release, remote):
    """Refuse a different file with the same PyPI version, even on a retry."""
    published = {
        file['filename']: file['digests']['sha256']
        for file in remote
    }
    expected = {
        name: digest
        for name, digest in release['files'].items()
        if name.endswith(('.whl', '.tar.gz'))
    }
    unexpected = set(published) - set(expected)
    if unexpected:
        raise ValueError('PyPI already has other files for this version: '
                         + str(sorted(unexpected)))
    pending = []
    for name, digest in expected.items():
        if name not in published:
            pending.append(name)
        elif published[name] != digest:
            raise ValueError(
                'PyPI content differs; use the original artifacts or a new version: '
                + name)
    return pending


def publish_pypi(directory, release):
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:rc\d+)?', release['version']):
        raise ValueError('PyPI publication requires a release or RC version')
    url = 'https://pypi.org/pypi/ms-agent/' + release['version'] + '/json'
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            remote = json.load(response)['urls']
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        remote = []
    pending = pending_packages(release, remote)
    if pending:
        # Twine credentials are environment variables, never command arguments.
        command = ['twine', 'upload', '--non-interactive']
        command.extend(str(directory / name) for name in pending)
        subprocess.run(command, check=True)
    else:
        print('PyPI already contains these exact wheel/sdist bytes')


def image_metadata(release, image_record):
    image = json.loads(image_record.read_text())
    for key in ('version', 'sdk_commit', 'files'):
        if image[key] != release[key]:
            raise ValueError(
                'Validated image belongs to different release inputs')
    if image.get('smoke') != 'passed':
        raise ValueError('Image has no successful container check')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', image.get('image_id', '')):
        raise ValueError('Invalid validated image ID')
    wheel_sha = next(value for name, value in release['files'].items()
                     if name.endswith('.whl'))
    if image.get('wheel_sha256') != wheel_sha:
        raise ValueError('Image wheel checksum differs from release inputs')
    return image


def checked_image(release, image_record):
    image = image_metadata(release, image_record)
    actual = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', image['image_id']],
        text=True).strip()
    if actual != image['image_id']:
        raise ValueError('Loaded image differs from the validated image')
    return image


def mask(value):
    if value and os.environ.get('GITHUB_ACTIONS') == 'true':
        value = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print('::add-mask::' + value, flush=True)


@contextlib.contextmanager
def registry_auth(source):
    """Scope saved runner credentials to one registry and remove the job copy."""
    registry = REGISTRY_IMAGE.split('/')[0]
    saved = json.loads(source.read_text())
    auth = saved.get('auths', {}).get(registry, {})
    helper = saved.get('credHelpers', {}).get(registry) or saved.get('credsStore')
    if not helper and not any(
            auth.get(key) for key in ('auth', 'identitytoken', 'registrytoken')):
        raise ValueError('Runner has no saved credentials for the image registry')
    for value in auth.values():
        if isinstance(value, str):
            mask(value)
    if auth.get('auth'):
        decoded = base64.b64decode(auth['auth'], validate=True).decode()
        mask(decoded)
        for value in decoded.split(':', 1):
            mask(value)
    config = {'auths': {registry: auth}}
    if helper:
        config['credHelpers'] = {registry: helper}
    # The workflow's final cleanup also removes this copy after cancellation.
    with tempfile.TemporaryDirectory(
            prefix='ms-agent-acr-',
            dir=os.environ.get('IMAGE_JOB_DIR')) as directory:
        target = Path(directory) / 'config.json'
        target.touch(mode=0o600)
        target.write_text(json.dumps(config))
        previous = os.environ.get('DOCKER_CONFIG')
        os.environ['DOCKER_CONFIG'] = directory
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('DOCKER_CONFIG', None)
            else:
                os.environ['DOCKER_CONFIG'] = previous


def remote_image_exists(target, expected_id):
    result = subprocess.run(['docker', 'pull', target],
                            capture_output=True,
                            text=True)
    if result.returncode == 0:
        actual = subprocess.check_output(
            ['docker', 'image', 'inspect', '--format', '{{.Id}}', target],
            text=True).strip()
        if actual != expected_id:
            raise ValueError(
                'ACR tag already contains a different image; do not overwrite: '
                + target)
        return True
    # Auth/network failures are never interpreted as an absent version.
    if re.search(r'manifest[ _]unknown', result.stderr, re.IGNORECASE):
        return False
    raise RuntimeError('Cannot check the ACR tag: ' + result.stderr.strip())


def publish_acr(release, image_record, image_tag, *, push):
    allowed = {release['version'], 'dev-' + release['sdk_commit'][:12]}
    if image_tag not in allowed:
        raise ValueError('Use the SDK version or dev-<SDK SHA first 12 characters>')
    image = checked_image(release, image_record)
    target = REGISTRY_IMAGE + ':' + image_tag
    exists = remote_image_exists(target, image['image_id'])
    if not push:
        print('ACR already contains this exact image' if exists else 'ACR tag can be published')
        return
    if not exists:
        subprocess.run(['docker', 'tag', image['image_id'], target], check=True)
        subprocess.run(['docker', 'push', target], check=True)
    if not remote_image_exists(target, image['image_id']):
        raise RuntimeError('ACR image was not found after pushing')
    command = ['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', target]
    digests = json.loads(subprocess.check_output(command, text=True))
    matching = [value for value in digests if re.fullmatch(
        re.escape(REGISTRY_IMAGE) + r'@sha256:[0-9a-f]{64}', value)]
    if len(matching) != 1:
        raise ValueError('Published image does not have a unique registry digest')
    receipt = {'version': release['version'], 'sdk_commit': release['sdk_commit'],
               'wheel_sha256': image['wheel_sha256'], 'image_id': image['image_id'],
               'image': target, 'image_digest': matching[0]}
    print('Validated image published: ' + matching[0])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pypi', 'check-acr', 'acr'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--image-record', type=Path)
    parser.add_argument('--image-tag')
    parser.add_argument('--auth-file', type=Path)
    parser.add_argument('--receipt', type=Path,
                        help='Write the verified registry identity after pushing')
    args = parser.parse_args()
    release = load_inputs(args.inputs)
    if args.action == 'pypi':
        publish_pypi(args.inputs, release)
        return
    if not args.image_record or not args.image_tag:
        parser.error('ACR actions require --image-record and --image-tag')
    auth = registry_auth(args.auth_file) if args.auth_file else contextlib.nullcontext()
    with auth:
        receipt = publish_acr(release, args.image_record, args.image_tag,
                              push=args.action == 'acr')
        if receipt and args.receipt:
            args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
