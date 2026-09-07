#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Publish verified artifacts; retries must reuse the original workflow artifacts."""
import argparse
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

REGISTRY_IMAGE = 'mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent'


def load_inputs(directory):
    release = json.loads((directory / 'release.json').read_text())
    for name, digest in release['files'].items():
        file = directory / name
        if file.parent != directory or hashlib.sha256(
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


def checked_image(release, image_record):
    image = json.loads(image_record.read_text())
    for key in ('version', 'sdk_commit', 'files'):
        if image[key] != release[key]:
            raise ValueError(
                'Validated image belongs to different release inputs')
    if image.get('smoke') != 'passed':
        raise ValueError('Image has no successful container check')
    actual = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', image['image']],
        text=True).strip()
    if actual != image['image_id']:
        raise ValueError('Loaded image differs from the validated image')
    return image


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pypi', 'check-acr', 'acr'])
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--image-record', type=Path)
    parser.add_argument('--image-tag')
    args = parser.parse_args()
    release = load_inputs(args.inputs)
    if args.action == 'pypi':
        publish_pypi(args.inputs, release)
        return
    if not args.image_record or not args.image_tag:
        parser.error('ACR actions require --image-record and --image-tag')
    allowed = {release['version'], 'dev-' + release['sdk_commit'][:12]}
    if args.image_tag not in allowed:
        parser.error(
            'Use the SDK version or dev-<SDK SHA first 12 characters>; latest is not published'
        )
    image = checked_image(release, args.image_record)
    target = REGISTRY_IMAGE + ':' + args.image_tag
    exists = remote_image_exists(target, image['image_id'])
    if args.action == 'acr':
        if not exists:
            subprocess.run(['docker', 'tag', image['image'], target],
                           check=True)
            subprocess.run(['docker', 'push', target], check=True)
        # Pull back from ACR and verify the image identity after every push.
        assert remote_image_exists(target, image['image_id'])
        command = [
            'docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}',
            target
        ]
        print(subprocess.check_output(command, text=True))
    else:
        print('ACR tag can be published'
              if not exists else 'ACR already contains this exact image')


if __name__ == '__main__':
    main()
