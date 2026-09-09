#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preserve and restore the exact image tested by a release workflow."""
import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import publish_webui_release as publish


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def find_artifact(name):
    """Find this run's original artifact; API failures must not trigger rebuilds."""
    repository = os.environ['GITHUB_REPOSITORY']
    run = os.environ['GITHUB_RUN_ID']
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository) or not run.isdigit():
        raise ValueError('Invalid workflow identity')
    url = (f'https://api.github.com/repos/{repository}/actions/runs/{run}'
           '/artifacts?per_page=100&name=' + urllib.parse.quote(name))
    request = urllib.request.Request(url, headers={
        'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        matches = [a for a in json.load(response)['artifacts'] if a['name'] == name]
    if len(matches) > 1 or any(a['expired'] for a in matches):
        raise ValueError('Original artifact is expired or ambiguous; do not rebuild a published release')
    return str(matches[0]['id']) if matches else ''


def save_image(directory, inputs):
    release = publish.load_inputs(inputs)
    record = directory / 'image.json'
    image = publish.checked_image(release, record)
    archive = directory / 'image.tar.gz'
    # Save by ID: a shared daemon's mutable tag is never the transfer identity.
    with gzip.open(archive, 'wb', compresslevel=1) as output:
        proc = subprocess.Popen(['docker', 'save', image['image_id']], stdout=subprocess.PIPE)
        try:
            shutil.copyfileobj(proc.stdout, output)
        finally:
            proc.stdout.close()
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, proc.args)
    image['archive_sha256'] = digest(archive)
    record.write_text(json.dumps(image, indent=2) + '\n')


def restore_image(directory, inputs):
    release = publish.load_inputs(inputs)
    record = directory / 'image.json'
    image = publish.image_metadata(release, record)
    archive = directory / 'image.tar.gz'
    if image.get('archive_sha256') != digest(archive):
        raise ValueError('Image archive checksum mismatch; restore the original artifact')
    with gzip.open(archive, 'rb') as source:
        proc = subprocess.Popen(['docker', 'load'], stdin=subprocess.PIPE)
        try:
            shutil.copyfileobj(source, proc.stdin)
        finally:
            proc.stdin.close()
            code = proc.wait()
        if code:
            raise subprocess.CalledProcessError(code, proc.args)
    return publish.checked_image(release, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['find', 'save', 'restore'])
    parser.add_argument('--name')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--local-tag')
    args = parser.parse_args()
    if args.action == 'find':
        if not args.name:
            parser.error('find requires --name')
        artifact_id = find_artifact(args.name)
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write('artifact_id=' + artifact_id + '\n')
    else:
        if not args.directory or not args.inputs:
            parser.error('save/restore require --directory and --inputs')
        image = (save_image if args.action == 'save' else restore_image)(args.directory, args.inputs)
        if args.action == 'restore' and args.local_tag:
            subprocess.run(['docker', 'tag', image['image_id'], args.local_tag], check=True)


if __name__ == '__main__':
    main()
