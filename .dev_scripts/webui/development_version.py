#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Assign a reproducible development version inside an isolated package source."""
import argparse
import os
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
DEVELOPMENT = re.compile(RELEASE + r'\.dev([1-9]\d*)')


def development_version(source_version, build_id, base_version=''):
    if not re.fullmatch(RELEASE + r'(?:rc\d+)?', source_version):
        raise ValueError('Development builds require an SDK release version')
    base = base_version or source_version.split('rc', 1)[0]
    if not re.fullmatch(RELEASE, base):
        raise ValueError('Development base version must be X.Y.Z')
    if not re.fullmatch(r'[1-9]\d*', str(build_id)):
        raise ValueError('Build ID must be a positive integer without leading zeroes')
    return base + '.dev' + str(build_id)


def source_version(source):
    text = (source / 'ms_agent/version.py').read_text()
    match = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$", text)
    if not match:
        raise ValueError('Cannot locate the SDK version')
    return match.group(1)


def apply_version(source, expected):
    if source.resolve() == ROOT:
        raise ValueError('Use an isolated source copy; do not change the working checkout')
    if not DEVELOPMENT.fullmatch(expected):
        raise ValueError('Expected a development package version')
    previous = source_version(source)
    lock_path = source / 'webui/backend/uv.lock'
    lock = lock_path.read_text()
    sdk = [p for p in tomllib.loads(lock)['package'] if p['name'] == 'ms-agent']
    if len(sdk) != 1 or sdk[0]['version'] != previous:
        raise ValueError('SDK version and backend lock disagree before preparation')
    pattern = r'(\[\[package\]\]\nname = "ms-agent"\nversion = ")[^"]+("\n)'
    updated, count = re.subn(pattern, lambda m: m[1] + expected + m[2], lock)
    if count != 1:
        raise ValueError('Cannot update the editable SDK entry in the backend lock')
    version_path = source / 'ms_agent/version.py'
    version_path.write_text(re.sub(
        r"(?m)^__version__\s*=\s*['\"][^'\"]+['\"]\s*$",
        "__version__ = '" + expected + "'", version_path.read_text()))
    lock_path.write_text(updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-id', required=True)
    parser.add_argument('--base-version', default='')
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--github-output', action='store_true')
    args = parser.parse_args()
    version = development_version(source_version(args.source), args.build_id, args.base_version)
    if args.apply:
        apply_version(args.source, version)
    if args.github_output:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write('package_version=' + version + '\n')
    print('Development package version: ' + version)


if __name__ == '__main__':
    main()
