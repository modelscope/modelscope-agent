#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check a built image locally without contacting a registry or a model."""
import argparse
import json
import subprocess
import uuid
import webui_smoke as smoke
from pathlib import Path


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='ms-agent-webui:validated')
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logs', type=Path, required=True)
    parser.add_argument('--resource-label', default='')
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    release = json.loads((args.inputs / 'release.json').read_text())
    image = json.loads(docker('image', 'inspect', args.image))[0]
    wheel_sha = next(value for name, value in release['files'].items()
                     if name.endswith('.whl'))
    labels = image['Config']['Labels']
    assert image['Architecture'] == 'amd64'
    assert labels['org.opencontainers.image.revision'] == release['sdk_commit']
    assert labels['org.opencontainers.image.version'] == release['version']
    assert labels['com.modelscope.ms-agent.wheel-sha256'] == wheel_sha
    name = 'ms-agent-smoke-' + uuid.uuid4().hex[:12]
    volume = name + '-data'
    label_args = ['--label', args.resource_label] if args.resource_label else []
    docker('volume', 'create', *label_args, volume)
    state = None
    try:
        for scenario in ('normal', 'restart', 'backend', 'frontend'):
            docker('run', *label_args, '--cpus', '2', '--memory', '4g',
                   '--detach', '--name', name, '--publish',
                   '127.0.0.1::8000', '--mount',
                   'type=volume,source=' + volume + ',target=/data',
                   args.image)
            try:
                mapping = docker('port', name, '8000/tcp')
                base = 'http://' + mapping
                smoke.ready(base)
                installed = json.loads(
                    docker(
                        'exec', name, 'python', '-c',
                        'import ms_agent.agent_hub; '
                        'import bs4, lxml, pyarrow, seaborn, sklearn; '
                        'from ms_agent.cli.ui_resources import find_webui; '
                        'p, installed = find_webui(); assert installed; '
                        'print((p / "RESOURCE-MANIFEST.json").read_text())'))
                assert installed['sdk_commit'] == release['sdk_commit']
                assert installed['sdk_version'] == release['version']
                if scenario == 'normal':
                    state = smoke.create(base)
                elif scenario == 'restart':
                    smoke.verify(base, state)
                    smoke.cleanup(base, state)
                if scenario in ('backend', 'frontend'):
                    # The marker is an argument, not interpolated shell code.
                    code = '''import os, signal, sys
from pathlib import Path
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args = (proc / 'cmdline').read_bytes().split(b'\\0')
        backend = len(args) > 2 and args[1] == b'-c' and args[2].startswith(b'from app.launcher import _run_api;')
        frontend = len(args) > 1 and args[0].endswith(b'node') and args[1].endswith(b'/server.js')
        if (backend and sys.argv[1] == 'backend') or (frontend and sys.argv[1] == 'frontend'):
            os.kill(int(proc.name), signal.SIGKILL)
            break
    except (FileNotFoundError, ProcessLookupError):
        continue
else:
    raise SystemExit('Expected service process was not found')
'''
                    docker('exec', name, 'python', '-c', code, scenario)
                    # Bound the failure wait, so a broken supervisor fails CI.
                    exit_code = subprocess.check_output(
                        ['docker', 'wait', name], text=True,
                        timeout=30).strip()
                    assert exit_code != '0'
                else:
                    docker('stop', '--time', '20', name)
                    assert docker('inspect', '--format', '{{.State.ExitCode}}',
                                  name) == '0'
            finally:
                (args.logs /
                 (scenario + '.log')).write_text(docker('logs', name))
                subprocess.run(['docker', 'rm', '--force', name], check=False)
        record = {
            **release, 'image_id': image['Id'],
            'image': args.image,
            'wheel_sha256': wheel_sha,
            'smoke': 'passed'
        }
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
    finally:
        subprocess.run(['docker', 'rm', '--force', name],
                       check=False,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        subprocess.run(['docker', 'volume', 'rm', volume], check=True)


if __name__ == '__main__':
    main()
