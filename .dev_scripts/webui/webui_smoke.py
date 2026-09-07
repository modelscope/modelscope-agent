#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline HTTP checks for a WebUI with an isolated MS_AGENT_HOME."""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import quote

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(base, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={'Content-Type': 'application/json'})
    return OPENER.open(req, timeout=15)


def api(base, path, body=None, method=None):
    with request(base, path, body, method) as response:
        result = json.load(response)
    assert result['code'] == 0, result
    return result['data']


def ready(base, timeout=120):
    deadline = time.monotonic() + timeout
    while True:
        try:
            with request(base, '/api/health') as response:
                assert json.load(response) == {'status': 'ok'}
            with request(base, '/') as response:
                html = response.read().decode()
            match = re.search(r'href="(/assets/antd\.[^"/]+\.css)"', html)
            assert match, 'SSR page has no generated Ant Design stylesheet'
            with request(base, match[1]) as response:
                assert 'text/css' in response.headers['Content-Type']
                css = response.read()
                assert len(css) > 1000 and b'.ant-' in css
                assert b'<!DOCTYPE html' not in css
            return match[1]
        except (OSError, AssertionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def create(base):
    """Create only a uniquely named test project; never change existing data."""
    name = 'webui-smoke-' + uuid.uuid4().hex[:10]
    project = api(base, '/api/projects', {
        'name': name,
        'memory_enabled': False
    })
    project_id = project['id']
    session = api(base, '/api/sessions', {
        'title': name,
        'project_id': project_id
    })
    state = {'project': project_id, 'session': session['id'], 'name': name}
    scope = 'project:' + project_id
    content = f'---\nname: {name}\ndescription: Offline release check\n---\nTest only.\n'
    skill = api(
        base, '/api/skills', {
            'name':
            name,
            'scope':
            scope,
            'kind':
            'bundle',
            'content':
            json.dumps({
                'format': 'webui.skill.bundle.v1',
                'files': [{
                    'path': 'SKILL.md',
                    'content': content
                }]
            })
        })
    state['skill'] = skill['id']
    # Activate a real native watcher, so stop/restart also exercises its cleanup.
    assert any(s['id'] == state['skill']
               for s in api(base, '/api/skills?scope='
                            + quote(scope, safe='')))
    assert api(base, '/api/projects/' + project_id,
               {'description': 'Offline release check'},
               'PATCH')['name'] == name
    api(base, '/api/sessions/' + session['id'], {'title': name + '-saved'},
        'PATCH')
    workspace = '/api/projects/' + project_id + '/workspace/files'
    api(base, workspace, {'path': 'release-smoke.txt', 'content': name})
    assert any(p['id'] == project_id for p in api(base, '/api/projects'))
    assert any(s['id'] == session['id'] for s in api(base, '/api/sessions'))
    # No model request: attaching to an idle session must finish with an SSE
    # done frame, with the LF framing consumed by the browser client.
    with request(base, '/api/chat/attach',
                 {'session_id': session['id']}) as response:
        assert 'text/event-stream' in response.headers['Content-Type']
        stream = response.read()
    frames = [
        json.loads(line[6:]) for line in stream.decode().splitlines()
        if line.startswith('data: ')
    ]
    assert b'\n\n' in stream and b'\r\n' not in stream, stream
    assert any(frame.get('type') == 'done' for frame in frames), frames
    verify(base, state)
    return state


def verify(base, state):
    assert api(base, '/api/skills/'
               + quote(state['skill'], safe=''))['name'] == state['name']
    assert api(base,
               '/api/projects/' + state['project'])['name'] == state['name']
    assert api(base, '/api/sessions/' + state['session'])['title'] == (
        state['name'] + '-saved')
    file = '/api/projects/' + state[
        'project'] + '/workspace/files/release-smoke.txt'
    assert api(base, file)['content'] == state['name']


def cleanup(base, state):
    api(base, '/api/skills/' + quote(state['skill'], safe=''), method='DELETE')
    file = '/api/projects/' + state[
        'project'] + '/workspace/files/release-smoke.txt'
    api(base, file, method='DELETE')
    api(base, '/api/sessions/' + state['session'], method='DELETE')
    api(base, '/api/projects/' + state['project'], method='DELETE')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument(
        '--state',
        type=Path,
        help='Keep created data and record IDs for a restart check')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify persisted IDs from --state, then clean up')
    args = parser.parse_args()
    base = args.url.rstrip('/')
    css = ready(base)
    if args.verify:
        if not args.state:
            parser.error('--verify requires --state')
        state = json.loads(args.state.read_text())
        verify(base, state)
        cleanup(base, state)
    else:
        state = create(base)
        if args.state:
            args.state.write_text(json.dumps(state) + '\n')
        else:
            cleanup(base, state)
    print(
        json.dumps({
            'http': 'passed',
            'css': css,
            'crud_sse': 'passed',
            'restart': args.verify
        }))


if __name__ == '__main__':
    main()
