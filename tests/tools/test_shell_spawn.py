"""Shell spawn isolation: no TTY steal, no interactive SSH login."""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from omegaconf import OmegaConf

from ms_agent.tools.code.local_code_executor import LocalCodeExecutionTool
from ms_agent.tools.code.shell_spawn import (
    interactive_login_error,
    isolated_subprocess_kwargs,
)
from ms_agent.utils.task_manager import TaskManager


@pytest.mark.parametrize(
    'command,reject',
    [
        ('ssh ecs-user@47.253.113.69', True),
        ('ssh -p 22 ecs-user@host', True),
        ('ssh -i ~/.ssh/id_rsa user@host', True),
        ('FOO=1 ssh user@host', True),
        ('sftp user@host', True),
        ("ssh user@host 'uname -a'", False),
        ('ssh user@host uptime', False),
        ('ssh -o BatchMode=yes user@host echo hi', False),
        ('ssh -N -L 8080:localhost:80 host', False),
        ('ssh -h', False),
        ('ssh --help', False),
        ('ssh -V', False),
        ('ssh', False),
        ('ssh -Q cipher', False),
        ('ssh -G host', False),
        ('echo hello', False),
        ('scp file user@host:', False),
        ('sftp -b batch.txt user@host', False),
    ],
)
def test_interactive_login_detection(command, reject):
    err = interactive_login_error(command)
    if reject:
        assert err
        assert 'ssh user@host' in err
    else:
        assert err is None


def test_isolated_kwargs_detach_stdin():
    kw = isolated_subprocess_kwargs()
    assert kw['stdin'] == asyncio.subprocess.DEVNULL
    assert kw['start_new_session'] is True
    assert kw['stdout'] == asyncio.subprocess.PIPE


def _tool(tmp_path):
    cfg = OmegaConf.create({
        'output_dir': str(tmp_path),
        'tools': {
            'code_executor': {
                'mcp': False,
                'implementation': 'python_env',
                'include': ['shell_executor'],
            }
        },
    })
    return LocalCodeExecutionTool(cfg)


@pytest.mark.asyncio
async def test_bare_ssh_rejected_without_spawn(tmp_path, monkeypatch):
    tool = _tool(tmp_path)

    async def boom(*_a, **_k):
        raise AssertionError('must not spawn')

    monkeypatch.setattr(tool, '_spawn_shell', boom)
    raw = await tool.shell_executor('ssh ecs-user@1.2.3.4')
    data = json.loads(raw)
    assert data['success'] is False
    assert 'remote command' in data['error']


@pytest.mark.asyncio
async def test_spawn_passes_isolation_kwargs(tmp_path, monkeypatch):
    tool = _tool(tmp_path)
    captured = {}

    class Proc:
        pid = 1
        returncode = 0

        async def communicate(self):
            return b'ok\n', b''

    async def fake_shell(cmd, **kwargs):
        captured.update(kwargs)
        return Proc()

    monkeypatch.setattr(asyncio, 'create_subprocess_shell', fake_shell)
    raw = await tool.shell_executor('echo ok', timeout=5)
    data = json.loads(raw)
    assert data.get('success') is True
    assert captured.get('stdin') == asyncio.subprocess.DEVNULL
    assert captured.get('start_new_session') is True


@pytest.mark.asyncio
async def test_timeout_kills_process_group(tmp_path):
    tool = _tool(tmp_path)
    pidfile = tmp_path / 'sleep.pid'
    # sh is the session leader; sleep is a grandchild. Process.kill() on sh
    # used to leak sleep; killpg must reap both.
    cmd = f'sh -c "echo $$ > {pidfile}; exec sleep 60"'
    raw = await tool.shell_executor(cmd, timeout=1)
    data = json.loads(raw)
    assert data['success'] is False
    assert 'timed out' in data['error']
    deadline = time.time() + 3
    while time.time() < deadline:
        if pidfile.exists():
            pid = int(pidfile.read_text().strip() or '0')
            if pid and not _pid_alive(pid):
                return
        await asyncio.sleep(0.05)
    pid = int(pidfile.read_text().strip()) if pidfile.exists() else None
    pytest.fail(f'sleep pid {pid} still alive after timeout kill')


@pytest.mark.asyncio
async def test_timeout_auto_backgrounds_when_task_manager(tmp_path):
    tool = _tool(tmp_path)
    tm = TaskManager()
    tool.set_task_manager(tm)
    raw = await tool.shell_executor('sleep 2', timeout=1)
    data = json.loads(raw)
    assert data['status'] == 'async_launched'
    assert data['auto_backgrounded'] is True
    task = tm.get_task(data['task_id'])
    assert task is not None
    assert task.status == 'running'
    tm.kill(data['task_id'])
    await asyncio.sleep(0.2)
    assert not _pid_alive(task.proc.pid)


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
