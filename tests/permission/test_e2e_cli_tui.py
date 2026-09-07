"""End-to-end CLI/TUI permission journeys with a real LLM.

These tests spawn the real ``ms-agent`` process, talk to DashScope, and drive
the one-layer permission prompt over a pipe (not a PTY, so the non-TTY
fallback is used). Missing API keys fail the run — they do not skip.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'e2e_agent.yaml'
_ENV_CANDIDATES = [
    _REPO / '.env',
    Path('/Users/luyan/workspace/modelscope-agent/.env'),
]
_MARKER = 'permission-e2e-ok'
_SECOND = 'permission-e2e-2'


def _load_e2e_env() -> Path:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    for path in _ENV_CANDIDATES:
        if path.is_file():
            if load_dotenv is not None:
                load_dotenv(path, override=False)
            return path
    pytest.fail(
        'E2E 需要真实 API Key。请在仓库根目录 .env 里设置 DASHSCOPE_API_KEY '
        '后重跑（或把 key 发给这次会话）。当前找不到 .env。')


def _require_api_key() -> str:
    env_file = _load_e2e_env()
    key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    if not key:
        pytest.fail(
            f'E2E 需要真实 API Key。{env_file} 已加载但没有 DASHSCOPE_API_KEY。'
            '请把 DashScope key 写进 .env 后重跑，或把 key 发给这次会话。')
    return key


class _Proc:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.buf = ''
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        stdout = self.proc.stdout
        assert stdout is not None
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                self._q.put(None)
                return
            self._q.put(chunk)

    def _drain(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                chunk = self._q.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if chunk is None:
                return
            self.buf += chunk.decode('utf-8', errors='replace')

    def wait_for(self, needle: str, timeout: float = 120.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.buf:
                return self.buf
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._drain(min(0.5, remaining))
        raise AssertionError(
            f'timed out waiting for {needle!r}\n--- transcript ---\n{self.buf}'
        )

    def send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write((line + '\n').encode('utf-8'))
        self.proc.stdin.flush()

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


def _spawn(
    tmp_path: Path,
    *,
    tui: bool,
    permission_mode: str = 'interactive',
    extra: list[str] | None = None,
    config: Path | None = None,
) -> _Proc:
    _require_api_key()
    env_file = next(p for p in _ENV_CANDIDATES if p.is_file())
    work = tmp_path / 'work'
    home = tmp_path / 'home'
    work.mkdir()
    home.mkdir()
    env = os.environ.copy()
    env['HOME'] = str(home)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONPATH'] = str(_REPO) + os.pathsep + env.get('PYTHONPATH', '')
    env['TERM'] = 'dumb'
    env['COLUMNS'] = '80'
    cfg = str(config or _FIXTURE)
    cmd = [sys.executable, '-m', 'ms_agent.cli.cli']
    if tui:
        cmd += [
            'tui', '--config', cfg, '--work-dir', str(work),
            '--env', str(env_file), '--permission_mode', permission_mode,
        ]
    else:
        cmd += [
            'run', '--config', cfg, '--output_dir', str(work),
            '--env', str(env_file), '--permission_mode', permission_mode,
        ]
    if extra:
        cmd.extend(extra)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(work),
        env=env,
        bufsize=0,
    )
    wrapped = _Proc(proc)
    wrapped.work = work  # type: ignore[attr-defined]
    return wrapped


def _memory_path(work: Path) -> Path:
    return work / '.ms_agent' / 'permission_memory.json'


def _assert_one_layer(transcript: str) -> None:
    assert 'Always allow' not in transcript
    assert 'Pattern [' not in transcript
    assert "Choice [y/s/a/e/n]" not in transcript
    assert "don't ask again" in transcript


def test_e2e_cli_persist_then_crud(tmp_path):
    sess = _spawn(tmp_path, tui=False)
    work = sess.work
    try:
        sess.wait_for('>>>', timeout=60)
        sess.send(
            'Use shell_executor only. Run exactly: '
            f'echo {_MARKER}. Do not use python.')
        sess.wait_for("don't ask again", timeout=180)
        _assert_one_layer(sess.buf)
        sess.wait_for('choice:', timeout=10)
        sess.send('2')
        sess.wait_for(_MARKER, timeout=60)
        sess.wait_for('>>>', timeout=60)
        before = sess.buf
        sess.send(
            f'Use shell_executor only. Run exactly: echo {_SECOND}.')
        sess.wait_for(_SECOND, timeout=180)
        asked_again = sess.buf[len(before):].count('choice:')
        assert asked_again == 0, sess.buf[len(before):]
        assert _memory_path(work).is_file()
        sess.send('/permission list')
        sess.wait_for('echo *', timeout=30)
        sess.send('/quit')
    finally:
        sess.close()


def test_e2e_tui_prefix_edit_and_list_delete(tmp_path):
    sess = _spawn(tmp_path, tui=True)
    work = sess.work
    try:
        sess.wait_for('>>>', timeout=90)
        sess.send(
            'Use shell_executor only. Run exactly: '
            f'echo {_MARKER}. Do not use python.')
        sess.wait_for("don't ask again for echo *", timeout=180)
        _assert_one_layer(sess.buf)
        sess.wait_for('choice:', timeout=10)
        sess.send('2=echo permission-e2e*')
        sess.wait_for(_MARKER, timeout=60)
        sess.wait_for('>>>', timeout=60)
        sess.send('/permission list')
        sess.wait_for('permission-e2e*', timeout=30)
        listed = sess.buf
        rule_id = None
        for line in listed.splitlines():
            stripped = line.strip().lstrip('│').strip()
            if '[project/' in stripped or '[global/' in stripped:
                token = stripped.split()[0]
                if all(c in '0123456789abcdef' for c in token.lower()):
                    rule_id = token
                    break
        assert rule_id, listed
        sess.send(f'/permission delete {rule_id}')
        sess.wait_for('Deleted', timeout=30)
        sess.send(
            'Use shell_executor only. Run exactly: '
            f'echo {_MARKER}. Do not use python.')
        sess.wait_for('choice:', timeout=180)
        sess.send('3')
        sess.send('/quit')
    finally:
        sess.close()


def test_e2e_cli_delegate_real_llm(tmp_path):
    sess = _spawn(
        tmp_path,
        tui=False,
        permission_mode='delegate',
        config=Path(__file__).resolve().parent / 'fixtures' / 'e2e_delegate.yaml',
        extra=['--query',
               f'Use shell_executor only. Run exactly: echo {_MARKER}.'],
    )
    work = sess.work
    try:
        sess.wait_for(_MARKER, timeout=180)
        assert 'choice:' not in sess.buf
        assert not _memory_path(work).exists() or 'echo *' not in _memory_path(
            work).read_text(encoding='utf-8')
    finally:
        sess.close()
