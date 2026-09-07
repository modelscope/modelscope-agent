# MS-Agent WebUI

The WebUI provides projects and sessions, streamed chat, model configuration,
skills, MCP tools, memory, and workspace files. It uses the same SDK and data
home as the CLI/TUI. [中文说明](README_ZH.md)

This checkout prepares the **1.7 release**. The wheel installation below applies
to packages built from this code; older 1.6 packages do not include this WebUI.
Until a 1.7 package is published, use the source checkout or a locally built wheel.

## Quick start from a source checkout

Prerequisites: Python **3.12+**, Node **22.22.0+**, pnpm **10.17.1**, and uv **0.5+**.
For pnpm, use `npm install --global pnpm@10.17.1` if it is not installed.

From the SDK repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ms-agent ui
```

The first run prepares `webui/backend/.venv`, installs locked frontend dependencies,
and runs `pnpm build`, including Ant Design CSS generation. Later starts reuse an
unchanged build. Open the single URL printed after the API, page and CSS checks
pass, normally **http://127.0.0.1:8000**. Configure a provider and model in
**Settings → Models**, then open a project/session and start a conversation.

Ctrl-C stops both services. On Windows, activate `.venv\Scripts\Activate.ps1` and
use `python` from a Python 3.12+ installation. The optional UTF-8 wrapper remains:

```powershell
.\webui\scripts\start-webui.ps1 --no-browser
```

## Install a wheel

Use the same Python/Node/pnpm prerequisites. A released wheel includes backend
and frontend source plus prebuilt SSR, client assets and CSS. No frontend build
or uv synchronization is needed on the user's machine.

```bash
# After the selected 1.7 RC has been published:
python -m pip install 'ms-agent[webui]==1.7.0rc0'
ms-agent ui

# Before publication, install the wheel produced from this checkout instead:
python -m pip install './dist/ms_agent-1.6.0-py3-none-any.whl[webui]'
ms-agent ui
```

Use the actual filename/version in `dist/`; the development version stays 1.6.0
until the release operator updates it. Ordinary `pip install ms-agent` downloads
the same WebUI resources; `[webui]` adds the required Python dependencies. `[all]`
also includes them. The WebUI itself requires Python 3.12+.

The first installed-wheel start installs **production Node dependencies** in a
user cache. Later starts reuse them. Nothing is installed into site-packages.
`MS_AGENT_WEBUI_CACHE` can select another writable cache root, including a fixed
location prepared when building a container. Otherwise the platform user cache
is used (`~/Library/Caches/ms-agent/webui` on macOS, `$XDG_CACHE_HOME/ms-agent/webui`
or `~/.cache/ms-agent/webui` on Linux, `%LOCALAPPDATA%\ms-agent\webui` on Windows).
Caches are separated by SDK version and resource content; Node version, platform
and lockfile changes invalidate the dependency cache.

## Options and development

`ms-agent ui` delegates to the same `app.launcher` as `uv run webui` from
`webui/backend/`. It runs the built SSR app, with a loopback API behind one
public port.

| Option | Behavior |
| --- | --- |
| `--host HOST` | Public bind address, default `127.0.0.1` |
| `--port PORT` | Exact public port; if omitted, choose a free port from 8000 |
| `--backend-port PORT` | Exact internal loopback port; otherwise use the next free port |
| `--no-browser` | Do not open a browser |
| `--skip-install` | Do not download/synchronize dependencies; still validate the build and CSS |
| `--production` | Compatibility alias for the default SSR mode |
| `--reload` | Currently rejected; use the development commands below |
| `--prepare-only` | Prepare dependencies/resources and exit; useful in image builds |
| `--startup-timeout SECONDS` | Startup deadline, default 120 seconds |

Explicit ports must be free, between 1 and 65535, and different. If the public
port is 65535, automatic API selection starts at 8000. The earlier default public
port was 7860; keep it with `ms-agent ui --port 7860`. A startup failure or the
unexpected exit of either service stops the whole stack and returns a failure.

For live frontend/backend development, use two terminals:

```bash
# From the SDK root, terminal 1:
cd webui/backend
uv sync --locked
uv run --no-sync dev

# From the SDK root, terminal 2:
cd webui/frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open **http://localhost:5173**; the development server proxies the API on 8000.
For the built app, run `pnpm build` in `webui/frontend/`, then
`uv run --no-sync webui --no-open` in `webui/backend/`. `pnpm build` generates CSS,
SSR, client output and `build/webui-build.json`; do not replace it with a raw
React Router command or combine an older SSR bundle with newly generated CSS.
With `--skip-install`, a stale source build can still be rebuilt using an already
installed pnpm/toolchain. Installed wheels always use their prebuilt frontend.

## Configuration and features

Data defaults to **`~/.ms_agent`**, shared with CLI/TUI. Set `MS_AGENT_HOME` to an
absolute directory to isolate a test or deployment. It contains settings,
projects, sessions and managed skills. Back it up before upgrading.

Source dotenv precedence is: process environment > `webui/backend/.env` >
`webui/.env` > SDK-root `.env`. See [the example](backend/.env.example).
Installed wheels read process environment and SDK settings, and do not discover
`.env` files around site-packages, the cache or the working directory.

- **Models:** configure provider credentials and models in Settings; select a
  model for the conversation. Compatible services can use `OPENAI_API_KEY` and
  `OPENAI_BASE_URL`; `MS_AGENT_LLM_PROVIDER` / `MS_AGENT_LLM_MODEL` support first-run defaults.
- **Skills and MCP:** manage sources, scopes and enabled tools in the UI. MCP
  environment placeholders resolve from the server's environment.
- **Memory and search:** configure the relevant provider/credentials in Settings.
  Optional local vector embeddings require `fastembed>=0.8` and a first-use model download.
- **Files:** projects can use an existing directory; browse, edit and attach files
  in the workspace. Each streamed turn uses the selected project/session.

The launcher sets both `MS_AGENT_API_BASE_URL` (Node proxy) and
`MS_AGENT_FRONTEND_API_BASE_URL` (SSR) to its internal API. When starting
`pnpm start` manually, set both to the same API URL.
`MS_AGENT_FRONTEND_HOSTED_MODE=1` hides local-path controls for remote users; it
is not authentication. The app has no built-in login, so shared deployments need
appropriate access control.

## Docker and release preparation

The planned registry remains
`mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent`.
After the selected image has been built, verified and published:

```bash
docker run --rm -p 9000:8000 \
  -e MS_AGENT_HOME=/data -v ms-agent-data:/data \
  mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent:1.7.0rc0
```

Open http://127.0.0.1:9000. The container uses port 8000, keeps the API on loopback
8001, and has its dependencies prepared before startup. Keep the named volume
when replacing a container. The new SDK image flow must pass actual build and
ACR push/pull validation before it replaces the existing standalone Aone flow.
The image runs as UID 1000. The named volume above is initialized for that user;
existing bind mounts must be writable by UID 1000. Change the host mapping
(`9000:8000`) to select a different public port while retaining the container's
health-check port. Mount extra workspace directories explicitly when needed.

The [release guide](../docs/RELEASING.md) describes the manual image check, ACR
setup and the shared RC/stable tag workflow. Normal code checks do not build or
push images. Manual image checks default to no push; published images install
the exact wheel verified by that workflow run.

To prepare release packages from the SDK root, using Python 3.12 and the frontend
toolchain above:

```bash
python scripts/prepare_webui.py
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Both wheel and sdist contain the same selected WebUI resources. The sdist can
rebuild a wheel without Node or another frontend build. Release preparation
fails on missing/stale resources; editable SDK installation works without any
frontend build. After a docs or version-only change, `prepare_webui.py --skip-build`
can refresh the resource manifest if the existing frontend is still valid.

## Verification and synchronization

```bash
# In webui/backend/ (offline; real-model tests are opt-in):
uv run --no-sync pytest
# In webui/frontend/:
pnpm typecheck
pnpm build
# From the SDK root, in a Python environment with WebUI and test dependencies:
python -m pytest tests/cli tests/ui tests/release
```

For all optional embedding tests, first run `uv sync --locked --extra local-embed`
in `webui/backend/`; otherwise those tests are skipped. They do not download models.

[SOURCE.json](SOURCE.json) records the standalone snapshot and SDK base. Common
application/launcher changes belong in the standalone WebUI first; the SDK owns
CLI preparation, packaging, Docker and release integration. The embedded
`pyproject.toml` uses this SDK worktree as an editable path dependency. For a
separate WebUI checkout, reinstall the intended SDK with `uv pip install -e
<absolute-sdk-path>` after `uv sync`, and use `uv run --no-sync`; verify
`ms_agent.__file__` before testing.

Development guidance lives in [AGENTS.md](AGENTS.md); shared skills live at the
SDK root under `.agents/skills/`. They are not installed as runtime WebUI assets.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing/old Node or pnpm | Check `node --version`, `pnpm --version` inside `webui/frontend/`, and which executable is on PATH |
| Missing Python dependencies after wheel installation | Install the same version with the `[webui]` extra using the launcher's Python |
| Missing or stale build/CSS | Source: run `pnpm build`; wheel: reinstall the matching wheel and prepare its cache |
| Port is occupied | Stop the previous instance or choose explicit free public/API ports |
| Model/authentication error | Verify provider, model and credentials in Settings; inspect backend logs |
| Cache is damaged | Stop that version, remove only its reported cache directory, and start again; keep `MS_AGENT_HOME` intact |
| Windows output encoding | Use the PowerShell wrapper and a UTF-8 terminal; full Windows runtime validation requires a Windows environment |
