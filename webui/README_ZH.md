# MS-Agent WebUI

WebUI 提供项目与会话、流式对话、模型配置、技能、MCP 工具、记忆和工作区文件管理，与 CLI/TUI 共用 SDK 和数据目录。[English](README.md)

本次代码用于准备 **1.7 发版**。以下 wheel 安装方式适用于本分支构建的包；旧版 1.6 包不包含这套 WebUI。1.7 包发布前，请使用当前源码或本地构建的 wheel。

## 从源码快速启动

前置环境：Python **3.12+**、Node **22.22.0+**、pnpm **10.17.1**、uv **0.5+**。没有 pnpm 时可执行 `npm install --global pnpm@10.17.1`。

在 SDK 仓库根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
ms-agent ui
```

首次运行会准备 `webui/backend/.venv`、安装锁定的前端依赖，并执行包含 Ant Design CSS 生成的 `pnpm build`；后续复用未变化的构建。页面、API 和 CSS 检查通过后会打印一个访问地址，通常为 **http://127.0.0.1:8000**。在 **设置 → 模型设置** 配置服务商与模型，再打开项目/会话进行对话。

Ctrl-C 会关闭前后端。Windows 请使用 Python 3.12+ 的 `python`，通过 `.venv\Scripts\Activate.ps1` 激活环境，也可使用保留的 UTF-8 包装脚本：

```powershell
.\webui\scripts\start-webui.ps1 --no-browser
```

## 安装 wheel

Python、Node、pnpm 要求相同。发布包携带前后端源码，以及预构建的 SSR、客户端资源和 CSS；用户无需再次构建前端，也不会运行源码项目的 uv 同步。

```bash
# 所选 1.7 RC 发布后：
python -m pip install 'ms-agent[webui]==1.7.0rc0'
ms-agent ui

# 发布前，安装当前源码构建的 wheel：
python -m pip install './dist/ms_agent-1.6.0-py3-none-any.whl[webui]'
ms-agent ui
```

文件名和版本以 `dist/` 实际产物为准；正式改版前开发版本仍为 1.6.0。普通 `pip install ms-agent` 也下载相同的 WebUI 资源，`[webui]` 额外安装 Python 运行依赖；`[all]` 同样包含这些依赖。运行 WebUI 要求 Python 3.12+。

wheel 首次启动会在用户缓存安装**生产 Node 依赖**，后续复用，不向 site-packages 安装依赖。可通过 `MS_AGENT_WEBUI_CACHE` 指定可写缓存根目录，镜像也用它预先准备固定位置。默认位置为 macOS 的 `~/Library/Caches/ms-agent/webui`、Linux 的 `$XDG_CACHE_HOME/ms-agent/webui` 或 `~/.cache/ms-agent/webui`、Windows 的 `%LOCALAPPDATA%\ms-agent\webui`。缓存按 SDK 版本和资源内容区分；Node 版本、平台或锁文件变化时需重新准备 Node 依赖。

## 参数与开发方式

`ms-agent ui` 调用与 `webui/backend/` 内 `uv run webui` 相同的 `app.launcher`。默认运行已构建的 SSR 应用，浏览器只访问一个端口，API 保持在本机回环地址。

| 参数 | 行为 |
| --- | --- |
| `--host HOST` | 公开监听地址，默认 `127.0.0.1` |
| `--port PORT` | 固定公开端口；省略时从 8000 开始选择可用端口 |
| `--backend-port PORT` | 固定内部 API 端口；省略时使用后续可用端口 |
| `--no-browser` | 不自动打开浏览器 |
| `--skip-install` | 不下载或同步依赖；仍校验构建与 CSS |
| `--production` | 默认 SSR 行为的兼容参数 |
| `--reload` | 暂不支持，开发时使用下面的命令 |
| `--prepare-only` | 准备依赖和资源后退出，供镜像构建等场景使用 |
| `--startup-timeout SECONDS` | 启动超时，默认 120 秒 |

显式指定的端口必须空闲、位于 1–65535，且前后端不能相同。公开端口为 65535 时，内部端口自动从 8000 开始选择。旧公开默认端口为 7860，如需沿用，可执行 `ms-agent ui --port 7860`。启动失败或任一服务意外退出时，整体停止并返回失败。

需要前后端热更新时，分别开两个终端：

```bash
# SDK 根目录，终端 1：
cd webui/backend
uv sync --locked
uv run --no-sync dev

# SDK 根目录，终端 2：
cd webui/frontend
pnpm install --frozen-lockfile
pnpm dev
```

访问 **http://localhost:5173**，开发服务器会代理 8000 端口上的 API。运行构建后的应用时，在 `webui/frontend/` 执行 `pnpm build`，再到 `webui/backend/` 执行 `uv run --no-sync webui --no-open`。

`pnpm build` 会生成 CSS、SSR、客户端资源及 `build/webui-build.json`；不要直接运行 React Router CLI 来替代它，也不要混用新 CSS 与旧 SSR。源码模式下，即使传了 `--skip-install`，发现旧构建时仍会使用已有 pnpm/依赖重新构建；wheel 模式始终使用包内预构建前端。

## 配置与功能

默认数据目录是 **`~/.ms_agent`**，与 CLI/TUI 共享。通过 `MS_AGENT_HOME` 指定绝对路径，可隔离测试或部署数据。目录内保存设置、项目、会话及托管技能，升级前请备份。

源码 dotenv 优先级为：进程环境 > `webui/backend/.env` > `webui/.env` > SDK 根 `.env`，支持的字段见[配置示例](backend/.env.example)。wheel 模式使用进程环境和 SDK 设置，不在 site-packages、缓存或当前工作目录附近自动寻找 `.env`。

- **模型**：在设置中配置服务商凭据和模型，对话时选择模型。兼容服务可通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 配置；`MS_AGENT_LLM_PROVIDER` / `MS_AGENT_LLM_MODEL` 支持首次初始化。
- **技能与 MCP**：通过界面管理来源、作用域和启用状态；MCP 环境占位符从服务进程环境解析。
- **记忆与搜索**：在设置中配置相关服务与凭据；可选的本地向量记忆需要 `fastembed>=0.8`，首次使用还需下载模型。
- **文件**：项目可使用已有目录，工作区支持浏览、编辑和添加附件；流式对话使用当前项目和会话。

统一启动器会同时设置 `MS_AGENT_API_BASE_URL`（Node 代理）与 `MS_AGENT_FRONTEND_API_BASE_URL`（SSR）。手动运行 `pnpm start` 时，需把两者指向同一后端。

`MS_AGENT_FRONTEND_HOSTED_MODE=1` 用于隐藏远程用户不适合操作的本地路径控件，不提供身份认证。应用没有内置登录，共享部署需要另行配置访问控制。

## Docker 与发版准备

镜像仓库继续为 `mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent`。所选镜像完成构建、验证并发布后：

```bash
docker run --rm -p 9000:8000 \
  -e MS_AGENT_HOME=/data -v ms-agent-data:/data \
  mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent:1.7.0rc0
```

访问 http://127.0.0.1:9000。容器内公开端口为 8000，API 使用回环地址 8001；启动前已准备好依赖。替换容器时保留命名数据卷。新的 SDK 镜像流程通过实际构建与 ACR 推拉验证后，再接替旧独立 WebUI 的 Aone 流程。

在 SDK 根目录准备发布包，使用 Python 3.12 和上述前端工具链：

```bash
python scripts/prepare_webui.py
python -m pip install build twine
python -m build
python -m twine check dist/*
```

wheel 和 sdist 包含同一组选定资源；从 sdist 重建 wheel 不需要 Node 或再次构建前端。资源缺失或过期时会拒绝打包，SDK editable 安装则不要求已有前端构建。只改文档或版本时，可用 `prepare_webui.py --skip-build` 刷新资源清单；它仍会检查前端是否有效。

## 验证与同步

```bash
# webui/backend/；默认离线，真实模型测试需显式开启：
uv run --no-sync pytest
# webui/frontend/：
pnpm typecheck
pnpm build
# SDK 根目录，使用已安装 WebUI 和测试依赖的 Python：
python -m pytest tests/cli tests/ui
```

如需完整覆盖可选的向量记忆测试，先在 `webui/backend/` 执行 `uv sync --locked --extra local-embed`；否则相关测试会跳过。测试不会下载模型。

[SOURCE.json](SOURCE.json) 记录独立仓库快照与 SDK 起点。通用应用/启动器修复先在独立 WebUI 完成；SDK 维护 CLI 准备、打包、Docker 和发版接线。嵌入后端使用当前 SDK worktree 的 editable path 依赖。

独立 WebUI 联调时，`uv sync` 后执行 `uv pip install -e <SDK绝对路径>`，之后使用 `uv run --no-sync`，避免本地覆盖被还原。测试前确认 `ms_agent.__file__` 指向预期 worktree。

开发规则见 [AGENTS.md](AGENTS.md)，共享技能位于 SDK 根 `.agents/skills/`；它们不作为 WebUI 运行资源安装。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| Node/pnpm 缺失或版本不符 | 检查 `node --version`、在 `webui/frontend/` 中检查 `pnpm --version`，以及 PATH 命中的实际程序 |
| wheel 安装后缺少 Python 依赖 | 使用启动器的 Python，安装同版本的 `[webui]` extra |
| 构建或 CSS 缺失/过期 | 源码重新执行 `pnpm build`；wheel 重新安装匹配的包并准备缓存 |
| 端口占用 | 关闭旧实例，或显式指定空闲的公开/API 端口 |
| 模型或认证报错 | 检查设置中的服务商、模型和凭据，并查看后端日志 |
| 缓存损坏 | 先停止对应版本，只移除报错中的该版本缓存后再启动；保留 `MS_AGENT_HOME` |
| Windows 输出乱码 | 使用 PowerShell 包装脚本和 UTF-8 终端；完整 Windows 运行验收仍需 Windows 环境 |
