# MS-Agent WebUI

在浏览器里使用 MS-Agent，让智能体围绕你的项目完成资料检索、代码编写和文件处理。对话、工具调用和生成结果都在同一个工作台中，方便查看过程并继续追问。[English](README.md)

- **按项目开展工作**：打开本地文件夹，管理多个会话，浏览和编辑项目文件。
- **看清任务进展**：流式查看回复、思考过程、工具调用和生成的文件。
- **选择模型与工具**：配置模型服务，接入 MCP 工具，并为项目启用所需的技能。
- **延续项目上下文**：保留会话记录，管理记忆，在后续对话中继续工作。

## 快速开始

### 1. 准备环境

| 工具 | 要求 | 用途 |
| --- | --- | --- |
| [Python](https://www.python.org/downloads/) | 3.12 或更高版本 | 运行 MS-Agent 和 WebUI 后端 |
| [Node.js](https://nodejs.org/en/download) | 22.22.0 或更高版本 | 运行 WebUI 前端服务 |
| pnpm | 10.17.1 | 安装前端依赖 |

安装 Node.js 后，在终端安装 pnpm：

```bash
npm install --global pnpm@10.17.1
```

可以用 `python --version`、`node --version` 和 `pnpm --version` 检查环境。使用已有的 Python 环境即可，也可以按需创建虚拟环境。

### 2. 安装并启动

```bash
pip install -U "ms-agent[webui]"
ms-agent ui
```

`[webui]` 会安装界面所需的 Python 依赖。首次启动还会下载前端运行依赖，请保持网络连接；后续启动会复用它们。安装包已经包含构建好的页面和样式，无需手动构建前端。

浏览器会自动打开 WebUI，通常是 **http://127.0.0.1:8000**。如果端口被占用，会选择后续可用端口，以终端打印的地址为准。按 **Ctrl-C** 停止服务。

### 3. 开始对话

1. 打开 **设置 → 模型设置**，添加模型服务的 API Key、接口地址和模型。
2. 新建或打开项目，按需选择工作目录、技能和 MCP 工具。
3. 创建会话，选择模型，输入任务；需要时可附上文件或图片。

## 常用启动方式

```bash
# 指定访问端口
ms-agent ui --port 8080

# 只启动服务，不自动打开浏览器
ms-agent ui --no-browser

# 允许通过本机的其他网络地址访问
ms-agent ui --host 0.0.0.0 --port 8000
```

应用没有内置登录。向其他用户开放服务时，需要通过反向代理或网络设置配置访问控制。

| 参数 | 说明 |
| --- | --- |
| `--host HOST` | 监听地址，默认 `127.0.0.1` |
| `--port PORT` | 指定浏览器访问的端口；省略时从 8000 开始选择 |
| `--backend-port PORT` | 指定内部 API 端口，通常无需设置 |
| `--no-browser` | 不自动打开浏览器 |
| `--skip-install` | 跳过依赖安装，仍校验页面和样式；源码构建过期时仍会重新构建 |
| `--prepare-only` | 准备依赖后退出，不启动服务 |
| `--startup-timeout SECONDS` | 启动等待时间，默认 120 秒 |
| `--production` | 兼容参数，默认已使用构建后的前端 |
| `--reload` | 暂不支持，热更新请使用下方开发命令 |

手动指定的端口必须空闲，且前端与内部 API 端口不能相同。任一服务异常退出时，启动器会停止另一服务并报告错误。

## 配置与数据

模型、工具和记忆通常可以直接在界面的设置中配置。也可通过环境变量提供配置，例如 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，以及首次启动时的 `MS_AGENT_LLM_PROVIDER`、`MS_AGENT_LLM_MODEL` 默认值。

项目配置、会话和托管技能默认保存在 `~/.ms_agent`；可以通过 `MS_AGENT_HOME` 指定其他目录。项目工作目录中的文件保存在原位置。前端依赖缓存与这些数据分开，通过 `MS_AGENT_WEBUI_CACHE` 可指定缓存位置。

从 pip 安装时，WebUI 读取进程环境变量和已保存的 SDK 设置，不自动查找当前目录的 `.env`。从源码运行时，还会依次读取仓库根目录、`webui/`、`webui/backend/` 下的 `.env`，后者优先，已设置的进程环境变量优先级最高。可参考 [配置示例](backend/.env.example)。

本地向量记忆需要额外安装 `fastembed`，首次使用时会下载嵌入模型。其他模型和搜索服务按各自配置使用，不必为普通对话安装本地嵌入模型。

## 从源码运行与开发

源码运行另需 [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.5 或更高版本，可用 `pip install uv` 安装。

```bash
git clone https://github.com/modelscope/ms-agent.git
cd ms-agent
pip install -e .
ms-agent ui
```

首次启动会准备后端环境、安装前端依赖并执行完整构建，包括 CSS 生成。修改源码后，再次启动会检查构建是否需要更新。

需要热更新时，在仓库根目录打开两个终端：

```bash
# 终端 1：后端
cd webui/backend
uv sync --locked
uv run dev
```

```bash
# 终端 2：前端
cd webui/frontend
pnpm install --frozen-lockfile
pnpm dev
```

访问 **http://localhost:5173**。前端开发服务器默认连接本机 8000 端口的后端。

运行检查：在 `webui/backend/` 执行 `uv run pytest`；在 `webui/frontend/` 执行 `pnpm typecheck` 和 `pnpm build`。完整构建会同时生成 CSS、页面和服务端文件，请使用 `pnpm build`，不要只运行其中一个子步骤。

Windows 使用相同的安装和启动命令。源码运行时也可使用 PowerShell 脚本：

```powershell
.\webui\scripts\start-webui.ps1 --no-browser
```

开发约定见 [AGENTS.md](AGENTS.md)，构建安装包的命令见 [构建工具说明](../.dev_scripts/webui/README.md)。

## Docker 运行

使用 Docker 时无需在宿主机安装 Python、Node.js 或 pnpm。将下面的 `TAG` 替换为要使用的已发布镜像标签：

```bash
docker run --rm -p 9000:8000 \
  -e MS_AGENT_HOME=/data -v ms-agent-data:/data \
  mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent:TAG
```

打开 **http://127.0.0.1:9000**。`ms-agent-data` 保存应用数据，替换容器时保留该数据卷；需要操作宿主机的项目文件时，另行挂载对应目录。更改访问端口只需调整 `9000:8000` 左侧的值。

设置 `MS_AGENT_FRONTEND_HOSTED_MODE=1` 可隐藏不适合远程用户操作的本地路径控件；访问控制仍需单独配置。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `ms-agent`、`node` 或 `pnpm` | 确认工具已安装，且当前终端可以访问相应命令；安装后可重新打开终端 |
| Python 或 Node 版本不满足要求 | 使用上方列出的版本，确认终端中实际使用的解释器 |
| 指定端口被占用 | 更换 `--port`，或省略它让启动器自动选择 |
| 页面或样式缺失 | 源码运行时重新执行 `pnpm build`；pip 安装时重新安装当前包，按报错提示重新准备对应缓存 |
| 模型连接或认证失败 | 检查模型设置中的 API Key、接口地址、模型名称及网络连接 |
| 缺少 WebUI 的 Python 依赖 | 在启动命令使用的环境中执行 `pip install -U "ms-agent[webui]"` |
