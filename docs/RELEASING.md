# MS-Agent 发版指南

RC 和正式版使用相同的构建、验收与发布流程。包版本采用 `X.Y.ZrcN` 或
`X.Y.Z`，Git tag 在版本前加 `v`，镜像标签与包版本相同。

## 工作流与前置条件

| 工作流 | 入口与执行内容 | 发布行为 |
| --- | --- | --- |
| WebUI checks | 相关代码 push / PR 或手动运行；测试后端与 SDK，检查前端，构建并验收 wheel/sdist | 不发布 |
| WebUI image | 手动运行；先执行 WebUI checks，再构建 Linux amd64 镜像并验收容器 | 默认不推送；选择 `push_dev=true` 才推送 `dev-<SDK SHA 前 12 位>` |
| release | 推送版本 tag；完成包和镜像验收后发布 | 仅在 `modelscope/ms-agent` 发布 PyPI 和 ACR，不更新 `latest` |

镜像地址为：
`mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent`。
镜像安装已经验收的 wheel，Python 依赖由后端 lockfile 导出，前端安装运行依赖。
Python、Node、pnpm 等工具版本以 [Dockerfile](../docker/webui.Dockerfile) 和
[工作流](../.github/workflows/webui-check.yaml) 为准。

首次发布前，确认以下配置：

- 已启用 GitHub Actions。手动工作流必须存在于默认分支，才能通过 `--ref` 选择待测分支或 tag。
- Actions secrets 已配置 `PYPI_API_TOKEN`、`ACR_USERNAME`、`ACR_PASSWORD`。
  不推送的镜像验证不需要这些凭据。
- ACR 公网入口与访问控制允许 GitHub runner 连接，账号有目标镜像仓库的推拉权限。
  以一次完整的测试镜像推送和回拉验证网络及权限。

## 准备版本

1. 从已完成 review 的代码建立或更新 release 分支，确认本轮要包含的功能和修复。
2. 更新 `ms_agent/version.py`。在 `webui/backend/` 执行 `uv lock`，提交版本和
   lockfile 的变更，再用 `uv sync --locked` 检查依赖能否安装。
3. 根据功能变化更新双语 README、安装、CLI 和相关使用文档。通用安装示例保持
   `pip install -U "ms-agent[webui]"`；只有测试或说明某个版本时才指定版本号。
4. 准备 GitHub Release 和用户公告：说明主要变化、升级方式、环境要求与已知问题，
   附对应 PR。RC 额外说明希望用户重点验证的功能。
5. 运行 WebUI checks 和不推送的 WebUI image，并检查真实模型与目标操作系统上的主要使用场景。
   自动测试覆盖启动、接口和数据持久化，不代替真实模型验收。

本地检查并构建安装包：

```bash
python .dev_scripts/webui/check_webui_release.py
python .dev_scripts/webui/prepare_webui.py
uv build --python 3.12
uvx --from twine==6.2.0 twine check dist/*.whl dist/*.tar.gz
```

构建产物写入 `dist/`，这些命令不会上传包。环境要求与工具职责见
[构建工具说明](../.dev_scripts/webui/README.md)。

## 验证镜像

在 GitHub Actions 页面选择 **WebUI image** 和目标分支，保持 `push_dev=false`，
或使用以下命令测试 `main`：

```bash
gh workflow run webui-image.yaml --repo modelscope/ms-agent \
  --ref main -f push_dev=false
gh run list --repo modelscope/ms-agent --workflow webui-image.yaml --limit 5
```

测试其他分支或 tag 时替换 `--ref`；在 fork 中测试时替换 `--repo`。
需要验证 ACR 推拉时，设为 `push_dev=true`。这只发布对应提交的测试镜像标签，
不发布 PyPI 包，也不会覆盖已有的不同内容。

工作流保存 `webui-release-inputs` 和 `webui-validated-image` 两份 artifact。
前者包含 wheel、sdist、锁定依赖、`release.json` 和校验和；后者包含验收后的镜像与记录。
镜像构建参数和本地复现方式以 [WebUI image 工作流](../.github/workflows/webui-image.yaml) 为准。

容器验收覆盖页面和 CSS、API、项目、会话、技能、文件、流式响应、重启后的数据及异常退出。
测试结束后清理创建的测试容器和数据卷。

## 发布版本

确认 release 分支的代码、版本和检查结果后，推送版本 tag。以下命令从
`ms_agent/version.py` 读取版本；`origin` 应指向有发布权限的官方仓库：

```bash
VERSION=$(python -c 'from ms_agent.version import __version__; print(__version__)')
python .dev_scripts/webui/check_webui_release.py --tag "v$VERSION"
git tag -a "v$VERSION" -m "Release $VERSION"
git push origin "v$VERSION"
```

tag 推送触发 [release 工作流](../.github/workflows/publish.yaml)。流程从该提交构建并
验收包和镜像，确认 ACR 可连接后上传 PyPI，再推送同一个镜像并回拉核对。
创建 GitHub Release 页面本身不触发发包。

工作流成功后：

- 从 PyPI 安装本轮版本，从 ACR 拉取同标签镜像，确认版本、启动和主要使用场景。
- 创建 GitHub Release。RC 标为 prerelease，正式版设为稳定发布。PyPI 项目介绍来自
  构建时的根 README，详细更新说明放在 GitHub Release。
- 检查文档站已经更新，再发送用户公告、升级命令和发布说明链接。
- 收集 RC 反馈，修复后发布下一 RC；准备就绪后改为正式版本并重复同一流程。
  将 release 分支上的修复和文档同步回 main。

## 失败恢复与发布记录

- 优先在原 Actions run 选择 **Re-run failed jobs**，复用已经保存的原始 artifact。
  PyPI 成功而 ACR 失败时，修复网络或凭据后重试原 `publish` job。
- 已上传的包按 SHA256 核对：相同内容可继续，缺少的文件才上传，不同内容会失败。
  已存在的镜像标签也会核对验收记录，网络或认证失败不会被当作“标签不存在”。
- 若已发布内容有误，发布下一 RC 或补丁版本，不移动旧 tag、不覆盖旧产物。
  原始产物丢失且无法确认内容一致时，也应使用新版本。

保存 tag、SDK commit、`release.json`、包的 SHA256、镜像 digest 和 Actions run 链接。
工作流 artifact 默认保留 30 天，需要长期留存时应另行归档。
