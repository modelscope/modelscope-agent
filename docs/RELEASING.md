# SDK 1.7 发版操作

RC 和正式版共用同一个 tag 工作流。`v1.7.0rc0`、`v1.7.0rc1`、`v1.7.0`
分别对应包版本和镜像标签 `1.7.0rc0`、`1.7.0rc1`、`1.7.0`。
目前迁移分支仍使用原有 SDK 版本；本文的 1.7 命令应在本轮版本准备好后执行。

## 工作流与前置条件

| 入口 | 执行内容 | 是否发布 |
| --- | --- | --- |
| WebUI checks：相关代码 push / PR，或手动执行 | 后端与 SDK 测试、前端类型/CSS/SSR、wheel/sdist、仓库外安装与退出检查 | 不发布、不构建镜像 |
| WebUI image：手动执行 | 上述检查 → Linux amd64 镜像 → 容器验收 → 保存原始产物 | 默认不推送；显式选择 `push_dev=true` 才推送 `dev-<SDK SHA 前 12 位>` 并重新拉取核对 |
| release：推送版本 tag | 上述完整验收 → 检查 ACR → PyPI → 推送并回拉同一个已验收镜像 | 仅官方 `modelscope/ms-agent` 仓库发布，不更新 `latest` |

镜像沿用现有地址：
`mshub-registry.cn-zhangjiakou.cr.aliyuncs.com/modelscope-repo/ms-agent`。
Python 3.12、Node 22、pnpm 10.17.1、uv 0.12.8、git/curl/tini、SSR/API 与持久化能力
参考独立 WebUI 的 Aone 流程。新镜像安装本轮已验收 wheel，第三方 Python 依赖来自
嵌入后端 lock 的导出，前端只安装运行依赖；使用共同 launcher 管理两个服务。

正式交接前确认：

- GitHub Actions 已启用；手动工作流文件必须先存在于仓库默认分支，随后才能通过
  `--ref` 选择待测分支。仅推到特性分支不会让 `workflow_dispatch` 入口生效。
- 仓库 Actions secrets 配置 `PYPI_API_TOKEN`、`ACR_USERNAME`、`ACR_PASSWORD`。
  Aone 的凭据不会自动迁入 GitHub。不推送的镜像验证不需要这些 secrets。
- ACR 公网入口和访问控制允许 GitHub runner 连接。先前基础预检中 DNS 正常、TCP 443
  超时，尚未验证登录和推拉；此项需在 ACR 控制台及实际 runner 上确认。
- 在旧 Aone 继续可用的期间，先完成一次完整 SDK 镜像构建、容器验收和测试标签推拉，
  再决定停用旧发布入口。基础镜像预检不能替代完整应用验收。

## 先测试镜像

由维护者确认代码后触发，默认不发布：

```bash
gh workflow run webui-image.yaml --repo modelscope/ms-agent \
  --ref main -f push_dev=false
gh run list --repo modelscope/ms-agent --workflow webui-image.yaml --limit 5
```

ACR 网络和凭据准备好后，再以 `push_dev=true` 运行同一入口。此时构建、检查后只发布
该提交的 `dev-...` 标签，不发布 PyPI 或正式版本。已存在的不同内容不会被覆盖。
若工作流尚未进入默认分支，先完成工作流配置的合入，再执行上述命令。
上述命令测试合入后的 `main`；测试其他版本时，把 `--ref` 换成目标仓库中实际存在的
分支或 tag。个人 fork 中的分支应使用该 fork 的 `--repo`。

本地复现完整镜像时，从同一轮检查下载 `webui-release-inputs` artifact，然后：

```bash
# 在对应 SDK commit 的根目录；release-inputs/ 是下载并解压的 artifact。
(cd release-inputs && shasum -a 256 -c SHA256SUMS)
python scripts/check_webui_release.py --dist release-inputs --sdk-sha "$(git rev-parse HEAD)"
```

实际 `docker build` 参数见 [WebUI image 工作流](../.github/workflows/webui-image.yaml)：
构建上下文仅为 `release-inputs/`，Dockerfile 是 [docker/webui.Dockerfile](../docker/webui.Dockerfile)，
版本、SDK SHA、wheel SHA 从 `release.json` 读取。构建后执行
`python scripts/check_webui_image.py --inputs release-inputs --output image.json --logs container-logs`。
它创建独立容器/数据卷，检查 SSR、CSS、API、项目/会话/技能/文件、SSE、重启后数据和失败退出，
最后清除测试容器和测试卷；不会请求模型或推送镜像。

## 准备本轮版本

1. 确认迁移及其他待发功能已经合入，更新 `release/1.7`，记录最终 SDK commit。
   不混入未跟踪文件，不移动已发布 tag；已有 release worktree 的改动先保留并核对。
2. 把 `ms_agent/version.py` 改成目标版本，例如 `1.7.0rc0`。在 `webui/backend/`
   执行 `uv lock`，提交版本和 lock；用 `uv sync --locked` 检查两者一致。
3. 检查根双语 README、WebUI 双语 README、安装/CLI 文档中的示例与兼容性说明；
   正式版说明应覆盖从 1.6.0 起的主要变化，RC 说明写清待验证事项。
4. 写好本轮 GitHub Release 与用户通知草稿，说明升级命令、WebUI 环境要求、镜像标签、
   数据保留方式、已知问题；附实际 commit/PR 变更链接。
5. 运行 WebUI checks 和默认不推送的 WebUI image，完成必要的真实模型/目标系统检查，
   再确认本轮 tag。自动测试使用临时 SDK home，不会验证个人配置或真实模型质量。

本地发包准备与版本校验：

```bash
python scripts/check_webui_release.py --tag v1.7.0rc0
python scripts/prepare_webui.py
uv build --python 3.12
uvx --from twine==6.2.0 twine check dist/*.whl dist/*.tar.gz
```

构建 wheel 不等于发布。新的 SDK wheel 已含 WebUI 资源和预构建前端；
`pip install "ms-agent[webui]==1.7.0rc0"` 额外安装 Python 运行依赖。
启动前仍需 Node 22.22+，首次准备还需 pnpm；完整环境与命令见
[WebUI README](../webui/README_ZH.md)。

## 由维护者触发发布

确认 release 分支工作区干净、提交及检查结果正确，再执行：

```bash
git switch release/1.7
git pull --ff-only upstream release/1.7
python scripts/check_webui_release.py --tag v1.7.0rc0
git tag -a v1.7.0rc0 -m "发布 1.7.0rc0"
git push upstream v1.7.0rc0
```

tag 推送触发 `.github/workflows/publish.yaml`；创建 GitHub Release 页面本身不触发发包。
流程从 tag commit 构建并验收一次：先保存 wheel/sdist/依赖清单及镜像，再检查 ACR
可用性，发布 PyPI，最后推送同一个镜像并回拉核对。RC 与正式版没有不同的构建路径。
首次创建远端 `release/1.7` 时，应先正常 push 分支，之后再使用上面的 pull 命令。

成功后由维护者完成：

- 从 PyPI 安装指定版本、从 ACR 拉取本轮镜像，检查版本、启动及关键使用场景。
- 创建 GitHub Release；RC 标为 prerelease 且非 Latest，正式版设置为稳定发布。
  PyPI 的项目介绍来自该包构建时的根 README，单独的发布说明放在 GitHub Release。
- 确认文档站实际构建了对应版本；公告只链接已经可访问、内容正确的文档。
- 在用户群发送已确认的公告与升级方式，收集 RC 反馈。修复后发布下一 RC，
  最终改为 `1.7.0` 并重复同一流程；将 release 分支中的修复和文档回合 main。

## 失败后恢复

- 优先在原 Actions run 选择 **Re-run failed jobs**，复用同一次运行保存的原始 artifact；
  不要从头重新构建已上传过的相同版本。artifact 默认保留 30 天，应及时保存发版记录。
- PyPI 已上传的文件会核对 SHA256：相同字节允许继续，缺少的文件才上传，不同内容直接失败。
  镜像标签已存在时也核对已验收的 image ID，不把网络/鉴权失败当作“版本不存在”。
- PyPI 成功而 ACR 失败时，先修复网络/凭据，再重试原 `publish` job；不会再次构建镜像。
  ACR 登录成功不代表公网推拉一定可用，以完整检查结果为准。
- 若已发布的包内容有错，应发下一 RC 或补丁版本；不覆盖旧包、旧镜像或移动 tag。
  产物丢失且无法证明一致时也应重新确定版本，不使用 `--skip-existing` 掩盖差异。

发布记录至少保存 tag、SDK SHA、`webui/SOURCE.json` 的源 commit、`release.json`、
wheel/sdist SHA256、镜像 digest 和对应 Actions run 链接。
