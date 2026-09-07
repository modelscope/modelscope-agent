# MS-Agent development

MS-Agent is a Python SDK with CLI/TUI interfaces and an embedded WebUI. Read the
code before relying on older design or handoff notes.

## Repository map

- `ms_agent/agent/`, `llm/`, `workflow/`: agent execution and model integration.
- `ms_agent/config/`, `project/`, `session/`: configuration and persisted state.
- `ms_agent/tools/`, `skill/`, `mcp/`: tools, skill discovery and MCP integration.
- `ms_agent/ui/`, `cli/`, `tui/`: interaction contracts and local interfaces.
- `ms_agent/agent_hub/`: import/export and conversion between agent frameworks.
- `webui/`: FastAPI adapters and React Router SSR frontend, synchronized from
  the standalone ms-agent-webui repository; provenance is in `webui/SOURCE.json`.
- `projects/`, `examples/`, `tests/`, `docs/`: applications, examples, tests and
  bilingual documentation.

## Development rules

For any WebUI change or `ms-agent ui` integration, read `webui/AGENTS.md` and the
relevant sections of `webui/README.md` / `README_ZH.md` first. Common WebUI code
belongs in the standalone repository and is synchronized as a committed
snapshot. SDK CLI preparation, package layout, Docker and release workflows are
maintained here. Preserve this boundary when fixing an issue.

Skills live in the root `.agents/skills/`; read only those relevant to the task.
WebUI conventions are scoped by `webui/AGENTS.md`, not imposed on the whole SDK.
Keep imported skills and their supporting files intact when synchronizing.

Use the intended editable SDK worktree and verify `ms_agent.__file__` before
testing. Never assume an existing virtual environment still points at it after
`uv sync`. Use temporary `MS_AGENT_HOME` directories in tests and avoid touching
the user's real projects, settings or credentials.

Run focused tests for changed behavior. WebUI changes also need backend offline
tests, frontend type checking and `pnpm build`; installed-package changes need
a wheel test outside the source checkout. See the WebUI README for commands.
The normal SDK Python formatting is defined in `setup.cfg`; copied WebUI and
skill snapshots retain their source conventions.

Use ordinary `feat/` or `fix/` branch names. Commit messages are a concise Chinese
sentence, without co-author trailers. Keep snapshots, SDK adaptations and release
configuration in separate commits when possible. Do not publish packages,
release tags or images unless the user has authorized that action.
