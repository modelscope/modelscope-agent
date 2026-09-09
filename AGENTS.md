# MS-Agent development

MS-Agent is a Python framework for agents, tools and workflows, with CLI, TUI
and WebUI interfaces.

## Architecture

- `ms_agent/agent/`, `llm/`, `workflow/`: execution loops, model adapters and
  multi-step orchestration.
- `ms_agent/config/`: configuration loading, provider settings, skills and MCP
  configuration. Extend these APIs when adding configuration options.
- `ms_agent/project/`, `session/`: project workspaces, session persistence and
  context assembly. Use these services rather than writing a second storage
  implementation in an interface layer.
- `ms_agent/tools/`, `skill/`, `mcp/`: tool execution, skill discovery and MCP
  connections. Keep discovery, configuration and runtime lifecycle separate.
- `ms_agent/ui/`: shared interaction events and input contracts; `cli/`, `tui/`
  and `webui/backend/` adapt them to their interfaces.
- `ms_agent/agent_hub/`: agent configuration and memory conversion between
  frameworks.
- `webui/`: FastAPI API and React Router frontend. See `webui/AGENTS.md` for UI
  conventions and request/stream handling.
- `projects/`, `examples/`, `tests/`, `docs/`: applications, examples, tests and
  bilingual documentation.
- `.dev_scripts/webui/`: package preparation and release validation; `setup.py`
  loads the build helper from this directory.

## Implementation guidelines

Keep model and tool execution in the SDK. Interface adapters should translate
requests, events and errors without duplicating agent execution or persistence.
When changing event payloads, check consumers in `ms_agent/ui/`, `ms_agent/tui/`
and `webui/backend/app/backends/ms_agent/` as well as frontend event types.

Async execution must handle cancellation and close the resources it owns,
including tasks, subprocesses, connections and file watchers. Cover normal
completion and interrupted execution when changing resource lifecycles.

Declare dependencies in the appropriate `requirements/` extra. WebUI dependencies
must also agree with `webui/backend/pyproject.toml` and its lockfile. Keep optional
integrations out of imports needed by unrelated SDK features.

Follow `setup.cfg` for SDK Python formatting and the existing conventions within
WebUI modules. Development skills are available in `.agents/skills/`.

Use the tests nearest the changed behavior. WebUI commands and checks are in
`webui/README.md`; installed-package behavior is checked by
`.dev_scripts/webui/check_webui_install.py`. Keep English and Chinese user
documentation aligned when changing commands, options or configuration.
