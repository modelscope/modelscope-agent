# WebUI development

Read the root `AGENTS.md` and [README.md](README.md) / [README_ZH.md](README_ZH.md)
for installation, configuration, development and verification commands.

## Layout and synchronization

- `backend/app/api/`: HTTP routes and envelopes.
- `backend/app/backends/ms_agent/`: SDK adapters for projects, sessions, chat,
  models, skills, MCP, memory and files.
- `backend/app/launcher.py`: the common SSR/API supervisor used by both
  `ms-agent ui` and the standalone `uv run webui` entry.
- `frontend/server.js`: SSR serving, same-origin API proxy and unbuffered SSE.
- `frontend/app/`: React routes, components, model/chat state and translations.
- `frontend/scripts/genAntdCss.tsx` and `buildManifest.mjs`: CSS generation and
  build input/output validation.

`SOURCE.json` records the standalone source commit. Keep common runtime and
launcher changes upstream in that repository, then synchronize the committed
files. Embedded `backend/pyproject.toml` and `uv.lock` use the current SDK as an
editable path dependency. SDK packaging/CLI changes belong outside this snapshot.
The root `.agents/skills/` contains the single copy of development skills.

## UI conventions

- Use project design tokens for colors, borders and backgrounds. Target Ant
  Design internals with a co-located CSS file and a component-specific prefix;
  avoid Tailwind arbitrary variants for those internal nodes.
- Obtain message, modal and notification APIs from `App.useApp()` so they
  inherit the application theme.
- Ant Design uses `zeroRuntime`: modify the shared theme in `app/lib/msaTheme.ts`,
  never add a nested `ConfigProvider theme`. Regenerate the complete frontend
  with `pnpm build` after theme changes.
- Route visible text through `useT()` and update both `app/lib/locales/en.json`
  and `zh.json` plus the corresponding types. Keep SSR and browser language
  selection consistent.
- Confirm delete actions with `Popconfirm`; give icon-only buttons a translated
  tooltip. Use `currentColor` in SVG assets.
- If a mutation affects mounted views, notify consumers through
  `app/lib/events.ts` instead of leaving stale panels. Follow existing workspace,
  skill/MCP, URL and session-completion event patterns.

## Runtime checks

Use the full `pnpm build` command: CSS, SSR, client output and build provenance
must agree. Both `MS_AGENT_API_BASE_URL` (proxy) and
`MS_AGENT_FRONTEND_API_BASE_URL` (SSR) must target the same loopback API.
Do not buffer or compress SSE. Keep one API worker while session/runtime state
is held in process memory. A service failure must stop the whole launched stack.

Run offline tests from `backend/` with `uv run --no-sync pytest`, and
`pnpm typecheck` / `pnpm build` from `frontend/`. Validate actual CSS responses,
port handling and shutdown after launcher changes. Real-model tests are opt-in;
never use the user's real SDK home for automated checks.
