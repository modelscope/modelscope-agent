# WebUI development

MS-Agent WebUI has a FastAPI API, SDK adapters and a React Router frontend.
Installation and development commands are in `README.md` and `README_ZH.md`.

## Architecture and contracts

- `backend/app/api/` handles HTTP requests and response envelopes. SDK adapters
  in `backend/app/backends/ms_agent/` implement projects, sessions, chat, models,
  skills, MCP, memory and files through the SDK services.
- `backend/app/launcher.py` supervises the API and frontend processes.
  `ms_agent/cli/ui.py` prepares the environment before invoking this launcher.
- `frontend/server.js` serves rendered pages and proxies API requests and SSE.
  `frontend/app/` contains routes, components, client state and translations.
- `frontend/scripts/genAntdCss.tsx` and `buildManifest.mjs` generate CSS and record
  frontend build inputs/outputs. `backend/app/frontend.py` validates the result.
- The API/client contract is represented by backend models and
  `frontend/app/lib/types.ts`; update both sides when changing request or event
  payloads. Preserve streaming frame order, identifiers and completion handling.

## UI conventions

Use the project's design tokens for colors, spacing and typography. Target Ant
Design internals in co-located CSS with a component-specific selector rather
than Tailwind arbitrary variants. Icon-only controls need translated labels or
tooltips; destructive actions use `Popconfirm`.

Ant Design uses `zeroRuntime`. Change shared theme values in `app/lib/msaTheme.ts`
and regenerate CSS with `pnpm build`; a nested `ConfigProvider theme` cannot
supply runtime styles. Obtain message, modal and notification APIs from
`App.useApp()` so they inherit the application theme.

Use `useT()` for visible text, update both `app/lib/locales/en.json` and `zh.json`,
and keep SSR and browser language selection consistent. After changing data
shown elsewhere, update the appropriate state or emit the existing event in
`app/lib/events.ts` so mounted views refresh.

## Runtime and build behavior

The frontend proxy and SSR requests must target the same backend:
`MS_AGENT_API_BASE_URL` and `MS_AGENT_FRONTEND_API_BASE_URL`. SSE responses must
remain unbuffered and uncompressed. Session/runtime state is held in process
memory, so the application runs one API worker.

A service failure must stop the launched stack. Shutdown must close SDK
resources and file watchers before Python exits.

`pnpm build` generates the CSS, client files, SSR output and build manifest as a
unit. Package preparation validates that they match. When adding files needed
at runtime, update `.dev_scripts/webui/resource-files.txt` as well.

Backend tests run from `backend/` with `uv run pytest`; frontend checks run from
`frontend/` with `pnpm typecheck` and `pnpm build`. For launcher changes, verify
port handling, actual page/CSS responses and shutdown. Model integration tests
are opt-in; ordinary backend tests do not require model credentials.
