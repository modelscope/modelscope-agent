/**
 * The frontend's runtime environment variables — the ONLY place `process.env` is
 * read. Adding a variable means touching this file and nothing else.
 *
 * Every value is SERVER-SIDE, hence the `SERVER_` prefix on each export. That is
 * not a convention, it is what the platform allows: the client build replaces
 * `process.env` with the literal `{}`, so reading any key in the browser yields
 * `undefined` no matter what the deployment sets. Verified in the built bundle.
 * The only way a value could reach client code is a `VITE_` prefix, which bakes
 * it into the artifact at build time — deliberately avoided, so one image can run
 * both locally and hosted.
 *
 * Therefore: to use one of these in a COMPONENT, return it from a loader and read
 * it back via loader data. `hosted.ts` does exactly that for `SERVER_HOSTED_MODE`
 * and is the only correct way for a component to ask about it. Importing the
 * constant into a component instead would give `true` during SSR and `false` in
 * the browser — a hydration mismatch. This module is isomorphic (`api.ts` needs
 * it and is itself isomorphic, so a `.server` module could not be imported
 * there), which means that mistake compiles; the `SERVER_` prefix is the warning
 * at the point of use.
 *
 * Where the values come from:
 *   - `pnpm dev` / `pnpm build` — `@react-router/dev`'s vite plugin merges
 *     `backend/.env` into `process.env` (all keys, not just `VITE_`-prefixed)
 *     from its `config` hook. One file configures both processes, so
 *     `vite.config.ts` points `envDir` at the backend. Every key this project
 *     owns carries an `MS_AGENT_` prefix so a shared shell cannot collide with it,
 *     and `MS_AGENT_FRONTEND_` marks the ones this side owns. Third-party keys are
 *     the exception — `OPENAI_API_KEY`, `EXA_API_KEY` and the like are matched
 *     verbatim by the SDK and by `mcp.json` `${VAR}` placeholders.
 *
 *     That merge is `Object.assign`, so it only ever ADDS: once a key has been
 *     seen it stays in `process.env` for the life of the process, and on the next
 *     reload `loadEnv` lets `process.env` win over the file. Editing or deleting
 *     a key that already took effect therefore does nothing until the dev server
 *     is restarted — adding a new one works on reload. Verified; it presents as
 *     "I changed `.env` and nothing happened".
 *   - `pnpm start` / the Docker image — the production entry (`server.js`) never
 *     loads vite, so NOTHING reads a file there. Pass real environment variables
 *     instead (`docker run -e MS_AGENT_FRONTEND_HOSTED_MODE=1 …`), which the
 *     entrypoint's subshell inherits.
 *
 * Declared field-by-field in `backend/.env.example`.
 */

/* Not `import.meta.env.SSR`: this module is also pulled in by `vite.config.ts`'s
   dependency graph during SSR builds, where `typeof window` is the check that
   holds everywhere without relying on vite's define step. */
const onServer = typeof window === 'undefined'

/**
 * Hosted deployment: the agent runs on a machine the user has no view of.
 *
 * Controls that ask them to type an absolute path on that machine are hidden,
 * because such a path is one they can neither browse nor verify — the project
 * location and the "directory path" skill-import mode. Anything other than the
 * exact string `1` counts as off, so an empty or unset variable keeps the local
 * behaviour.
 *
 * Frontend-only: the backend does not read this and still accepts a path on the
 * project and skill endpoints. It removes the entry points, not the capability.
 *
 * COMPONENTS MUST NOT import this — it is `false` in the browser. Use
 * `useHosted()` from `hosted.ts`, which reads the value the root loader sent.
 */
export const SERVER_HOSTED_MODE =
  onServer && process.env.MS_AGENT_FRONTEND_HOSTED_MODE === '1'

/**
 * Where server-side code reaches FastAPI.
 *
 * On the server (loaders/SSR) there is no page origin, so relative `/api` paths
 * cannot be fetched and have to be resolved against the backend directly. Empty
 * in the browser, where a same-origin proxy handles routing (vite's in dev,
 * `frontend/server.js` in production) — which is why the empty string is the
 * correct client-side value rather than a fallback.
 *
 * `||`, not `??`: a dotenv line left as bare `MS_AGENT_FRONTEND_API_BASE_URL=`
 * yields an EMPTY STRING, which `??` would accept — leaving the server to fetch a
 * relative `/api` path it has no origin for. Empty means unset here.
 *
 * `vite.config.ts` needs the same value for its dev proxy but cannot read it from
 * here: it is evaluated before the plugin's `config` hook has merged the file, so
 * it calls vite's `loadEnv` itself. The two must agree or SSR and the browser
 * talk to different backends.
 *
 * The default below is what makes this variable easy to get wrong: anything that
 * moves the API off 8000 (`uv run webui`'s port probing, the Docker image) must
 * set it explicitly, or SSR quietly fetches from whatever else is on 8000. It
 * fails soft — a rejected loader degrades to empty data, not an error page — so
 * both of those set it.
 */
export const SERVER_API_BASE = onServer
  ? process.env.MS_AGENT_FRONTEND_API_BASE_URL || 'http://127.0.0.1:8000'
  : ''
