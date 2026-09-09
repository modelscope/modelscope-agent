import { useRouteLoaderData } from 'react-router'

/**
 * True when this instance is a HOSTED deployment
 * (`MS_AGENT_FRONTEND_HOSTED_MODE=1`).
 *
 * The agent then runs on a machine that is not the user's own and whose
 * filesystem they cannot see, so every control asking them to NAME a directory
 * on it is meaningless: an absolute path they can neither browse nor verify.
 * Those controls are hidden rather than disabled — a greyed-out field would
 * still advertise a capability this deployment does not have.
 *
 * Sourced from the ROOT loader, not `import.meta.env`, for two reasons: one
 * server-side variable then decides it for SSR and the browser alike (no
 * build-time baking, no hydration mismatch), and root sits above both the app
 * and settings layouts — the two path controls live under different ones.
 * The variable itself is declared in `env.ts`; this is the client half.
 *
 * Typed structurally instead of importing `RootData`, exactly as root's own
 * `Layout` reads its data, so no component has to import the route module.
 */
export function useHosted(): boolean {
  const data = useRouteLoaderData('root') as { hosted?: boolean } | undefined
  // Absent only where root's data is out of reach (the error boundary renders
  // without it). Default to the local behaviour: wrongly hiding a control on
  // someone's own machine costs them a feature, while wrongly showing one on a
  // hosted instance is what the flag is for and cannot happen here — root
  // always resolves before anything that calls this.
  return data?.hosted === true
}
