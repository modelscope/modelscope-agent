import { redirect } from 'react-router'

/**
 * `/settings` has no page of its own — it opens on the models tab.
 *
 * The redirect lives in the loader, not in a `<Navigate>` element: SSR renders
 * under a StaticRouter, where navigating during the initial render is a no-op,
 * so a component-level redirect would ship the empty settings shell and only
 * move the user after hydration. From the loader it is a real redirect
 * response, on the server as well as on client-side navigations.
 *
 * Absolute path on purpose — a relative `Location` would resolve against
 * `/settings`, i.e. to `/models`.
 */
export function loader() {
  return redirect('/settings/models')
}

export default function SettingsIndex() {
  return null
}
