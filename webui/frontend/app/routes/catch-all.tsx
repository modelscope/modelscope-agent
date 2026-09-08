import { redirect } from 'react-router'

/**
 * Catch-all fallback — any URL that matches no route lands on the home page.
 * This is purely the unmatched-URL (404) fallback; the one redirect-only route
 * for a valid destination (`/settings` → models) redirects from its loader the
 * same way, so no redirect is left to a `<Navigate>` element.
 */
export function loader() {
  return redirect('/')
}

export default function CatchAll() {
  return null
}
