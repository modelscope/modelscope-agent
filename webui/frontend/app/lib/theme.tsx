import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

/** The RESOLVED theme — what components style against. Kept as a two-value type
 * on purpose: it drives the `.dark` class and antd's cssinjs theme (a JS value),
 * neither of which can express "it depends on the OS". */
export type Theme = "light" | "dark";

/** What the user PICKED. `system` defers to the OS and keeps following it. */
export type ThemePref = "system" | Theme;

export const THEME_COOKIE = "ms-agent-webui:theme";
/**
 * Last OS colour scheme this browser reported, mirrored into a cookie by the
 * client. The server cannot read `prefers-color-scheme`, so without this a
 * `system` preference would render light on the server and get corrected after
 * hydration — a dark-mode flash on EVERY page load. With it, SSR renders the
 * scheme the browser last had, so the correction is a no-op in the normal case
 * (only a first-ever visit, or an OS switch made while the app was closed, can
 * still produce one).
 */
export const SCHEME_COOKIE = "ms-agent-webui:scheme";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 365; // 1 year

const DARK_QUERY = "(prefers-color-scheme: dark)";

interface ThemeContextValue {
  /** Resolved theme: `system` already collapsed to light/dark. */
  theme: Theme;
  /** The user's choice, for the settings UI to render as selected. */
  pref: ThemePref;
  setPref: (p: ThemePref) => void;
  /** Flip to the explicit opposite of what is currently showing (leaves
   * `system` behind, since "toggle" implies pinning a choice). */
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function writeCookie(name: string, value: string) {
  if (typeof document === "undefined") return;
  document.cookie = `${name}=${value}; path=/; max-age=${COOKIE_MAX_AGE}; SameSite=Lax`;
}

export function ThemeProvider({
  initialPref = "system",
  initialSystemTheme = "light",
  children,
}: {
  initialPref?: ThemePref;
  /** What the server resolved from the scheme cookie. */
  initialSystemTheme?: Theme;
  children: React.ReactNode;
}) {
  // Both pieces of state must start at what the server rendered, otherwise
  // antd's cssinjs emits different class names per theme and hydration
  // mismatches.
  const [pref, setPrefState] = useState<ThemePref>(initialPref);
  const [systemTheme, setSystemTheme] = useState<Theme>(initialSystemTheme);

  const theme: Theme = pref === "system" ? systemTheme : pref;

  // Track the OS scheme continuously — and persist it even when the preference
  // is light/dark, so that switching to `system` later is already correct on the
  // next server render instead of costing a round trip to learn the scheme.
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia(DARK_QUERY);
    const sync = () => {
      const next: Theme = mq.matches ? "dark" : "light";
      setSystemTheme(next);
      writeCookie(SCHEME_COOKIE, next);
    };
    sync(); // correct a stale/absent cookie on load
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  // Keep the <html> class in sync with the RESOLVED theme — needed for the OS
  // tracking above and any user choice. The server-rendered class is set in
  // root.tsx Layout. This class also drives `color-scheme` (app.css `html` /
  // `html.dark`), which themes browser-drawn UI (scrollbars, form controls,
  // native pickers) — so it must stay a class on <html>, not move to a data
  // attribute or inline style.
  useEffect(() => {
    if (typeof document === "undefined") return;
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  const setPref = useCallback((next: ThemePref) => {
    setPrefState(next);
    writeCookie(THEME_COOKIE, next);
  }, []);

  const toggleTheme = useCallback(
    () => setPref(theme === "dark" ? "light" : "dark"),
    [setPref, theme],
  );

  const value = useMemo(
    () => ({ theme, pref, setPref, toggleTheme }),
    [theme, pref, setPref, toggleTheme],
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside <ThemeProvider>");
  return ctx;
}
