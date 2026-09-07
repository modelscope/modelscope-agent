import { Radio } from 'antd'
import { useT } from '~/lib/i18n'
import { useTheme } from '~/lib/theme'
import appearanceLight from '~/assets/images/appearance-light.png'
import appearanceDark from '~/assets/images/appearance-dark.png'
import { metaDict, pageTitle } from '~/lib/pageTitle'
import type { Route } from './+types/appearance'
import './appearance.css'

export function meta({ matches }: Route.MetaArgs) {
  const t = metaDict(matches)
  return [{ title: pageTitle(t, t.settings.navAppearance, t.settings.title) }]
}

export default function AppearanceSettings() {
  const { t, lang, setLang } = useT()
  const { pref, setPref } = useTheme()

  return (
    <div className="space-y-8">
      {/* Appearance */}
      <section>
        <div className="mb-4 text-base font-semibold text-msa-text-1">
          {t.settings.appearanceTheme}
        </div>
        <div className="flex flex-wrap gap-4">
          <ThemeCard
            label={t.settings.themeSystem}
            variant="system"
            selected={pref === 'system'}
            onClick={() => setPref('system')}
          />
          <ThemeCard
            label={t.settings.themeLight}
            variant="light"
            selected={pref === 'light'}
            onClick={() => setPref('light')}
          />
          <ThemeCard
            label={t.settings.themeDark}
            variant="dark"
            selected={pref === 'dark'}
            onClick={() => setPref('dark')}
          />
        </div>
      </section>

      {/* Language */}
      <section>
        <div className="mb-4 text-base font-semibold text-msa-text-1">
          {t.settings.appearanceLanguage}
        </div>
        <Radio.Group value={lang} onChange={(e) => setLang(e.target.value)}>
          <Radio value="zh">中文</Radio>
          <Radio value="en">English</Radio>
        </Radio.Group>
      </section>
    </div>
  )
}

function ThemeCard({
  label,
  variant,
  selected,
  onClick
}: {
  label: string
  variant: 'light' | 'dark' | 'system'
  selected: boolean
  onClick: () => void
}) {
  return (
    <div
      className={`w-full cursor-pointer overflow-hidden rounded-xl border-2 transition-all sm:w-[323px] ${
        selected
          ? 'border-msa-purple-5 shadow-sm'
          : 'border-msa-line-1 hover:border-msa-line-3'
      }`}
      onClick={onClick}
    >
      {variant === 'system' ? (
        // Both shots ship; appearance.css picks one by `prefers-color-scheme`.
        // Each is paired with its own label so the label's colour keeps matching
        // the screenshot behind it.
        <>
          <div className="appearance-sys-light">
            <Preview variant="light" label={label} />
          </div>
          <div className="appearance-sys-dark">
            <Preview variant="dark" label={label} />
          </div>
        </>
      ) : (
        <Preview variant={variant} label={label} />
      )}
    </div>
  )
}

/** Design-spec preview: the screenshot with the theme name overlaid near its
 * bottom. Label colors are bound to the IMAGE's own palette (not the active
 * theme) — dark text on the light shot, light text on the dark shot — so they
 * stay readable under either app theme. */
function Preview({
  variant,
  label
}: {
  variant: 'light' | 'dark'
  label: string
}) {
  return (
    <div
      className={`relative ${variant === 'light' ? 'bg-msa-fill-1' : 'bg-[#141414]'}`}
    >
      {/* The PNGs carry their OWN ~24px transparent rounded corners, which scale
          to ~12px on screen — slightly larger than the card's inner corner
          (rounded-xl 12px minus the 2px border = 10px), so the container clip
          can't hide them and the page colour shows through as a pale notch at
          each corner. Rounding the image itself to that same 10px trims the
          transparent wedge away. */}
      <img
        src={variant === 'light' ? appearanceLight : appearanceDark}
        alt=""
        className="block w-full select-none rounded-[12px]"
        draggable={false}
      />
      <span
        className={`absolute inset-x-0 bottom-[10px] text-center text-md font-medium ${
          variant === 'light' ? 'text-msa-text-2' : 'text-msa-text-0'
        }`}
      >
        {label}
      </span>
    </div>
  )
}
