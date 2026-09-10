// Static-site foundation adapted from AstroWind, pinned in THIRD_PARTY.md.
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://modelscope.github.io',
  base: '/ms-agent',
  trailingSlash: 'always',
  output: 'static',
  integrations: [sitemap()],
  vite: { plugins: [tailwindcss()] },
});
