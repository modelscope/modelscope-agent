# MS-Agent website

Bilingual project website: [中文](https://modelscope.github.io/ms-agent/) · [English](https://modelscope.github.io/ms-agent/en/).

## Development

Use Node.js 24 and the public npm registry configured in `.npmrc`.

```bash
cd website
npm ci
npm run build
npm run check
npm run preview -- --port 4321
```

Open `http://127.0.0.1:4321/ms-agent/`. For live updates, use `npm run dev -- --port 4321`.

Edit bilingual copy in `src/data/content.ts`, sections in `src/components/`, and styling in `src/styles/global.css`. Media files and captions are in `public/media/`; `manifest.json` records their provenance. The site is static and requires no model credentials or backend service.

## Deployment

The official repository maintains this site on the long-lived `website` branch. Open website pull requests against that branch. `.github/workflows/website-pages.yml` builds and checks pull requests; pushes to `website` build and publish only `website/dist` to GitHub Pages.

Pages uses GitHub Actions as its build source. The `github-pages` environment permits deployment only from the `website` branch. This workflow does not publish Python packages or container images. Website updates do not need to be merged into `main`.

Before publishing, run the build and checks, review both languages on desktop and mobile, and check tabs, navigation, copying, image enlargement, and video playback. Keep local research, capture files, credentials, and review records out of commits and public assets.

See [THIRD_PARTY.md](THIRD_PARTY.md) for licenses and attribution.
