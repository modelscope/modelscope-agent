import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import assert from 'node:assert/strict';

const root = path.resolve('dist');
const base = '/ms-agent/';
const pages = ['index.html', 'en/index.html'];
const failures = [];
const seen = new Set();
const inlineScripts = new Set();
for (const page of pages) {
  const html = await readFile(path.join(root, page), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  assert.equal(new Set(ids).size, ids.length, `${page}: duplicate element IDs`);
  assert.match(html, /<html lang="(?:zh-CN|en)"/);
  assert.equal((html.match(/<h1\b/g) || []).length, 1, `${page}: exactly one main heading`);
  assert.equal((html.match(/class="install-code"/g) || []).length, 3, `${page}: three installation snippets`);
  for (const script of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g)) {
    if (script[1].trim()) inlineScripts.add(script[1]);
  }
  assert.ok(!/\/Users\/|registry\.anpm|sk-[a-zA-Z0-9]{24,}/.test(html), 'Private data in public HTML');
  for (const match of html.matchAll(/(?:href|src|poster|data-src|data-frame)="([^"]+)"/g)) {
    const raw = match[1].replaceAll('&amp;', '&');
    if (raw.startsWith('data:') || raw.startsWith('mailto:')) continue;
    const url = new URL(raw, `https://modelscope.github.io${base}${page.replace('index.html', '')}`);
    if (url.origin !== 'https://modelscope.github.io') continue;
    assert.ok(url.pathname.startsWith(base), `${page}: missing GitHub Pages base: ${raw}`);
    const local = url.pathname.slice(base.length) || 'index.html';
    const file = local.endsWith('/') ? `${local}index.html` : local;
    const key = `${file}${url.hash}`;
    if (seen.has(key)) continue;
    seen.add(key);
    try {
      await stat(path.join(root, file));
      if (url.hash && file.endsWith('.html')) {
        const target = await readFile(path.join(root, file), 'utf8');
        assert.ok(target.includes(`id="${decodeURIComponent(url.hash.slice(1))}"`), `Missing anchor: ${raw}`);
      }
    } catch (error) {
      failures.push(`${page}: ${raw} (${error.message})`);
    }
  }
}

let scriptGzip = 0;
for (const script of inlineScripts) scriptGzip += gzipSync(script).byteLength;
for (const name of await readdir(path.join(root, '_astro'))) {
  if (name.endsWith('.js'))
    scriptGzip += gzipSync(await readFile(path.join(root, '_astro', name))).byteLength;
}
assert.ok(scriptGzip < 30_000, `Client scripts exceed the 30 kB gzip budget: ${scriptGzip}`);
assert.ok((await stat(path.join(root, 'media/webui-demo.mp4'))).size < 12_000_000, 'Video exceeds 12 MB');
const manifest = JSON.parse(await readFile(path.join(root, 'media/manifest.json'), 'utf8'));
for (const item of manifest.assets) await stat(path.join(root, 'media', item.file));
assert.deepEqual(failures, [], failures.join('\n'));
console.log(
  `Validated ${pages.length} language pages, ${seen.size} local targets, media provenance and asset budgets.`,
);
console.log(`Client JavaScript: ${(scriptGzip / 1024).toFixed(1)} KiB gzip.`);
