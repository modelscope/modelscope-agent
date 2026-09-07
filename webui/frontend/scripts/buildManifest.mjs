/** Record the exact source inputs and output files of a successful pnpm build. */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

const root = process.cwd()
const hash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex')
const ignored = (rel) => rel.split('/').some((part) => part.startsWith('.')) ||
  rel.startsWith('public/antd/') || rel.startsWith('public/assets/')

function collect(dir, skip = () => false) {
  const files = {}
  if (!fs.existsSync(dir)) return files
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name)
    const rel = path.relative(root, file).split(path.sep).join('/')
    if (skip(rel)) continue
    if (entry.isSymbolicLink()) throw new Error(`Unexpected symlink in build input: ${rel}`)
    if (entry.isDirectory()) Object.assign(files, collect(file, skip))
    else if (entry.isFile()) files[rel] = hash(file)
  }
  return files
}

const inputs = {}
for (const name of ['app', 'scripts', 'public']) {
  Object.assign(inputs, collect(path.join(root, name), ignored))
}
for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
  if (entry.isFile() && !ignored(entry.name) && /\.(?:json|ya?ml|[cm]?js|ts)$/.test(entry.name)) {
    inputs[entry.name] = hash(path.join(root, entry.name))
  }
}
const outputs = collect(path.join(root, 'build'), (rel) => rel === 'build/webui-build.json')
const manifest = JSON.parse(fs.readFileSync('build/client/antd/manifest.json', 'utf8'))
if (!manifest.href?.startsWith('/assets/') || !fs.existsSync(`build/client${manifest.href}`)) {
  throw new Error('The generated Ant Design CSS is missing from build/client')
}
for (const required of ['build/server/index.js', 'build/client/antd/manifest.json']) {
  if (!outputs[required]) throw new Error(`Missing build output: ${required}`)
}
fs.writeFileSync('build/webui-build.json', JSON.stringify({ format: 1, inputs, outputs }, null, 2) + '\n')
console.log(`[webui] Recorded ${Object.keys(inputs).length} build inputs and ${Object.keys(outputs).length} outputs`)
