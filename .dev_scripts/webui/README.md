# WebUI build tools

These tools prepare the WebUI resources included in MS-Agent packages and check
packages and images before release. Run the commands below from the repository
root with Python 3.12+, Node.js 22.22.0+, pnpm 10.17.1 and uv installed.

## Build a package

```bash
python .dev_scripts/webui/check_webui_release.py
python .dev_scripts/webui/prepare_webui.py
uv build --python 3.12
uvx --from twine==6.2.0 twine check dist/*.whl dist/*.tar.gz
```

Preparation installs locked frontend dependencies, builds CSS and frontend
output, then records file hashes and the SDK version/commit in
`webui/RESOURCE-MANIFEST.json`. Use `--skip-build` to reuse an unchanged, validated
frontend build. When preparing a source tree without Git metadata, pass the full
SDK commit with `--sdk-sha`.

The wheel includes WebUI source and prebuilt frontend output. The sdist contains
the same resources and this build helper, so rebuilding a wheel from the sdist
does not need Node.js or pnpm. Editable SDK installation does not need a frontend
build; `ms-agent ui` prepares it when first started.

## Package inputs

[`resource-files.txt`](resource-files.txt) lists WebUI source files relative to
`webui/`. Add new runtime or frontend build inputs here when adding files.
Generated frontend outputs are read from the frontend build manifest.
Local configuration, caches and developer notes are not package inputs.

`setup.py` loads [`webui_packaging.py`](webui_packaging.py) directly, and
`MANIFEST.in` includes the helper in the sdist. `.gitignore` controls Git tracking;
package contents are controlled separately by this file list and the build rules.

## Validation tools

| File | Purpose |
| --- | --- |
| `prepare_webui.py` | Build and record WebUI package resources |
| `check_webui_release.py` | Check version, tag and dependencies; validate wheel/sdist and write release checksums |
| `check_webui_install.py` | Exercise an installed wheel outside the checkout, including pages, CSS, streaming, cache reuse and shutdown |
| `check_webui_image.py` | Exercise a built Docker image and record its identity |
| `webui_smoke.py` | Shared HTTP checks used by package and image validation |
| `publish_webui_release.py` | Publish already validated artifacts and reject conflicting existing versions |

The installed-package check needs the Python interpreter from a fresh wheel
environment. Run it from outside the checkout, passing its absolute script path
and `--logs` with a directory for test logs. Image validation uses test containers
and volumes and removes them when finished. Neither check calls a real model.

For exact automated commands, see the [package workflow](../../.github/workflows/webui-check.yaml)
and [image workflow](../../.github/workflows/webui-image.yaml). Publishing requires
configured credentials and a separate action; see the [release guide](../../docs/RELEASING.md).
