"""Pre-warm fastembed's model cache from ModelScope instead of HuggingFace.

The local embedding mode runs on fastembed, whose model table maps our default
model to an ONNX conversion hosted ONLY on HuggingFace — on networks where
huggingface.co is unreachable, local vector memory simply cannot be set up.
ModelScope hosts the same model (the official sentence-transformers mirror,
complete with the standard ``onnx/`` exports), so this module downloads the
needed files from there and lays them out the way fastembed expects.

Why this works with zero changes to fastembed or mem0: ``download_model``
ALWAYS tries the local cache first (``local_files_only=True``) and returns on a
hit before any network I/O — so a warmed cache short-circuits the HuggingFace
path entirely, and a failed warm-up degrades to exactly today's behaviour
(fastembed goes to HuggingFace itself).

The cache layout is huggingface_hub's on-disk contract (stable for years):

    <cache>/models--<org>--<repo>/refs/main            -> "<revision>"
    <cache>/models--<org>--<repo>/snapshots/<revision>/<files>

We write a new revision directory and flip ``refs/main`` last, so a crashed
warm-up can never leave a half-readable snapshot behind.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
from pathlib import Path

logger = logging.getLogger("app.ms_agent.embed_warmup")

#: refs/main value for snapshots this module lays out. Versioned so a future
#: layout change can re-warm by bumping the suffix.
_REVISION = "modelscope-mirror-v1"

#: The ModelScope repo the files come from: the official sentence-transformers
#: mirror, which carries the standard ``onnx/`` exports of the same weights the
#: HuggingFace conversion was made from.
_MS_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

#: Tokenizer/config files fastembed needs next to the model file. Mirrors the
#: ``allow_patterns`` list in fastembed's own downloader.
_SIDE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def fastembed_cache_dir() -> Path:
    """The shared fastembed cache, exported for every later consumer.

    fastembed's built-in default lands in a TEMP directory (wiped on reboot,
    one copy per tmpdir); models are immutable machine-wide data, so default to
    ``~/.cache/fastembed`` instead. Setting the env var here means mem0's own
    ``TextEmbedding(...)`` — constructed later, in-process, out of our hands —
    resolves the same cache. A user-set FASTEMBED_CACHE_PATH always wins.
    """
    path = os.environ.get("FASTEMBED_CACHE_PATH")
    if not path:
        path = str(Path.home() / ".cache" / "fastembed")
        os.environ["FASTEMBED_CACHE_PATH"] = path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _registry_entry(model: str) -> tuple[str, str] | None:
    """``(hf_repo, model_file)`` fastembed's table maps ``model`` to.

    Read from fastembed itself rather than hardcoded, so a registry change
    (new source repo, renamed onnx file) shifts the warm-up target with it.
    """
    try:
        from fastembed import TextEmbedding

        for m in TextEmbedding.list_supported_models():
            if m.get("model") == model:
                hf = (m.get("sources") or {}).get("hf")
                model_file = m.get("model_file")
                if hf and model_file:
                    return str(hf), str(model_file)
    except Exception:
        logger.debug("fastembed registry lookup failed", exc_info=True)
    return None


def _onnx_variant() -> str:
    """The ModelScope-side ONNX file matching this CPU.

    The mirror ships several exports; pick the int8 quantization built for the
    local architecture (118 MB — smaller than the HuggingFace conversion), and
    fall back to the universal fp32 export anywhere unrecognized.
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    if machine in ("x86_64", "amd64"):
        return "onnx/model_quint8_avx2.onnx"
    return "onnx/model.onnx"


def _ms_download(model_id: str, local_dir: str,
                 allow_file_pattern: list[str]) -> str:
    """The ModelScope download call, one seam wide so tests can replace it."""
    from modelscope.hub.snapshot_download import \
        snapshot_download as ms_snapshot_download

    return ms_snapshot_download(
        model_id, local_dir=local_dir, allow_file_pattern=allow_file_pattern)


def _snapshot_ready(repo_dir: Path, model_file: str) -> bool:
    """Whether refs/main points at a snapshot that has every needed file."""
    try:
        rev = (repo_dir / "refs" / "main").read_text().strip()
    except OSError:
        return False
    snap = repo_dir / "snapshots" / rev
    needed = list(_SIDE_FILES) + [model_file]
    return all((snap / name).is_file() and (snap / name).stat().st_size > 0
               for name in needed)


def _bm25_registry_entry() -> tuple[str, list[str]] | None:
    """``(hf_repo, files)`` for mem0's BM25 sparse encoder, from fastembed."""
    try:
        from fastembed import SparseTextEmbedding

        for m in SparseTextEmbedding.list_supported_models():
            if m.get("model") == "Qdrant/bm25":
                hf = (m.get("sources") or {}).get("hf")
                files = list(m.get("additional_files") or [])
                if hf and files:
                    return str(hf), files
    except Exception:
        logger.debug("fastembed sparse registry lookup failed", exc_info=True)
    return None


def warm_bm25_cache(source: str = "modelscope") -> str:
    """Lay out the BM25 stopword files from the VENDORED copies; never raises.

    mem0's qdrant store encodes a BM25 sparse vector next to every dense one
    (hybrid search) via ``SparseTextEmbedding("Qdrant/bm25")`` — not a neural
    model, just Snowball stopword lists, but they too live only on HuggingFace
    and ModelScope carries no mirror. Without them the store silently degrades
    to dense-only on HF-unreachable networks. The lists are ~25 KB of plain
    text, so they ship with this package (``data/bm25_stopwords``, provenance
    in its README) and this only copies files — no network at all.
    """
    if source != "modelscope":
        return f"skipped: source={source}"
    entry = _bm25_registry_entry()
    if entry is None:
        return "skipped: bm25 not in fastembed registry"
    hf_repo, wanted = entry

    data_dir = Path(__file__).parent / "data" / "bm25_stopwords"
    have = [n for n in wanted if (data_dir / n).is_file()]
    if not have:
        return "skipped: no vendored stopword files"

    cache = fastembed_cache_dir()
    repo_dir = cache / f"models--{hf_repo.replace('/', '--')}"
    try:
        rev = (repo_dir / "refs" / "main").read_text().strip()
        snap = repo_dir / "snapshots" / rev
        if all((snap / n).is_file() for n in have):
            return "cached"
    except OSError:
        pass

    try:
        snap = repo_dir / "snapshots" / _REVISION
        snap.mkdir(parents=True, exist_ok=True)
        for name in have:
            shutil.copy2(data_dir / name, snap / name)
        refs = repo_dir / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        tmp_ref = refs / "main.tmp"
        tmp_ref.write_text(_REVISION)
        tmp_ref.replace(refs / "main")
        logger.info("bm25 stopwords laid out from the vendored copies "
                    "(%d files)", len(have))
        return "warmed"
    except Exception as exc:
        logger.warning("bm25 stopword layout failed (%s); hybrid search may "
                       "degrade to dense-only", exc)
        return f"failed: {exc}"


def warm_local_memory_models(model: str, source: str = "modelscope") -> None:
    """Everything the local memory chain loads, warmed in one call.

    The dense embedder comes from ModelScope, the BM25 stopwords from the
    vendored copies. Both are idempotent and never raise — callers treat this
    as fire-and-forget ahead of constructing fastembed/mem0 objects.
    """
    warm_fastembed_cache(model, source)
    warm_bm25_cache(source)


def warm_fastembed_cache(model: str, source: str = "modelscope") -> str:
    """Make ``model`` loadable offline; returns a status string, never raises.

    - ``"cached"``: the cache already serves the model — nothing to do. This
      also protects a model previously downloaded from HuggingFace: an existing
      store keeps embedding with the exact weights it was built with.
    - ``"warmed"``: files fetched from ModelScope and laid out.
    - ``"skipped: ..."`` / ``"failed: ..."``: warm-up did not run or did not
      finish; fastembed proceeds with its own (HuggingFace) path, so this is
      never worse than the behaviour before this module existed.
    """
    if source != "modelscope":
        return f"skipped: source={source}"
    entry = _registry_entry(model)
    if entry is None:
        return f"skipped: {model!r} not in fastembed registry"
    hf_repo, model_file = entry

    cache = fastembed_cache_dir()
    repo_dir = cache / f"models--{hf_repo.replace('/', '--')}"
    if _snapshot_ready(repo_dir, model_file):
        return "cached"

    variant = _onnx_variant()
    staging = repo_dir / f".staging-{_REVISION}"
    try:
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        _ms_download(
            _MS_REPO,
            local_dir=str(staging),
            allow_file_pattern=list(_SIDE_FILES) + [variant],
        )
        for name in _SIDE_FILES:
            if not (staging / name).is_file():
                raise FileNotFoundError(f"{name} missing from {_MS_REPO}")
        onnx_src = staging / variant
        if not onnx_src.is_file():
            raise FileNotFoundError(f"{variant} missing from {_MS_REPO}")

        # Assemble the snapshot in place, then flip refs/main last: readers
        # either see the old state or the complete new one, never a torso.
        snap = repo_dir / "snapshots" / _REVISION
        snap.mkdir(parents=True, exist_ok=True)
        for name in _SIDE_FILES:
            shutil.move(str(staging / name), str(snap / name))
        # fastembed knows the model file by the HuggingFace conversion's name.
        (snap / model_file).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(onnx_src), str(snap / model_file))
        refs = repo_dir / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        tmp_ref = refs / "main.tmp"
        tmp_ref.write_text(_REVISION)
        tmp_ref.replace(refs / "main")
        logger.info("fastembed cache warmed from ModelScope: %s (%s)", model,
                    variant)
        return "warmed"
    except Exception as exc:
        logger.warning(
            "fastembed warm-up from ModelScope failed (%s); falling back to "
            "fastembed's own HuggingFace download", exc)
        return f"failed: {exc}"
    finally:
        shutil.rmtree(staging, ignore_errors=True)
