"""The ModelScope warm-up for fastembed's local embedding model.

The contract under test: a warmed cache is indistinguishable from one fastembed
downloaded itself (its cache-first path must resolve offline), an already-warm
cache is never touched again, and every failure degrades to fastembed's own
behaviour instead of raising.
"""
from __future__ import annotations

import pathlib

import pytest

from app.backends.ms_agent import embed_warmup

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "fe-cache"))
    return pathlib.Path(tmp_path / "fe-cache")


def _registry():
    entry = embed_warmup._registry_entry(MODEL)
    assert entry is not None, "default model missing from fastembed registry"
    return entry


def test_registry_resolves_hf_source_and_model_file():
    hf_repo, model_file = _registry()
    # Guards against silent registry drift: the warm-up writes THESE names.
    assert "/" in hf_repo
    assert model_file.endswith(".onnx")


def test_source_switch_skips_everything(cache, monkeypatch):
    called = []
    monkeypatch.setattr(embed_warmup, "_registry_entry",
                        lambda m: called.append(m))
    status = embed_warmup.warm_fastembed_cache(MODEL, "huggingface")
    assert status.startswith("skipped")
    assert not called  # not even a registry lookup


def test_unknown_model_is_skipped(cache):
    status = embed_warmup.warm_fastembed_cache("no/such-model", "modelscope")
    assert status.startswith("skipped")


def _lay_out_snapshot(cache_dir: pathlib.Path, hf_repo: str, model_file: str,
                      revision: str = "test-rev") -> pathlib.Path:
    repo_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / revision
    snap.mkdir(parents=True)
    for name in embed_warmup._SIDE_FILES:
        (snap / name).write_text("{}")
    (snap / model_file).parent.mkdir(parents=True, exist_ok=True)
    (snap / model_file).write_bytes(b"onnx-bytes")
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(revision)
    return snap


def test_warm_cache_short_circuits_without_network(cache, monkeypatch):
    hf_repo, model_file = _registry()
    _lay_out_snapshot(cache, hf_repo, model_file)

    def boom(*a, **k):  # any download attempt is a test failure
        raise AssertionError("network path must not run on a warm cache")

    monkeypatch.setattr(embed_warmup, "_ms_download", boom)
    assert embed_warmup.warm_fastembed_cache(MODEL, "modelscope") == "cached"


def test_incomplete_snapshot_triggers_rewarm(cache, monkeypatch):
    hf_repo, model_file = _registry()
    snap = _lay_out_snapshot(cache, hf_repo, model_file)
    (snap / model_file).unlink()  # torso: model file missing

    def fake_download(model_id, local_dir=None, allow_file_pattern=None, **k):
        staging = pathlib.Path(local_dir)
        for name in embed_warmup._SIDE_FILES:
            (staging / name).write_text("{}")
        variant = embed_warmup._onnx_variant()
        (staging / variant).parent.mkdir(parents=True, exist_ok=True)
        (staging / variant).write_bytes(b"fresh-onnx")
        return str(staging)

    monkeypatch.setattr(embed_warmup, "_ms_download", fake_download)
    assert embed_warmup.warm_fastembed_cache(MODEL, "modelscope") == "warmed"

    # The result must be resolvable through the exact API fastembed's
    # cache-first path uses — offline.
    from huggingface_hub import snapshot_download as hf_dl

    out = pathlib.Path(
        hf_dl(repo_id=hf_repo, cache_dir=str(cache), local_files_only=True))
    assert (out / model_file).read_bytes() == b"fresh-onnx"
    for name in embed_warmup._SIDE_FILES:
        assert (out / name).is_file()
    # Idempotent from here on.
    assert embed_warmup.warm_fastembed_cache(MODEL, "modelscope") == "cached"


def test_download_failure_degrades_instead_of_raising(cache, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("modelscope unreachable")

    monkeypatch.setattr(embed_warmup, "_ms_download", boom)
    status = embed_warmup.warm_fastembed_cache(MODEL, "modelscope")
    assert status.startswith("failed")
    # No half-written snapshot may be left behind.
    hf_repo, _ = _registry()
    repo_dir = cache / f"models--{hf_repo.replace('/', '--')}"
    assert not (repo_dir / "refs" / "main").exists()


def test_bm25_layout_from_vendored_files(cache):
    status = embed_warmup.warm_bm25_cache("modelscope")
    assert status == "warmed"
    entry = embed_warmup._bm25_registry_entry()
    assert entry is not None
    hf_repo, wanted = entry
    from huggingface_hub import snapshot_download as hf_dl

    out = pathlib.Path(
        hf_dl(repo_id=hf_repo, cache_dir=str(cache), local_files_only=True))
    assert (out / "english.txt").is_file()  # the file mem0 actually loads
    words = (out / "english.txt").read_text().split()
    assert "the" in words and len(words) > 100
    # No network involved at any point, and idempotent.
    assert embed_warmup.warm_bm25_cache("modelscope") == "cached"


def test_bm25_respects_source_switch(cache):
    assert embed_warmup.warm_bm25_cache("huggingface").startswith("skipped")
