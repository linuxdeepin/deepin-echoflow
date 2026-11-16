"""Hermetic tests for kaggle_harness HF-token resolution (F9).

No network, no Kaggle, no real /kaggle/input: each test builds a fake dataset
mount tree in a tempdir and points the harness at it via the KAGGLE_INPUT_ROOT
environment variable (the harness's test seam; defaults to /kaggle/input on a
real worker).

Why these tests exist: Kaggle mounts attached datasets under TWO layouts
depending on the worker environment --
    /kaggle/input/<slug>/                     (classic, "short")
    /kaggle/input/datasets/<owner>/<slug>/    (newer, "long")
A resolver that only scans the short layout returns no token on a long-layout
worker, and every HF upload 401s AFTER the compute finished (this burned a full
21-minute imatrix run, CrispEmbed T19-E3 run 1). resolve_hf_token(require=True)
must therefore abort UP FRONT when no token can be found.

Write-the-guard-first verification hooks (used to prove this suite detects the
original defect):
  KH_PATH            -- path to an alternate kaggle_harness.py to test
                        (e.g. a pre-fix copy with its hard-coded /kaggle/input
                        literals rewritten to a fixed temp root).
  KH_TEST_FIXED_ROOT -- use this fixed directory as the fake input root instead
                        of pytest's per-test tmp_path (needed when the module
                        under test has a baked-in root instead of reading
                        KAGGLE_INPUT_ROOT). Wiped and recreated per test.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

HARNESS_PATH = Path(
    os.environ.get(
        "KH_PATH",
        str(Path(__file__).resolve().parents[1] / "tools" / "kaggle" / "kaggle_harness.py"),
    )
)

TOK_SHORT = "hf_short_dummy_token_0123456789"
TOK_LONG = "hf_long_dummy_token_0123456789"


@pytest.fixture()
def kh(monkeypatch):
    """Freshly import the harness under test with a clean HF environment."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_ENABLE_HF_TRANSFER"):
        monkeypatch.delenv(var, raising=False)
    spec = importlib.util.spec_from_file_location("kh_under_test", HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kh_under_test"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("kh_under_test", None)


def fake_input_root(monkeypatch, tmp_path: Path) -> Path:
    """Create an empty fake /kaggle/input and point the harness at it."""
    fixed = os.environ.get("KH_TEST_FIXED_ROOT")
    if fixed:
        root = Path(fixed)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
    else:
        root = tmp_path / "kaggle" / "input"
        root.mkdir(parents=True)
    monkeypatch.setenv("KAGGLE_INPUT_ROOT", str(root))
    return root


def put_token(root: Path, rel: str, token: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token + "\n")
    return p


# ── (1) classic short layout: /kaggle/input/<slug>/hf_token.txt ─────────────

def test_short_path_layout(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/hf_token.txt", TOK_SHORT)
    assert kh.kaggle_token_from_dataset() == TOK_SHORT


def test_short_path_layout_nonstandard_slug(kh, monkeypatch, tmp_path):
    # Behavior parity with the historical resolver: ANY mounted dataset dir is
    # probed for the token file, not just the crispasr-hf-token slug.
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "my-other-token-ds/hf_token.txt", TOK_SHORT)
    assert kh.kaggle_token_from_dataset() == TOK_SHORT


# ── (2) long layout: /kaggle/input/datasets/<owner>/<slug>/hf_token.txt ─────
# This is the layout the T19-E3 run-1 worker had ("/kaggle/input contains
# 1 entries: ['datasets']") and the one the pre-fix resolver missed.

def test_long_path_layout_someacct(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "datasets/someacct/crispasr-hf-token/hf_token.txt", TOK_LONG)
    assert kh.kaggle_token_from_dataset() == TOK_LONG


def test_long_path_layout_any_owner(kh, monkeypatch, tmp_path):
    # Owner-agnostic: no hard-coded account names.
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "datasets/someoneelse/their-hf-token/hf_token.txt", TOK_LONG)
    assert kh.kaggle_token_from_dataset() == TOK_LONG


# ── (3) both layouts present: short path preferred (t19 driver precedence) ──

def test_both_layouts_short_preferred(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/hf_token.txt", TOK_SHORT)
    put_token(root, "datasets/someacct/crispasr-hf-token/hf_token.txt", TOK_LONG)
    assert kh.kaggle_token_from_dataset() == TOK_SHORT


# ── (4) no token anywhere: loud, early failure ──────────────────────────────

def test_no_token_returns_none(kh, monkeypatch, tmp_path):
    fake_input_root(monkeypatch, tmp_path)  # empty tree
    assert kh.kaggle_token_from_dataset() is None


def test_resolve_no_token_is_falsy_before_any_upload(kh, monkeypatch, tmp_path):
    # The resolver itself is the early tripwire: a falsy return means the
    # caller can abort BEFORE burning GPU time, not at the first upload 401.
    fake_input_root(monkeypatch, tmp_path)
    assert not kh.resolve_hf_token()
    assert "HF_TOKEN" not in os.environ


def test_resolve_require_aborts_up_front(kh, monkeypatch, tmp_path):
    # Mirrors the t19 driver: `raise SystemExit(...)` when no token is found,
    # so an uploading kernel dies in its first seconds instead of losing a
    # 21-minute run's artifacts to 401s.
    fake_input_root(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        kh.resolve_hf_token(require=True)


def test_resolve_require_passes_when_token_present(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "datasets/someacct/crispasr-hf-token/hf_token.txt", TOK_LONG)
    assert kh.resolve_hf_token(require=True) == TOK_LONG


# ── behavior parity: env exports and 3-tier precedence unchanged ────────────

def test_resolve_exports_env_on_success(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/hf_token.txt", TOK_SHORT)
    assert kh.resolve_hf_token() == TOK_SHORT
    assert os.environ["HF_TOKEN"] == TOK_SHORT
    assert os.environ["HUGGING_FACE_HUB_TOKEN"] == TOK_SHORT
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


def test_resolve_env_var_wins_over_dataset(kh, monkeypatch, tmp_path):
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/hf_token.txt", TOK_SHORT)
    monkeypatch.setenv("HF_TOKEN", "hf_env_dummy_token_0123456789")
    assert kh.resolve_hf_token() == "hf_env_dummy_token_0123456789"


def test_alternate_token_filenames_still_probed(kh, monkeypatch, tmp_path):
    # Historical behavior: `token` / `access_token` filenames are accepted too.
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/token", TOK_SHORT)
    assert kh.kaggle_token_from_dataset() == TOK_SHORT


def test_short_empty_token_falls_through_to_long(kh, monkeypatch, tmp_path):
    # An empty/short token file must not shadow a real one elsewhere.
    root = fake_input_root(monkeypatch, tmp_path)
    put_token(root, "crispasr-hf-token/hf_token.txt", "")
    put_token(root, "datasets/someacct/crispasr-hf-token/hf_token.txt", TOK_LONG)
    assert kh.kaggle_token_from_dataset() == TOK_LONG
