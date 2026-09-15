# D:\hermes\plugins\agy-link\tests\test_paths.py
"""paths 模块测试。"""
from pathlib import Path

from agylink import paths


def test_accounts_dir_honours_env(accounts_root):
    assert paths.accounts_dir() == accounts_root
    assert paths.pool_file() == accounts_root / "pool.json"
    assert paths.pending_auth_file() == accounts_root / ".pending-auth.json"


def test_token_file_for_uses_official_layout(tmp_path):
    home = tmp_path / "acc"
    assert paths.token_file_for(home) == home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"


def test_dsh_accounts_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DSH_AGY_ACCOUNTS_DIR", str(tmp_path))
    assert paths.dsh_accounts_dir() == tmp_path
