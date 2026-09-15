# D:\hermes\plugins\agy-link\tests\conftest.py
"""pytest 公共夹具。

把 hermes-agent 仓库根（提供 providers / agent / hermes_constants）和本插件目录
插入 sys.path；把号池目录重定向到临时目录，保证测试不碰真实账号。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
_env_agent = Path(os.environ["HERMES_AGENT"]) if os.environ.get("HERMES_AGENT") else None
HERMES_AGENT_DIR = next(
    (
        path
        for path in (
            _env_agent,
            PLUGIN_DIR.parent.parent / "hermes-agent",
            Path(r"D:\hermes\hermes-agent"),
        )
        if path is not None and (path / "providers").is_dir()
    ),
    PLUGIN_DIR.parent.parent / "hermes-agent",
)

for p in (str(HERMES_AGENT_DIR), str(PLUGIN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(autouse=True)
def _agy_oauth_test_client(monkeypatch):
    """测试用占位客户端，避免用例依赖或泄漏真实 Google OAuth 明文。"""
    monkeypatch.setenv("HERMES_AGY_CLIENT_ID", "test-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("HERMES_AGY_CLIENT_SECRET", "test-client-secret")


@pytest.fixture(autouse=True)
def _clear_transport_signature_cache():
    """每个用例前后清空 transport thoughtSignature 缓存，避免用例间串扰。"""
    from agylink.transport import clear_thought_signature_cache

    clear_thought_signature_cache()
    yield
    clear_thought_signature_cache()


@pytest.fixture
def accounts_root(tmp_path, monkeypatch) -> Path:
    """把号池目录指向 tmp_path/agy-accounts 并返回该路径。"""
    root = tmp_path / "agy-accounts"
    root.mkdir()
    monkeypatch.setenv("HERMES_AGY_ACCOUNTS_DIR", str(root))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return root
