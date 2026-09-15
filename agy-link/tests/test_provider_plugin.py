"""agy-oauth ProviderProfile 测试：注册字段、create_client、fetch_models 兜底、核心注册表能看见。"""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from agylink import pool as pool_module
from agylink import quota
from agylink.token_store import TokenSet, write_token

_REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_INIT = next(
    (
        path
        for path in (
            _REPO_ROOT / "agy-oauth" / "__init__.py",
            _REPO_ROOT / "model-providers" / "agy-oauth" / "__init__.py",
        )
        if path.is_file()
    ),
    _REPO_ROOT / "agy-oauth" / "__init__.py",
)


@pytest.fixture
def provider_module(accounts_root):
    spec = importlib.util.spec_from_file_location("agy_oauth_plugin_test", PLUGIN_INIT,
                                                  submodule_search_locations=[str(PLUGIN_INIT.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def profile(provider_module):
    return provider_module.agy_oauth


def _account(accounts_root, account_id, *, auth_required=False, proxy_url=None):
    return pool_module.Account(
        id=account_id,
        alias=account_id,
        dir=str(accounts_root / account_id),
        system_home=False,
        source="hermes",
        email=None,
        enabled=True,
        proxy_url=proxy_url,
        auth_required=auth_required,
        created_at=0,
        last_used_at=0,
        project_id=None,
        cooldowns={},
        quotas={},
    )


def test_profile_fields(profile):
    assert profile.name == "agy-oauth" and "agy" in profile.aliases and "antigravity" in profile.aliases
    assert profile.auth_type == "external_process"
    assert profile.api_mode == "chat_completions"
    assert profile.base_url == "agy://oauth"
    assert profile.process_command == sys.executable
    assert profile.default_aux_model == "gemini-3.6-flash"


def test_create_client_returns_agy_client(profile):
    from agylink.client import AgyOAuthClient
    c = profile.create_client(api_key="agy-oauth", base_url="agy://oauth", command="x", args=[])
    assert isinstance(c, AgyOAuthClient)
    assert c.HERMES_SKIP_TRANSPORT_WRAP and c.HERMES_SKIP_ASYNC_WRAP


def test_fetch_models_falls_back_when_no_account(profile):
    ids = profile.fetch_models(api_key="ignored", base_url=None)
    assert "gemini-3.6-flash" in ids and "claude-sonnet-4-6" in ids


def test_fetch_models_reads_catalog_via_first_usable_account(
    provider_module, accounts_root, monkeypatch
):
    acc_1 = _account(accounts_root, "acc_1", auth_required=True)
    acc_2 = _account(accounts_root, "acc_2", proxy_url="http://proxy.example:8080")
    pool_path = accounts_root / "pool.json"
    store = pool_module.PoolStore(pool_path)
    store.save(pool_module.Pool(accounts=[acc_1, acc_2], active_account_ids={}))
    write_token(
        acc_2.token_path(),
        TokenSet(
            "acc-2-access-token",
            "acc-2-refresh-token",
            datetime.now(timezone.utc) + timedelta(hours=1),
        ),
    )
    authorization_headers = []
    factory_proxies = []

    def handler(request):
        authorization_headers.append(request.headers["Authorization"])
        return httpx.Response(
            200,
            json={"models": {"gemini-3-pro": {}, "claude-sonnet-4-6": {}}},
        )

    monkeypatch.setattr(
        provider_module, "_pool_store", lambda: pool_module.PoolStore(pool_path)
    )
    monkeypatch.setattr(
        provider_module,
        "_catalog_http",
        lambda proxy_url: (
            factory_proxies.append(proxy_url)
            or httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )

    assert provider_module.agy_oauth.fetch_models() == [
        "claude-sonnet-4-6",
        "gemini-3-pro",
    ]
    assert authorization_headers == ["Bearer acc-2-access-token"]
    assert factory_proxies == ["http://proxy.example:8080"]


def test_fetch_models_falls_back_when_catalog_errors(
    provider_module, accounts_root, monkeypatch
):
    account = _account(accounts_root, "acc_1")
    pool_path = accounts_root / "pool.json"
    store = pool_module.PoolStore(pool_path)
    store.save(pool_module.Pool(accounts=[account], active_account_ids={}))
    write_token(
        account.token_path(),
        TokenSet(
            "catalog-error-token",
            "catalog-error-refresh-token",
            datetime.now(timezone.utc) + timedelta(hours=1),
        ),
    )

    def handler(request):
        return httpx.Response(500, json={"error": "catalog unavailable"})

    monkeypatch.setattr(
        provider_module, "_pool_store", lambda: pool_module.PoolStore(pool_path)
    )
    monkeypatch.setattr(
        provider_module,
        "_catalog_http",
        lambda proxy_url: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert provider_module.agy_oauth.fetch_models() == quota.fallback_model_ids()


def test_core_registry_sees_provider(profile):
    from providers import get_provider_profile
    assert get_provider_profile("agy-oauth").name == "agy-oauth"
    assert get_provider_profile("antigravity").name == "agy-oauth"
