"""commands 测试：status 表、import-dsh、remove 守卫、auth 无浏览器模式、auth-code、help。"""
import json
from datetime import datetime, timedelta, timezone

import httpx

from agylink import commands as CMD
from agylink import pool as P
from agylink.oauth import PendingAuth, load_pending, save_pending
from agylink.token_store import TokenSet, read_token, write_token


def _mock(handler):
    return lambda proxy: httpx.Client(transport=httpx.MockTransport(handler))


def test_help_and_unknown():
    assert "auth" in CMD.dispatch(["help"]) and "import-dsh" in CMD.dispatch([])
    assert "未知子命令" in CMD.dispatch(["bogus"])


def test_status_empty_hints_login(accounts_root):
    out = CMD.dispatch(["status"])
    assert "hermes agy auth" in out and "import-dsh" in out


def test_import_dsh_then_status_lists_account(accounts_root, tmp_path, monkeypatch):
    dsh = tmp_path / "dsh"; dsh.mkdir()
    (dsh / "pool.json").write_text(json.dumps({"accounts": [
        {"id": "acc_x", "alias": "备用", "dir": str(dsh / "acc_x"), "email": "u@e.c", "enabled": True,
         "quotas": {"google": {"remainingFraction": 0.5, "weeklyFraction": 0.9}}}]}))
    monkeypatch.setenv("DSH_AGY_ACCOUNTS_DIR", str(dsh))
    out = CMD.dispatch(["import-dsh"])
    assert "导入 1" in out
    st = CMD.dispatch(["status"])
    assert "acc_x" in st and "dsh" in st and "u@e.c" in st and "50%" in st


def test_remove_dsh_only_unregisters(accounts_root, tmp_path, monkeypatch):
    dsh = tmp_path / "dsh"; (dsh / "acc_x").mkdir(parents=True)
    (dsh / "pool.json").write_text(json.dumps({"accounts": [{"id": "acc_x", "dir": str(dsh / "acc_x")}]}))
    monkeypatch.setenv("DSH_AGY_ACCOUNTS_DIR", str(dsh))
    CMD.dispatch(["import-dsh"])
    out = CMD.dispatch(["remove", "acc_x"])
    assert "取消登记" in out and (dsh / "acc_x").exists()
    assert CMD.dispatch(["remove", "acc_x"]).startswith("找不到")


def test_remove_hermes_account_deletes_directory(accounts_root):
    account_dir = accounts_root / "acc_local"
    account_dir.mkdir()
    (account_dir / "marker.txt").write_text("owned", encoding="utf-8")
    account = P.Account(
        id="acc_local",
        alias="local",
        dir=str(account_dir),
        system_home=False,
        source="hermes",
        email=None,
        enabled=True,
        proxy_url=None,
        auth_required=False,
        created_at=0,
        last_used_at=0,
        project_id=None,
        cooldowns={},
        quotas={},
    )
    P.PoolStore(accounts_root / "pool.json").save(
        P.Pool(accounts=[account], active_account_ids={"google": account.id})
    )

    out = CMD.dispatch(["remove", account.id])

    assert "已删除" in out
    assert not account_dir.exists()
    pool = P.PoolStore(accounts_root / "pool.json").load()
    assert pool.accounts == []
    assert "google" not in pool.active_account_ids


def test_auth_without_browser_prints_url_and_saves_pending(accounts_root):
    def reject_network(request):
        raise AssertionError(f"network must not be used: {request.url}")

    out = CMD.dispatch(
        ["auth", "--alias", "工作号"],
        open_browser=False,
        http_factory=_mock(reject_network),
    )
    assert "accounts.google.com" in out and "auth-code" in out
    p = load_pending()
    assert p is not None and p.alias == "工作号" and p.account_dir.startswith(str(accounts_root))


def test_auth_code_completes_login(accounts_root):
    save_pending(PendingAuth("verif", "st", "别名", None, str(accounts_root / "acc_new"), expires_at_ms=2**62))
    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            assert "code_verifier=verif" in req.content.decode()
            return httpx.Response(200, json={"access_token": "ya29.a", "refresh_token": "1//r", "expires_in": 3600})
        if req.url.host == "www.googleapis.com":
            return httpx.Response(200, json={"email": "me@x.y"})
        return httpx.Response(200, json={})
    out = CMD.dispatch(["auth-code", "http://localhost:51121/oauth-callback?code=4%2Fabc&state=st"], http_factory=_mock(handler))
    assert "me@x.y" in out and "ya29" not in out
    pool = P.PoolStore(accounts_root / "pool.json").load()
    assert pool.accounts[0].email == "me@x.y" and pool.accounts[0].source == "hermes"
    assert read_token(pool.accounts[0].token_path()).refresh_token == "1//r"
    assert load_pending() is None


def test_auth_code_rejects_without_pending(accounts_root):
    assert "先运行" in CMD.dispatch(["auth-code", "4/abc"])


def test_models_uses_fallback_without_accounts(accounts_root):
    out = CMD.dispatch(["models"])
    assert "gemini-3.6-flash" in out and "兜底" in out


def test_refresh_quota_updates_pool(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    d = accounts_root / "acc_1"; d.mkdir()
    acc = P.Account(id="acc_1", alias="a", dir=str(d), system_home=False, source="hermes", email=None, enabled=True,
                    proxy_url=None, auth_required=False, created_at=0, last_used_at=0, project_id=None, cooldowns={}, quotas={})
    write_token(acc.token_path(), TokenSet("ya29.t", "1//r", datetime.now(timezone.utc) + timedelta(hours=1)))
    store.save(P.Pool(accounts=[acc], active_account_ids={}))
    def handler(req):
        if req.url.path.endswith("retrieveUserQuotaSummary"):
            return httpx.Response(200, json={"groups": [{"displayName": "Gemini", "buckets": [
                {"window": "5h", "remainingFraction": 0.25, "resetTime": "2026-09-15T05:00:00Z"}]}]})
        return httpx.Response(200, json={"models": {}})
    out = CMD.dispatch(["refresh-quota"], http_factory=_mock(handler))
    assert "acc_1" in out
    assert store.load().accounts[0].quotas["google"]["remainingFraction"] == 0.25


def test_refresh_quota_transport_error_preserves_auth_state(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    account_dir = accounts_root / "acc_1"; account_dir.mkdir()
    account = P.Account(
        id="acc_1", alias="a", dir=str(account_dir), system_home=False,
        source="hermes", email=None, enabled=True, proxy_url=None,
        auth_required=False, created_at=0, last_used_at=0, project_id=None,
        cooldowns={}, quotas={},
    )
    write_token(
        account.token_path(),
        TokenSet("ya29.t", "1//r", datetime.now(timezone.utc) + timedelta(hours=1)),
    )
    store.save(P.Pool(accounts=[account], active_account_ids={}))

    def failing_factory(proxy):
        raise httpx.ConnectTimeout("temporary timeout")

    out = CMD.dispatch(["refresh-quota"], http_factory=failing_factory)
    assert "拉取配额失败" in out
    assert store.load().accounts[0].auth_required is False


def test_refresh_quota_transient_refresh_error_keeps_account(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    account_dir = accounts_root / "acc_1"; account_dir.mkdir()
    account = P.Account(
        id="acc_1", alias="a", dir=str(account_dir), system_home=False,
        source="hermes", email=None, enabled=True, proxy_url=None,
        auth_required=False, created_at=0, last_used_at=0, project_id=None,
        cooldowns={}, quotas={},
    )
    write_token(
        account.token_path(),
        TokenSet("ya29.old", "1//r", datetime.now(timezone.utc) - timedelta(hours=1)),
    )
    store.save(P.Pool(accounts=[account], active_account_ids={}))

    def handler(request):
        assert request.url.host == "oauth2.googleapis.com"
        return httpx.Response(500, json={"error": "server_error"})

    out = CMD.dispatch(["refresh-quota"], http_factory=_mock(handler))

    assert "拉取配额失败" in out
    assert store.load().accounts[0].auth_required is False


def test_refresh_quota_invalid_grant_marks_auth_required(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    account_dir = accounts_root / "acc_1"; account_dir.mkdir()
    account = P.Account(
        id="acc_1", alias="a", dir=str(account_dir), system_home=False,
        source="hermes", email=None, enabled=True, proxy_url=None,
        auth_required=False, created_at=0, last_used_at=0, project_id=None,
        cooldowns={}, quotas={},
    )
    write_token(
        account.token_path(),
        TokenSet("ya29.old", "1//r", datetime.now(timezone.utc) - timedelta(hours=1)),
    )
    store.save(P.Pool(accounts=[account], active_account_ids={}))

    def handler(request):
        assert request.url.host == "oauth2.googleapis.com"
        return httpx.Response(400, json={"error": "invalid_grant"})

    CMD.dispatch(["refresh-quota"], http_factory=_mock(handler))

    assert store.load().accounts[0].auth_required is True


def test_auth_oauth_error_clears_pending(accounts_root, monkeypatch):
    monkeypatch.setattr(CMD.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(
        CMD.oauth,
        "wait_for_loopback_code",
        lambda state, timeout_s: (_ for _ in ()).throw(CMD.oauth.OAuthError("denied")),
    )

    out = CMD.dispatch(["auth"], open_browser=True)

    assert "授权失败" in out
    assert load_pending() is None
