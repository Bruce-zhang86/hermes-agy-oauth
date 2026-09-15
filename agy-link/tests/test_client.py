"""AgyOAuthClient 测试：项目发现、刷新写回、429 换号重试一次、无号可用、invalid_grant。"""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from agylink import client as C
from agylink import pool as P
from agylink.token_store import TokenSet, read_token, write_token


def _acc(root, i, **kw):
    d = root / f"acc_{i}"
    d.mkdir(parents=True, exist_ok=True)
    a = P.Account(id=f"acc_{i}", alias=f"a{i}", dir=str(d), system_home=False, source="hermes", email=None,
                  enabled=True, proxy_url=None, auth_required=False, created_at=0, last_used_at=0,
                  project_id=None, cooldowns={}, quotas={})
    a.__dict__.update(kw)
    return a


def _seed(root, ids, *, expired=False):
    store = P.PoolStore(root / "pool.json")
    pool = P.Pool(accounts=[_acc(root, i) for i in ids], active_account_ids={})
    exp = datetime.now(timezone.utc) + (timedelta(seconds=-5) if expired else timedelta(hours=1))
    for a in pool.accounts:
        write_token(a.token_path(), TokenSet(f"ya29.{a.id}", f"1//{a.id}", exp))
    store.save(pool)
    return store


def _sse(*objs):
    body = "".join(f"data: {json.dumps(o)}\n\n" for o in objs)
    return httpx.Response(200, content=body.encode(), headers={"Content-Type": "text/event-stream"})


def _ok_text(text):
    return _sse({"response": {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                              "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}})


def test_project_discovery_then_stream_and_project_cached(accounts_root):
    store = _seed(accounts_root, [1])
    seen = []

    def handler(req):
        seen.append((req.url.path, req.url.host))
        if req.url.path.endswith("loadCodeAssist"):
            assert json.loads(req.content) == {}
            return httpx.Response(200, json={"cloudaicompanionProject": "proj-1"})
        env = json.loads(req.content)
        assert env["project"] == "proj-1" and env["model"] == "gemini-3-flash"
        assert req.headers["Authorization"] == "Bearer ya29.acc_1"
        return _ok_text("hi")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "hi"
    assert store.load().accounts[0].project_id == "proj-1"
    cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "y"}])
    assert sum(1 for p, _ in seen if p.endswith("loadCodeAssist")) == 1


def test_stream_true_returns_chunks(accounts_root):
    store = _seed(accounts_root, [1])

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        return _ok_text("streamed")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    chunks = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}], stream=True)
    assert chunks[0].choices[0].delta.content == "streamed"
    assert chunks[-1].usage.total_tokens == 2


def test_expired_token_is_refreshed_and_written_back(accounts_root):
    store = _seed(accounts_root, [1], expired=True)

    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "ya29.fresh", "expires_in": 3600})
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        assert req.headers["Authorization"] == "Bearer ya29.fresh"
        return _ok_text("ok")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    tok = read_token(store.load().accounts[0].token_path())
    assert tok.access_token == "ya29.fresh" and tok.refresh_token == "1//acc_1"


def test_401_refreshes_once_and_retries_same_account(accounts_root):
    store = _seed(accounts_root, [1])
    token_posts = []
    stream_bearers = []

    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            token_posts.append(req.url.path)
            return httpx.Response(200, json={"access_token": "ya29.fresh-401", "expires_in": 3600})
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        stream_bearers.append(req.headers["Authorization"])
        if len(stream_bearers) == 1:
            return httpx.Response(401, text="expired bearer")
        return _ok_text("retried")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "retried"
    assert token_posts == ["/token"]
    assert stream_bearers == ["Bearer ya29.acc_1", "Bearer ya29.fresh-401"]
    assert read_token(store.load().accounts[0].token_path()).access_token == "ya29.fresh-401"


def test_401_twice_marks_auth_required_and_advances(accounts_root):
    store = _seed(accounts_root, [1, 2])
    stream_bearers = []

    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            return httpx.Response(200, json={"access_token": "ya29.fresh-401", "expires_in": 3600})
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        bearer = req.headers["Authorization"]
        stream_bearers.append(bearer)
        if bearer in {"Bearer ya29.acc_1", "Bearer ya29.fresh-401"}:
            return httpx.Response(401, text="still unauthorized")
        return _ok_text("from-next")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "from-next"
    pool = store.load()
    assert pool.accounts[0].auth_required is True
    assert pool.active_account_ids["google"] == "acc_2"
    assert stream_bearers == ["Bearer ya29.acc_1", "Bearer ya29.fresh-401", "Bearer ya29.acc_2"]


def test_invalid_grant_marks_auth_required_and_uses_next(accounts_root):
    store = _seed(accounts_root, [1, 2], expired=True)

    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            form = dict(p.split("=", 1) for p in req.content.decode().split("&"))
            if "acc_1" in form["refresh_token"]:
                return httpx.Response(400, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "ya29.two", "expires_in": 3600})
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        return _ok_text("from-2")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "from-2"
    pool = store.load()
    assert pool.accounts[0].auth_required is True and pool.active_account_ids["google"] == "acc_2"


def test_refresh_server_error_advances_without_auth_required(accounts_root):
    store = _seed(accounts_root, [1, 2], expired=True)

    def handler(req):
        if req.url.host == "oauth2.googleapis.com":
            form = dict(p.split("=", 1) for p in req.content.decode().split("&"))
            if "acc_1" in form["refresh_token"]:
                return httpx.Response(500, text="temporary token endpoint failure")
            return httpx.Response(200, json={"access_token": "ya29.two", "expires_in": 3600})
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        return _ok_text("from-2")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "from-2"
    pool = store.load()
    assert pool.accounts[0].auth_required is False
    assert pool.active_account_ids["google"] == "acc_2"


def test_missing_account_home_marks_auth_required_and_advances(accounts_root):
    store = _seed(accounts_root, [1, 2])
    pool = store.load()
    pool.accounts[0].dir = ""
    store.save(pool)

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        return _ok_text("from-valid-home")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "from-valid-home"
    pool = store.load()
    assert pool.accounts[0].auth_required is True
    assert pool.active_account_ids["google"] == "acc_2"


def test_env_project_id_is_used_but_not_cached(accounts_root, monkeypatch):
    store = _seed(accounts_root, [1])
    pool = store.load()
    pool.accounts[0].project_id = ""
    store.save(pool)
    monkeypatch.setenv("HERMES_AGY_PROJECT_ID", "env-project")
    projects = []

    def handler(req):
        assert not req.url.path.endswith("loadCodeAssist")
        projects.append(json.loads(req.content)["project"])
        return _ok_text("env-project-ok")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "env-project-ok"
    assert projects == ["env-project"]
    assert store.load().accounts[0].project_id == ""


@pytest.mark.parametrize(
    "discovery_case",
    ["server-error", "missing-project"],
    ids=["server-error", "missing-project"],
)
def test_project_discovery_failure_stops_before_stream(accounts_root, discovery_case):
    store = _seed(accounts_root, [1])
    stream_hits = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            if discovery_case == "server-error":
                return httpx.Response(500, text="catalog unavailable")
            return httpx.Response(200, json={})
        stream_hits.append(req.url.path)
        return _ok_text("must-not-run")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(C.NoAccountAvailable, match="无法发现 Cloud Code 项目"):
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert stream_hits == []


def test_429_cools_family_and_retries_once_on_next_account(accounts_root):
    store = _seed(accounts_root, [1, 2])
    hits = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        token = req.headers["Authorization"]
        hits.append(token)
        if token.endswith("acc_1"):
            return httpx.Response(429, json={"error": {"status": "RESOURCE_EXHAUSTED"}})
        return _ok_text("second")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    out = cl.chat.completions.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "second"
    pool = store.load()
    assert P.is_cooling(pool.accounts[0], "anthropic", now_ms=cl._now_ms())
    assert not P.is_cooling(pool.accounts[0], "google", now_ms=cl._now_ms())
    assert pool.active_account_ids["anthropic"] == "acc_2"
    assert len(hits) == 2


def test_429_on_all_accounts_raises_after_single_retry(accounts_root):
    store = _seed(accounts_root, [1, 2, 3])
    hits = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        hits.append(1)
        return httpx.Response(429, text="RESOURCE_EXHAUSTED")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(C.UpstreamError):
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert len(hits) == 2


def test_429_without_reset_waits_once_and_retries_same_account(accounts_root):
    store = _seed(accounts_root, [1])
    hits = []
    sleeps = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        hits.append(req.headers["Authorization"])
        if len(hits) == 1:
            return httpx.Response(429, text="RESOURCE_EXHAUSTED")
        return _ok_text("after-short-wait")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    cl._sleep = sleeps.append
    out = cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert out.choices[0].message.content == "after-short-wait"
    assert len(hits) == 2
    assert sleeps == [15.0]
    assert store.load().accounts[0].cooldowns.get("google") is None


def test_two_consecutive_429_leave_cooldown_persisted(accounts_root):
    store = _seed(accounts_root, [1])
    sleeps = []
    hits = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        hits.append(req.headers["Authorization"])
        return httpx.Response(429, text="RESOURCE_EXHAUSTED")

    cl = C.AgyOAuthClient(
        store=store,
        http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)),
        now_ms=lambda: 1_000,
    )
    cl._sleep = sleeps.append
    with pytest.raises(C.UpstreamError):
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert len(hits) == 2
    assert sleeps == [15.0]
    assert store.load().accounts[0].cooldowns["google"]["untilMs"] > cl._now_ms()


def test_429_with_reset_time_does_not_wait(accounts_root):
    store = _seed(accounts_root, [1])
    sleeps = []
    reset_time = "2099-01-01T00:00:00Z"

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        return httpx.Response(
            429,
            json={"error": {"status": "RESOURCE_EXHAUSTED", "resetTime": reset_time}},
        )

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    cl._sleep = sleeps.append
    with pytest.raises(C.UpstreamError):
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert sleeps == []
    assert store.load().accounts[0].cooldowns["google"]["untilMs"] > cl._now_ms()


def test_other_4xx_does_not_try_second_host(accounts_root):
    store = _seed(accounts_root, [1])
    stream_hits = []

    def handler(req):
        if req.url.path.endswith("loadCodeAssist"):
            return httpx.Response(200, json={"cloudaicompanionProject": "p"})
        stream_hits.append(req.url.host)
        return httpx.Response(418, text="client error")

    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(C.UpstreamError) as exc_info:
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert exc_info.value.status == 418
    assert len(stream_hits) == 1


def test_no_account_raises_with_hint(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client())
    with pytest.raises(C.NoAccountAvailable) as ei:
        cl.chat.completions.create(model="gemini-3-flash", messages=[{"role": "user", "content": "x"}])
    assert "hermes agy auth" in str(ei.value)
    assert isinstance(ei.value, RuntimeError)


def test_unknown_family_rejected(accounts_root):
    store = _seed(accounts_root, [1])
    cl = C.AgyOAuthClient(store=store, http_factory=lambda proxy: httpx.Client())
    with pytest.raises(ValueError):
        cl.chat.completions.create(model="llama-3", messages=[{"role": "user", "content": "x"}])


def test_dsh_account_refresh_writes_to_original_directory(accounts_root, tmp_path):
    dsh_dir = tmp_path / "dsh-original"
    dsh_dir.mkdir()
    account = _acc(accounts_root, "dsh", dir=str(dsh_dir), source="dsh")
    expired = datetime.now(timezone.utc) - timedelta(seconds=1)
    write_token(account.token_path(), TokenSet("ya29.old-dsh", "1//dsh-refresh", expired))

    def handler(req):
        assert req.url.host == "oauth2.googleapis.com"
        return httpx.Response(200, json={"access_token": "ya29.new-dsh", "expires_in": 3600})

    cl = C.AgyOAuthClient(http_factory=lambda proxy: httpx.Client(transport=httpx.MockTransport(handler)))
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        assert cl._access_token(account, http) == "ya29.new-dsh"
    assert read_token(account.token_path()).access_token == "ya29.new-dsh"
    assert not (accounts_root / "dsh" / ".config" / "antigravity" / "antigravity-oauth-token").exists()
