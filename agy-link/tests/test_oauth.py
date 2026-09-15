"""oauth 模块测试：PKCE、授权 URL、换票、刷新、pending、回环。"""
import socket
import threading
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from agylink import oauth
from agylink.paths import pending_auth_file
from agylink.oauth import (
    InvalidGrant,
    LoopbackBusy,
    PendingAuth,
    build_authorize_url,
    clear_pending,
    exchange_code,
    fetch_userinfo,
    generate_pkce,
    load_pending,
    parse_code_input,
    refresh_access_token,
    save_pending,
    wait_for_loopback_code,
)


def test_pkce_is_s256():
    import base64, hashlib
    p = generate_pkce()
    expected = base64.urlsafe_b64encode(hashlib.sha256(p.verifier.encode()).digest()).rstrip(b"=").decode()
    assert p.challenge == expected
    assert len(p.verifier) >= 43


def test_authorize_url_has_exact_scopes_and_no_openid():
    url = build_authorize_url("chal", "st8")
    q = parse_qs(urlparse(url).query)
    assert urlparse(url).netloc == "accounts.google.com"
    scopes = q["scope"][0].split(" ")
    assert scopes == list(oauth.SCOPES)
    assert "openid" not in scopes
    assert q["code_challenge_method"] == ["S256"]
    assert q["redirect_uri"] == ["http://localhost:51121/oauth-callback"]
    assert q["access_type"] == ["offline"] and q["prompt"] == ["consent"]
    assert q["state"] == ["st8"]


def test_parse_code_input_accepts_raw_and_url():
    assert parse_code_input(" 4/0Abc ") == ("4/0Abc", None)
    assert parse_code_input("http://localhost:51121/oauth-callback?code=4%2F0Abc&state=s1") == ("4/0Abc", "s1")


def _http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_exchange_code_posts_pkce_form():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"] = str(req.url)
        seen["form"] = parse_qs(req.content.decode())
        return httpx.Response(200, json={"access_token": "ya29.new", "refresh_token": "1//r", "expires_in": 3599})

    tok = exchange_code("4/code", "verif", http=_http(handler))
    assert seen["url"] == oauth.TOKEN_URL
    f = seen["form"]
    assert f["grant_type"] == ["authorization_code"]
    assert f["code"] == ["4/code"] and f["code_verifier"] == ["verif"]
    assert f["redirect_uri"] == [oauth.REDIRECT_URI]
    assert "client_id" in f and "client_secret" in f
    assert tok.access_token == "ya29.new" and tok.refresh_token == "1//r"


def test_refresh_keeps_old_refresh_token_and_maps_invalid_grant():
    ok = _http(lambda r: httpx.Response(200, json={"access_token": "ya29.r", "expires_in": 100}))
    tok = refresh_access_token("1//old", http=ok)
    assert tok.refresh_token == "1//old"
    bad = _http(lambda r: httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(InvalidGrant):
        refresh_access_token("1//old", http=bad)


def test_fetch_userinfo_sends_bearer():
    def handler(req):
        assert req.headers["Authorization"] == "Bearer ya29.t"
        return httpx.Response(200, json={"email": "a@b.c"})
    assert fetch_userinfo("ya29.t", http=_http(handler))["email"] == "a@b.c"


def test_constants_are_pinned():
    assert oauth.SCOPES == (
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    )
    assert oauth.TOKEN_URL == "https://oauth2.googleapis.com/token"
    assert oauth.AUTHORIZE_URL == "https://accounts.google.com/o/oauth2/v2/auth"
    assert oauth.REDIRECT_URI == "http://localhost:51121/oauth-callback"


def test_load_pending_corrupt_returns_none(accounts_root):
    path = pending_auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"verifier":"v","state":"s","alias":"a","proxy_url":null,"account_dir":"d","expires_at_ms":"soon"}',
        encoding="utf-8",
    )
    assert load_pending(now_ms=0) is None
    path.write_text("not json{{{", encoding="utf-8")
    assert load_pending(now_ms=0) is None
    clear_pending()


def test_pending_roundtrip_and_expiry(accounts_root):
    p = PendingAuth("v", "s", "alias", None, str(accounts_root / "acc_1"), expires_at_ms=2000)
    save_pending(p)
    assert load_pending(now_ms=1000) == p
    assert load_pending(now_ms=3000) is None
    clear_pending()
    assert load_pending(now_ms=0) is None


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close(); return port


def test_loopback_returns_code_when_state_matches():
    port = _free_port()
    result = {}

    def run():
        try:
            result["code"] = wait_for_loopback_code("st", timeout_s=5, port=port)
        except Exception as exc:
            result["exc"] = exc

    t = threading.Thread(target=run); t.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            conn = socket.create_connection(("127.0.0.1", port), timeout=0.2)
            conn.close()
            break
        except (ConnectionRefusedError, TimeoutError, OSError):
            time.sleep(0.05)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    opener.open(f"http://127.0.0.1:{port}/oauth-callback?code=4%2Fzz&state=st", timeout=3).read()
    t.join(5)
    assert "exc" not in result
    assert result["code"] == "4/zz"


def test_loopback_busy_when_port_taken():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(1)
    port = s.getsockname()[1]
    try:
        with pytest.raises(LoopbackBusy):
            wait_for_loopback_code("st", timeout_s=1, port=port)
    finally:
        s.close()
