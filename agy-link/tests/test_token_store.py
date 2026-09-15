# D:\hermes\plugins\agy-link\tests\test_token_store.py
"""token_store 测试：文件形态必须与官方 agy / DSH 一致。"""
import json
from datetime import datetime, timedelta, timezone

from agylink.token_store import TokenSet, is_expiring, read_token, token_from_response, write_token

SAMPLE = {
    "token": {
        "access_token": "ya29.test",
        "token_type": "Bearer",
        "refresh_token": "1//test",
        "expiry": "2026-09-15T03:35:56.832Z",
    },
    "auth_method": "consumer",
}


def test_read_official_file(tmp_path):
    p = tmp_path / "antigravity-oauth-token"
    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    tok = read_token(p)
    assert tok.access_token == "ya29.test"
    assert tok.refresh_token == "1//test"
    assert tok.expiry == datetime(2026, 9, 15, 3, 35, 56, 832000, tzinfo=timezone.utc)


def test_write_roundtrip_creates_parents(tmp_path):
    p = tmp_path / "a" / "b" / "antigravity-oauth-token"
    tok = TokenSet("ya29.x", "1//y", datetime(2026, 1, 1, tzinfo=timezone.utc))
    write_token(p, tok)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["auth_method"] == "consumer"
    assert data["token"]["expiry"] == "2026-01-01T00:00:00Z"
    assert read_token(p) == tok


def test_read_missing_returns_none(tmp_path):
    assert read_token(tmp_path / "nope") is None


def test_is_expiring_with_skew():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    tok = TokenSet("a", "r", now + timedelta(seconds=30))
    assert is_expiring(tok, skew_seconds=60, now=now)
    assert not is_expiring(TokenSet("a", "r", now + timedelta(hours=1)), now=now)
    assert is_expiring(TokenSet("a", "r", None), now=now)


def test_token_from_response_uses_expires_in_and_fallback_refresh():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tok = token_from_response({"access_token": "ya29.n", "expires_in": 3600}, fallback_refresh="1//old", now=now)
    assert tok.refresh_token == "1//old"
    assert tok.expiry == now + timedelta(seconds=3600)


def test_write_none_expiry_writes_expired_iso_string(tmp_path):
    p = tmp_path / "antigravity-oauth-token"
    tok = TokenSet("ya29.x", "1//y", None)
    write_token(p, tok)
    data = json.loads(p.read_text(encoding="utf-8"))
    expiry = data["token"]["expiry"]
    assert isinstance(expiry, str)
    assert expiry.endswith("Z")
    read_tok = read_token(p)
    assert is_expiring(read_tok) is True
