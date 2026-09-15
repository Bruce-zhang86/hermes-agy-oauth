# D:\hermes\plugins\agy-link\tests\test_pool.py
"""pool 模块测试：族映射、调度、冷却、DSH 导入、删除守卫。"""
import json
from pathlib import Path

import pytest

from agylink import pool as P


def test_family_of():
    assert P.family_of("gemini-3.6-flash") == "google"
    assert P.family_of("gemma-2") == "google"
    assert P.family_of("claude-sonnet-4-6") == "anthropic"
    assert P.family_of("gpt-oss-120b-medium") == "openai"
    assert P.family_of("gpt_oss-1") is None
    assert P.family_of("llama-3") is None


def _acc(i, **kw):
    base = dict(id=f"acc_{i}", alias=f"a{i}", dir=f"D:/x/acc_{i}", system_home=False, source="hermes",
                email=None, enabled=True, proxy_url=None, auth_required=False, created_at=0,
                last_used_at=0, project_id=None, cooldowns={}, quotas={})
    base.update(kw)
    return P.Account(**base)


def test_home_requires_dir_unless_system_home(tmp_path):
    assert _acc(1, system_home=True, dir="").home() == Path.home()
    d = tmp_path / "acc_x"
    assert _acc(2, system_home=False, dir=str(d)).home() == d
    with pytest.raises(ValueError):
        _acc(3, system_home=False, dir="").home()


def test_store_load_corrupt_raises(accounts_root):
    path = accounts_root / "pool.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        P.PoolStore(path).load()


def test_store_roundtrip_and_empty(accounts_root):
    store = P.PoolStore(accounts_root / "pool.json")
    pool = store.load()
    assert pool.accounts == [] and pool.version == 1
    assert pool.rate_limit_cooldown_ms == 60_000
    pool.accounts.append(_acc(1))
    pool.rate_limit_cooldown_ms = 12_345
    store.save(pool)
    again = store.load()
    assert again.accounts[0].id == "acc_1"
    assert again.rate_limit_cooldown_ms == 12_345
    raw = json.loads((accounts_root / "pool.json").read_text())
    assert raw["accounts"][0]["source"] == "hermes"
    assert raw["rateLimitCooldownMs"] == 12_345


def test_select_sticky_then_first_available():
    pool = P.Pool(accounts=[_acc(1), _acc(2)], active_account_ids={})
    assert P.select_account(pool, "google", now_ms=0).id == "acc_1"
    assert pool.active_account_ids["google"] == "acc_1"
    pool.accounts[0].auth_required = True
    assert P.select_account(pool, "google", now_ms=0).id == "acc_2"


def test_cooldown_is_family_scoped_and_capped():
    pool = P.Pool(accounts=[_acc(1)], active_account_ids={})
    until = P.mark_cooldown(
        pool,
        "acc_1",
        "anthropic",
        reason="429",
        reset_time_iso="2099-01-01T00:00:00Z",
        now_ms=1_000,
    )
    assert until == 1_000 + pool.max_cooldown_ms
    assert P.is_cooling(pool.accounts[0], "anthropic", now_ms=2_000)
    assert not P.is_cooling(pool.accounts[0], "google", now_ms=2_000)
    far = P.mark_cooldown(pool, "acc_1", "google", reason="quota", reset_time_iso="2099-01-01T00:00:00Z", now_ms=1_000)
    assert far == 1_000 + pool.max_cooldown_ms


def test_mark_cooldown_without_reset_uses_rate_limit_window():
    pool = P.Pool(
        accounts=[_acc(1)],
        active_account_ids={},
        default_cooldown_ms=900_000,
        rate_limit_cooldown_ms=60_000,
    )
    until = P.mark_cooldown(pool, "acc_1", "google", reason="429", now_ms=1_000)
    assert until == 61_000
    assert pool.accounts[0].cooldowns["google"]["reason"] == "429"


def test_mark_cooldown_with_expired_reset_uses_default_window():
    pool = P.Pool(
        accounts=[_acc(1)],
        active_account_ids={},
        default_cooldown_ms=900_000,
        rate_limit_cooldown_ms=60_000,
    )
    until = P.mark_cooldown(
        pool,
        "acc_1",
        "google",
        reason="expired-reset",
        reset_time_iso="1970-01-01T00:00:00Z",
        now_ms=1_000,
    )
    assert until == 901_000
    assert pool.accounts[0].cooldowns["google"]["reason"] == "expired-reset"


def test_advance_skips_current_and_cooling():
    pool = P.Pool(accounts=[_acc(1), _acc(2), _acc(3)], active_account_ids={"google": "acc_1"})
    P.mark_cooldown(pool, "acc_2", "google", reason="429", now_ms=0)
    nxt = P.advance(pool, "google", exclude_id="acc_1", now_ms=10)
    assert nxt.id == "acc_3" and pool.active_account_ids["google"] == "acc_3"
    assert P.advance(pool, "google", exclude_id="acc_3", now_ms=10) is None


def test_import_dsh_references_dir_and_dedupes(tmp_path):
    dsh = tmp_path / "dsh"; dsh.mkdir()
    (dsh / "pool.json").write_text(json.dumps({
        "accounts": [
            {"id": "acc_primary", "alias": "主账号", "dir": "", "systemHome": True, "enabled": True},
            {"id": "acc_x", "alias": "备用", "dir": str(dsh / "acc_x"), "email": "u@e.c", "enabled": True},
        ]
    }))
    before = (dsh / "pool.json").read_text()
    pool = P.Pool(accounts=[], active_account_ids={})
    assert P.import_dsh(pool, dsh / "pool.json") == (2, 0)
    assert P.import_dsh(pool, dsh / "pool.json") == (0, 2)
    imported = {a.id: a for a in pool.accounts}
    assert imported["acc_x"].source == "dsh" and imported["acc_x"].dir == str(dsh / "acc_x")
    assert imported["acc_primary"].system_home is True
    assert (dsh / "pool.json").read_text() == before


def test_delete_account_dir_guards(tmp_path):
    root = tmp_path / "agy-accounts"; (root / "acc_1").mkdir(parents=True)
    P.delete_account_dir(_acc(1, dir=str(root / "acc_1")), root)
    assert not (root / "acc_1").exists()
    with pytest.raises(ValueError, match="账号缺少 dir"):
        P.delete_account_dir(_acc("empty", dir=""), root)
    outside = tmp_path / "elsewhere"; outside.mkdir()
    with pytest.raises(ValueError):
        P.delete_account_dir(_acc(2, dir=str(outside)), root)
    with pytest.raises(ValueError):
        P.delete_account_dir(_acc(3, dir=str(root / "acc_3"), source="dsh"), root)
