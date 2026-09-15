# D:\hermes\plugins\agy-link\agylink\pool.py
"""号池：pool.json 读写、模型族映射、按族粘滞的顺次耗尽调度、冷却、DSH 导入。

pool.json 字段与 spec 第 6 节一致（camelCase 落盘，Python 侧 snake_case）。
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agylink.paths import token_file_for

FAMILIES: tuple[str, ...] = ("google", "anthropic", "openai")


def family_of(model_id: str) -> Optional[str]:
    """根据模型 ID 判断共享额度的模型族。

    参数 model_id：待分类的模型 ID。
    返回：gemini/gemma 为 google，claude 为 anthropic，gpt-oss 为 openai；其他返回 None。
    """
    m = (model_id or "").lower()
    if m.startswith("gemini") or m.startswith("gemma"):
        return "google"
    if "claude" in m:
        return "anthropic"
    if "gpt-oss" in m:
        return "openai"
    return None


@dataclass
class Account:
    """一个账号槽。dir 为空且 system_home=True 时 HOME 用当前用户主目录。"""

    id: str
    alias: str
    dir: str
    system_home: bool
    source: str  # "hermes" | "dsh"
    email: Optional[str]
    enabled: bool
    proxy_url: Optional[str]
    auth_required: bool
    created_at: int
    last_used_at: int
    project_id: Optional[str]
    cooldowns: dict = field(default_factory=dict)
    quotas: dict = field(default_factory=dict)

    def home(self) -> Path:
        """取得账号隔离使用的 HOME 目录。

        参数：无。
        返回：system_home 为真时返回当前用户目录，否则返回账号 dir。
        """
        if self.system_home:
            return Path.home()
        if self.dir:
            return Path(self.dir)
        raise ValueError(f"账号 {self.id} 缺少 dir 且非 systemHome")

    def token_path(self) -> Path:
        """取得账号 HOME 下的官方 token 文件路径。

        参数：无。
        返回：账号对应的 antigravity OAuth token 文件路径。
        """
        return token_file_for(self.home())


@dataclass
class Pool:
    """整个号池。"""

    accounts: list[Account]
    active_account_ids: dict[str, str]
    version: int = 1
    mode: str = "sequential"
    default_cooldown_ms: int = 900_000
    rate_limit_cooldown_ms: int = 60_000
    max_cooldown_ms: int = 3_600_000


_ACC_KEYS = {
    "id": "id", "alias": "alias", "dir": "dir", "systemHome": "system_home", "source": "source",
    "email": "email", "enabled": "enabled", "proxyUrl": "proxy_url", "authRequired": "auth_required",
    "createdAt": "created_at", "lastUsedAt": "last_used_at", "projectId": "project_id",
    "cooldowns": "cooldowns", "quotas": "quotas",
}


def _account_from_json(d: dict) -> Account:
    """把 camelCase 持久化字典转换为 Account 并补齐缺省值。

    参数 d：pool.json 中的单个账号字典。
    返回：规范化后的 Account。
    """
    kw = {py: d.get(js) for js, py in _ACC_KEYS.items()}
    kw["alias"] = kw["alias"] or kw["id"] or ""
    kw["dir"] = kw["dir"] or ""
    kw["system_home"] = bool(kw["system_home"])
    kw["source"] = kw["source"] or "hermes"
    kw["enabled"] = True if kw["enabled"] is None else bool(kw["enabled"])
    kw["auth_required"] = bool(kw["auth_required"])
    kw["created_at"] = int(kw["created_at"] or 0)
    kw["last_used_at"] = int(kw["last_used_at"] or 0)
    kw["cooldowns"] = kw["cooldowns"] or {}
    kw["quotas"] = kw["quotas"] or {}
    return Account(**kw)


def _account_to_json(a: Account) -> dict:
    """把 Account 转换为 camelCase 持久化字典。

    参数 a：待序列化账号。
    返回：可写入 pool.json 的账号字典。
    """
    return {js: getattr(a, py) for js, py in _ACC_KEYS.items()}


class PoolStore:
    """pool.json 的加载与原子保存。"""

    def __init__(self, path: Path):
        """绑定此存储实例读写的 pool.json 路径。

        参数 path：号池 JSON 文件路径。
        返回：无。
        """
        self.path = Path(path)

    def load(self) -> Pool:
        """读取 pool.json 并补齐池级默认配置。

        参数：无。
        返回：反序列化后的 Pool；文件不存在时返回空池，其他读取或 JSON 错误向上抛出。
        """
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Pool(accounts=[], active_account_ids={})
        data = json.loads(text)
        return Pool(
            accounts=[_account_from_json(a) for a in data.get("accounts", []) if isinstance(a, dict)],
            active_account_ids=dict(data.get("activeAccountIds") or {}),
            version=int(data.get("version") or 1),
            mode=str(data.get("mode") or "sequential"),
            default_cooldown_ms=int(data.get("defaultCooldownMs") or 900_000),
            rate_limit_cooldown_ms=int(data.get("rateLimitCooldownMs") or 60_000),
            max_cooldown_ms=int(data.get("maxCooldownMs") or 3_600_000),
        )

    def save(self, pool: Pool) -> None:
        """把完整号池以 camelCase 字段原子写回 pool.json。

        参数 pool：待持久化的 Pool。
        返回：无。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": pool.version,
            "mode": pool.mode,
            "defaultCooldownMs": pool.default_cooldown_ms,
            "rateLimitCooldownMs": pool.rate_limit_cooldown_ms,
            "maxCooldownMs": pool.max_cooldown_ms,
            "accounts": [_account_to_json(a) for a in pool.accounts],
            "activeAccountIds": pool.active_account_ids,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def new_account_id(now_ms: int) -> str:
    """生成带时间戳和随机后缀的新账号 ID。

    参数 now_ms：当前毫秒时间戳。
    返回：``acc_<毫秒时间戳>_<5位随机>`` 形式的 ID。
    """
    return f"acc_{now_ms}_{secrets.token_hex(3)[:5]}"


def is_cooling(acc: Account, family: str, now_ms: int) -> bool:
    """判断账号在指定模型族是否仍处于冷却期。

    参数 acc：待检查账号。
    参数 family：模型族。
    参数 now_ms：当前毫秒时间戳。
    返回：冷却截止时间晚于 now_ms 时返回 True。
    """
    cd = acc.cooldowns.get(family) or {}
    return int(cd.get("untilMs") or 0) > now_ms


def _usable(acc: Account, family: str, now_ms: int) -> bool:
    """判断账号是否可用于指定模型族。

    参数 acc：待检查账号。
    参数 family：模型族。
    参数 now_ms：当前毫秒时间戳。
    返回：账号启用、无需重新授权且未冷却时返回 True。
    """
    return acc.enabled and not acc.auth_required and not is_cooling(acc, family, now_ms)


def _find(pool: Pool, account_id: Optional[str]) -> Optional[Account]:
    """按账号 ID 在号池中查找账号。

    参数 pool：待搜索号池。
    参数 account_id：目标账号 ID，可为 None。
    返回：匹配的 Account；不存在时返回 None。
    """
    return next((a for a in pool.accounts if a.id == account_id), None)


def select_account(pool: Pool, family: str, now_ms: int) -> Optional[Account]:
    """按模型族执行粘滞选号。

    参数 pool：待选择的号池。
    参数 family：模型族。
    参数 now_ms：当前毫秒时间戳。
    返回：仍可用的 active 账号，否则首个可用账号；没有时返回 None。
    """
    current = _find(pool, pool.active_account_ids.get(family))
    if current and _usable(current, family, now_ms):
        return current
    for acc in pool.accounts:
        if _usable(acc, family, now_ms):
            pool.active_account_ids[family] = acc.id
            return acc
    return None


def _parse_iso_ms(iso: Optional[str]) -> Optional[int]:
    """解析 ISO-8601 时间为 UTC 毫秒时间戳。

    参数 iso：ISO-8601 字符串，可为 None。
    返回：UTC 毫秒时间戳；缺失或无效时返回 None。
    """
    if not iso:
        return None
    try:
        text = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
        return int(datetime.fromisoformat(text).astimezone(timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


def mark_cooldown(pool: Pool, account_id: str, family: str, *, reason: str,
                  reset_time_iso: Optional[str] = None, now_ms: int) -> int:
    """给账号的指定模型族记录冷却截止时间和原因。

    参数 pool：待修改的账号池。
    参数 account_id：发生限流的账号 ID。
    参数 family：需要独立冷却的模型族。
    参数 reason：写入冷却记录的原因字符串。
    参数 reset_time_iso：服务端给出的 ISO resetTime；缺失或无法解析时视为短时限流。
    参数 now_ms：当前毫秒时间戳。
    返回：冷却截止毫秒；找不到账号时原样返回 now_ms。

    有效的未来 resetTime 直接使用并受 max_cooldown_ms 限制；有效但已过期的
    resetTime 沿用 default_cooldown_ms；缺失或无效值使用 rate_limit_cooldown_ms。
    """
    acc = _find(pool, account_id)
    if acc is None:
        return now_ms
    until = _parse_iso_ms(reset_time_iso)
    if until is None:
        until = now_ms + pool.rate_limit_cooldown_ms
    elif until <= now_ms:
        until = now_ms + pool.default_cooldown_ms
    until = min(until, now_ms + pool.max_cooldown_ms)
    acc.cooldowns[family] = {"untilMs": until, "reason": reason}
    return until


def advance(pool: Pool, family: str, *, exclude_id: str, now_ms: int) -> Optional[Account]:
    """从失败账号之后向前扫描并切换到下一个可用账号。

    参数 pool：待调度号池。
    参数 family：模型族。
    参数 exclude_id：当前失败账号 ID。
    参数 now_ms：当前毫秒时间戳。
    返回：后续首个可用账号并将其设为 active；没有时返回 None。
    """
    after_exclude = False
    for acc in pool.accounts:
        if acc.id == exclude_id:
            after_exclude = True
            continue
        if after_exclude and _usable(acc, family, now_ms):
            pool.active_account_ids[family] = acc.id
            return acc
    return None


def import_dsh(pool: Pool, dsh_pool_path: Path) -> tuple[int, int]:
    """把 DSH pool.json 里的账号登记为 source=dsh（只读引用 dir，不复制 token）。

    参数 pool：接收导入账号的 Hermes 号池。
    参数 dsh_pool_path：只读的 DSH pool.json 路径。
    返回：(新导入数, 跳过数)；按 ID 去重且不修改 DSH 文件。
    """
    data = json.loads(Path(dsh_pool_path).read_text(encoding="utf-8"))
    existing = {a.id for a in pool.accounts}
    imported = skipped = 0
    for raw in data.get("accounts", []):
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        if raw["id"] in existing:
            skipped += 1
            continue
        pool.accounts.append(Account(
            id=str(raw["id"]), alias=str(raw.get("alias") or raw["id"]), dir=str(raw.get("dir") or ""),
            system_home=bool(raw.get("systemHome")), source="dsh", email=raw.get("email"),
            enabled=bool(raw.get("enabled", True)), proxy_url=raw.get("proxyUrl"), auth_required=False,
            created_at=int(raw.get("createdAt") or 0), last_used_at=0, project_id=None,
            cooldowns={}, quotas=dict(raw.get("quotas") or {}),
        ))
        existing.add(str(raw["id"]))
        imported += 1
    return imported, skipped


def delete_account_dir(acc: Account, accounts_root: Path) -> None:
    """在路径守卫通过后删除 Hermes 自建账号目录。

    参数 acc：待删除目录的账号。
    参数 accounts_root：允许删除的号池根目录。
    返回：无；dir 缺失、source 非 hermes 或目标不在 accounts_root 下时抛 ValueError。
    """
    if not acc.dir:
        raise ValueError("账号缺少 dir，拒绝删除")
    if acc.source != "hermes":
        raise ValueError("只允许删除 source=hermes 的账号目录")
    target = Path(acc.dir).resolve()
    root = Path(accounts_root).resolve()
    if root not in target.parents:
        raise ValueError(f"拒绝删除号池目录之外的路径：{target}")
    shutil.rmtree(target, ignore_errors=True)
