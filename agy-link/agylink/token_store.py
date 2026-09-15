# D:\hermes\plugins\agy-link\agylink\token_store.py
"""官方 antigravity-oauth-token 文件的读写。

文件形态（与 agy CLI / dsh-agy-link 一致）：
{"token": {"access_token", "token_type", "refresh_token", "expiry"}, "auth_method": "consumer"}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class TokenSet:
    """一组 OAuth 凭证。expiry 为 UTC datetime，None 表示未知（按已过期处理）。"""

    access_token: str
    refresh_token: str
    expiry: datetime | None
    token_type: str = "Bearer"
    auth_method: str = "consumer"


def _parse_expiry(raw: object) -> datetime | None:
    """把 ISO-8601 时间字符串解析为 UTC datetime。

    参数 raw：token 文件里的 expiry 原始值（末尾 Z 或带偏移；非字符串视为无效）。
    返回：UTC datetime；解析失败或为空返回 None。
    """
    if not isinstance(raw, str) or not raw:
        return None
    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_expiry(dt: datetime) -> str:
    """把 datetime 格式化为 agy 接受的 ISO 字符串。

    参数 dt：任意时区的 datetime（会先转 UTC）。
    返回：形如 2026-09-15T08:00:00Z 或 ...:00.123Z（有微秒时保留毫秒）的字符串。
    """
    dt = dt.astimezone(timezone.utc)
    if dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_token(path: Path) -> TokenSet | None:
    """读取官方 token 文件。

    参数 path：antigravity-oauth-token 文件路径。
    返回：TokenSet；文件不存在、JSON 损坏或缺 access_token 时返回 None。
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, dict) or not token.get("access_token"):
        return None
    return TokenSet(
        access_token=str(token["access_token"]),
        refresh_token=str(token.get("refresh_token") or ""),
        expiry=_parse_expiry(token.get("expiry")),
        token_type=str(token.get("token_type") or "Bearer"),
        auth_method=str(data.get("auth_method") or "consumer"),
    )


def write_token(path: Path, token: TokenSet) -> None:
    """原子写入 token 文件（先写 .tmp 再 os.replace），自动创建父目录。

    参数 path：目标 token 文件路径。
    参数 token：待写入的 TokenSet；expiry 为 None 时以当前时间落盘（等价于立即过期）。
    返回：无。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token": {
            "access_token": token.access_token,
            "token_type": token.token_type,
            "refresh_token": token.refresh_token,
            "expiry": _format_expiry(
                token.expiry if token.expiry else datetime.now(timezone.utc)
            ),
        },
        "auth_method": token.auth_method,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def is_expiring(token: TokenSet, *, skew_seconds: int = 60, now: datetime | None = None) -> bool:
    """判断 access token 是否即将过期。

    参数 token：待判断的 TokenSet。
    参数 skew_seconds：提前判定过期的安全余量（秒），默认 60。
    参数 now：当前时间（UTC），默认取系统时间；测试可注入。
    返回：True 表示已过期或将在 skew_seconds 内过期；expiry 未知一律 True。
    """
    if token.expiry is None:
        return True
    now = now or datetime.now(timezone.utc)
    return token.expiry - now <= timedelta(seconds=skew_seconds)


def token_from_response(payload: dict, *, fallback_refresh: str = "", now: datetime | None = None) -> TokenSet:
    """把 Google token 端点响应转成 TokenSet。

    参数 payload：token 端点 JSON 响应，必须含 access_token；expires_in（秒）换算成绝对 expiry。
    参数 fallback_refresh：响应无 refresh_token 时沿用的旧值（刷新场景 Google 不回传）。
    参数 now：计算 expiry 的基准时间（UTC），默认系统时间。
    返回：TokenSet（expires_in 缺失或非数字时 expiry 为 None）。
    """
    now = now or datetime.now(timezone.utc)
    expires_in = payload.get("expires_in")
    expiry = now + timedelta(seconds=int(expires_in)) if isinstance(expires_in, (int, float)) else None
    return TokenSet(
        access_token=str(payload["access_token"]),
        refresh_token=str(payload.get("refresh_token") or fallback_refresh),
        expiry=expiry,
        token_type=str(payload.get("token_type") or "Bearer"),
    )
