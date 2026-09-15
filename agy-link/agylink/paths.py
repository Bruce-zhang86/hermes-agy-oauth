"""路径解析：HERMES_HOME、Hermes 号池目录、DSH 号池目录、官方 token 文件位置。"""
from __future__ import annotations

import os
from pathlib import Path


def hermes_home() -> Path:
    """返回 Hermes 家目录。

    优先环境变量 HERMES_HOME；其次 hermes_constants.get_hermes_home()；
    若 hermes_constants 不可用（ImportError）则回退 ~/.hermes。
    参数：无。
    返回：Hermes 家目录 Path。
    """
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env)
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except ImportError:
        return Path.home() / ".hermes"


def accounts_dir() -> Path:
    """Hermes 自有号池根目录。

    参数：无。
    返回：环境变量 HERMES_AGY_ACCOUNTS_DIR 非空时取其值，否则 <hermes_home>/agy-accounts。
    """
    env = os.environ.get("HERMES_AGY_ACCOUNTS_DIR", "").strip()
    return Path(env) if env else hermes_home() / "agy-accounts"


def pool_file() -> Path:
    """号池注册表文件路径。

    参数：无。
    返回：<accounts_dir>/pool.json。
    """
    return accounts_dir() / "pool.json"


def pending_auth_file() -> Path:
    """进行中 PKCE 登录状态文件路径。

    参数：无。
    返回：<accounts_dir>/.pending-auth.json。
    """
    return accounts_dir() / ".pending-auth.json"


def dsh_accounts_dir() -> Path:
    """DSH 号池目录。

    参数：无。
    返回：环境变量 DSH_AGY_ACCOUNTS_DIR 非空时取其值，否则 ~/.dsh/agy-accounts。
    """
    env = os.environ.get("DSH_AGY_ACCOUNTS_DIR", "").strip()
    return Path(env) if env else Path.home() / ".dsh" / "agy-accounts"


def token_file_for(home: Path) -> Path:
    """给定账号 HOME，返回官方 agy 的 OAuth token 文件路径。

    参数 home：账号的 HOME 目录。
    返回：<home>/.gemini/antigravity-cli/antigravity-oauth-token。
    """
    return Path(home) / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
