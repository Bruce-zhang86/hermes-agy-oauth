"""agy-oauth 模型提供方：把 Antigravity 订阅模型当作 Hermes 的 LLM 后端。

auth_type 选 external_process 的原因：这是 Hermes 核心唯一按 auth_type（而非 provider 名）
键入的凭证与客户端路径，用户插件无需改核心即可让 `hermes -m agy-oauth`、/model、辅助客户端
都走到 create_client。process_command 仅为占位（必须能被 shutil.which 找到），第一期不会拉起子进程。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx

def _agy_link_dir() -> Path:
    """定位共享 ``agylink`` 包所在的 agy-link 插件目录。

    参数：无。
    返回：含 ``agylink/`` 的目录。优先同级 ``plugins/agy-link``（GitHub 安装），
    其次 ``plugins/model-providers/../agy-link``（Hermes 家目录嵌套布局）。
    """
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "agy-link", here.parent.parent / "agy-link"):
        if (candidate / "agylink").is_dir():
            return candidate
    return here.parent / "agy-link"


_AGY_LINK_DIR = _agy_link_dir()
if str(_AGY_LINK_DIR) not in sys.path:
    sys.path.insert(0, str(_AGY_LINK_DIR))

from providers import register_provider  # noqa: E402
from providers.base import ProviderProfile  # noqa: E402

from agylink import pool as _pool  # noqa: E402
from agylink import quota as _quota  # noqa: E402
from agylink.client import AgyOAuthClient, default_http_factory  # noqa: E402
from agylink.paths import pool_file  # noqa: E402
from agylink.token_store import is_expiring, read_token  # noqa: E402


def _pool_store() -> _pool.PoolStore:
    """创建模型目录查询使用的标准账号池存储。

    参数：无。
    返回：绑定标准 pool.json 路径的 PoolStore；测试可替换此注入点。
    """
    return _pool.PoolStore(pool_file())


def _catalog_http(proxy_url: str | None) -> httpx.Client:
    """按账号代理设置创建模型目录查询使用的短生命周期 HTTP 客户端。

    参数 proxy_url：当前账号的可选代理 URL。
    返回：由 agylink 默认工厂创建的 httpx.Client；测试可替换此注入点。
    """
    return default_http_factory(proxy_url)


class AgyOAuthProfile(ProviderProfile):
    """Antigravity 订阅模型 — 自定义 Cloud Code 客户端，无 REST /models 端点。"""

    def create_client(self, **client_kwargs: Any) -> Any:
        """创建使用本地 OAuth 号池的 AgyOAuthClient。

        参数 client_kwargs：Hermes 提供的客户端选项；忽略通用凭证、地址和外部进程参数，
        其余参数可用于注入号池、HTTP 工厂或时钟。
        返回：已初始化的 AgyOAuthClient。
        """
        agy_kwargs = dict(client_kwargs)
        for ignored_key in ("api_key", "base_url", "command", "args"):
            agy_kwargs.pop(ignored_key, None)
        return AgyOAuthClient(**agy_kwargs)

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """从 google 族当前账号拉取模型目录，失败时返回内置兜底列表。

        参数 api_key：Hermes 兼容参数，本插件使用账号 token，故忽略。
        参数 base_url：Hermes 兼容参数，本插件使用固定 Cloud Code 主机，故忽略。
        参数 timeout：Hermes 兼容参数；HTTP 客户端超时由共享工厂配置，故忽略。
        返回：实时模型 ID；无账号、token 不可用、目录为空或任意异常时返回兜底模型 ID。
        """
        del api_key, base_url, timeout
        try:
            account_pool = _pool_store().load()
            now_ms = int(time.time() * 1000)
            for account in account_pool.accounts:
                if (
                    not account.enabled
                    or account.auth_required
                    or _pool.is_cooling(account, "google", now_ms=now_ms)
                ):
                    continue
                try:
                    token_path = account.token_path()
                except ValueError:
                    continue
                token = read_token(token_path)
                if token is None or is_expiring(token):
                    continue
                with _catalog_http(account.proxy_url) as http:
                    payload = _quota.fetch_available_models(
                        token.access_token, http=http
                    )
                model_ids = _quota.model_ids_from_available(payload)
                if model_ids:
                    return model_ids
            return _quota.fallback_model_ids()
        except Exception:
            return _quota.fallback_model_ids()


agy_oauth = AgyOAuthProfile(
    name="agy-oauth",
    aliases=("agy", "antigravity"),
    display_name="Google Antigravity (OAuth)",
    description="Antigravity 订阅模型：Gemini / Claude / GPT-OSS，通过 hermes agy 登录",
    api_mode="chat_completions",
    env_vars=(),
    base_url="agy://oauth",
    auth_type="external_process",
    process_command=sys.executable,
    process_args=(),
    process_command_env_vars=("HERMES_AGY_PROCESS_COMMAND",),
    supports_health_check=False,
    supports_vision=True,
    default_aux_model="gemini-3.6-flash",
    fallback_models=tuple(_quota.fallback_model_ids()),
)

register_provider(agy_oauth)
