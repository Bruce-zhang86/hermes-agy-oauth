"""AgyOAuthClient：Hermes 可直接使用的 OpenAI 形态客户端，底层调用 Cloud Code。

每次 ``create()`` 会完成模型分族、粘滞选号、OAuth token 刷新、项目发现、
Cloud Code SSE 请求以及必要的账号轮换。客户端在内部消费完整 SSE，再返回
OpenAI 形态 completion；调用方要求流式时，则返回兼容 Hermes 的伪流分块。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx
from agent.acp_openai_bridge import completion_to_stream_chunks

from agylink import pool as P
from agylink.oauth import InvalidGrant, OAuthError, refresh_access_token
from agylink.paths import pool_file
from agylink.quota import post_internal, preferred_hosts
from agylink.redact import redact
from agylink.token_store import is_expiring, read_token, write_token
from agylink.transport import (
    UpstreamError,
    accumulate,
    build_envelope,
    iter_sse_events,
    request_headers,
    to_completion,
)

logger = logging.getLogger(__name__)

# 进程级粘滞：Hermes 可能在重试/换客户端时新建 AgyOAuthClient，
# 实例字段扛不住；隐式缓存必须打到同一台 Cloud Code 主机才容易命中。
_LAST_GOOD_HOST: Optional[str] = None
_LAST_GOOD_HOST_LOCK = threading.Lock()


def last_good_host() -> Optional[str]:
    """读取进程内上次 streamGenerateContent 成功的主机。

    参数：无。
    返回：完整主机 URL；尚未成功过则 None。
    """
    with _LAST_GOOD_HOST_LOCK:
        return _LAST_GOOD_HOST


def remember_good_host(host: str) -> None:
    """记录进程内上次成功的 Cloud Code 主机。

    参数 host：完整主机 URL（含 https://）。
    返回：无。
    """
    global _LAST_GOOD_HOST
    if not host:
        return
    with _LAST_GOOD_HOST_LOCK:
        _LAST_GOOD_HOST = host


def clear_last_good_host() -> None:
    """清空主机粘滞（测试用，避免用例间串扰）。

    参数：无。
    返回：无。
    """
    global _LAST_GOOD_HOST
    with _LAST_GOOD_HOST_LOCK:
        _LAST_GOOD_HOST = None

_AUTH_HINT = (
    "没有可用的 Antigravity 账号。先运行 `hermes agy auth` 登录，"
    "或 `hermes agy import-dsh` 导入 DSH 号池。"
)


class NoAccountAvailable(RuntimeError):
    """号池为空或目标模型族没有可用账号。"""


def default_http_factory(proxy_url: Optional[str]) -> httpx.Client:
    """构造供单次账号请求使用的 httpx 同步客户端。

    参数 proxy_url：账号级代理 URL；为空时仅使用 httpx 的环境代理配置。
    返回：启用环境配置、连接超时 30 秒、总超时 300 秒的 httpx.Client。
    """
    kwargs: dict[str, Any] = {
        "trust_env": True,
        "timeout": httpx.Timeout(300.0, connect=30.0),
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
    return httpx.Client(**kwargs)


class _Completions:
    """把 OpenAI 风格的 completions 入口转发给主客户端。"""

    def __init__(self, client: "AgyOAuthClient"):
        """保存主客户端引用。

        参数 client：实际执行请求的 AgyOAuthClient。
        返回：无。
        """
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        """执行一次 OpenAI 形态的聊天补全。

        参数 kwargs：模型、消息、工具、流式标志及生成参数。
        返回：OpenAI 形态 completion，或 stream=True 时的伪流分块列表。
        """
        return self._client._create(**kwargs)


class AgyOAuthClient:
    """集成账号池和 Cloud Code 的 OpenAI 形态客户端。"""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        store: Optional[P.PoolStore] = None,
        http_factory: Optional[Callable[[Optional[str]], httpx.Client]] = None,
        now_ms: Optional[Callable[[], int]] = None,
        **_: Any,
    ):
        """初始化客户端及 OpenAI 风格入口。

        参数 api_key：兼容 Hermes 构造协议的占位 API key。
        参数 base_url：兼容 Hermes 构造协议的占位基础 URL。
        参数 store：可注入的账号池存储；缺省使用标准 pool.json。
        参数 http_factory：按账号代理 URL 创建 httpx.Client 的工厂。
        参数 now_ms：可注入、可 monkeypatch 的毫秒时钟。
        参数 _：忽略其他 OpenAI 客户端兼容参数。
        返回：无；实例的 ``_sleep`` 默认为 time.sleep，可在测试中替换。
        """
        self.api_key = api_key or "agy-oauth"
        self.base_url = base_url or "agy://oauth"
        self._store = store or P.PoolStore(pool_file())
        self._http_factory = http_factory or default_http_factory
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))
        self._sleep = time.sleep
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.is_closed = False

    def close(self) -> None:
        """把客户端标记为已关闭。

        参数：无。
        返回：无；每次请求使用的短生命周期 httpx.Client 已在请求结束时关闭。
        """
        self.is_closed = True

    def _refresh_token(self, acc: P.Account, http: httpx.Client) -> Optional[str]:
        """强制刷新账号 access token 并写回其独立 token 文件。

        参数 acc：需要刷新的账号。
        参数 http：用于访问 Google token 端点的客户端。
        返回：新 access token；缺 refresh token、路径无效或 invalid_grant 时返回 None。

        非 invalid_grant 的 OAuth/网络错误保持原异常向上传递，由调用方临时换号。
        """
        try:
            token = read_token(acc.token_path())
        except ValueError:
            return None
        if token is None or not token.refresh_token:
            return None
        try:
            fresh = refresh_access_token(token.refresh_token, http=http)
        except InvalidGrant:
            return None
        write_token(acc.token_path(), fresh)
        return fresh.access_token

    def _access_token(self, acc: P.Account, http: httpx.Client) -> Optional[str]:
        """读取可用 access token，临近过期时先刷新。

        参数 acc：当前账号。
        参数 http：用于必要 OAuth 刷新的客户端。
        返回：可用 access token；凭证/路径缺失或刷新 invalid_grant 时返回 None。

        非 invalid_grant 的 OAuth/网络错误保持原异常向上传递，由调用方临时换号。
        """
        try:
            token = read_token(acc.token_path())
        except ValueError:
            return None
        if token is None:
            return None
        if not is_expiring(token, now=datetime.now(timezone.utc)):
            return token.access_token
        return self._refresh_token(acc, http)

    def _project_id(self, acc: P.Account, token: str, http: httpx.Client) -> str:
        """按固定优先级取得 Cloud Code 项目 ID。

        参数 acc：当前账号，可能已带缓存 project_id。
        参数 token：当前 OAuth access token。
        参数 http：用于 loadCodeAssist 的客户端。
        返回：账号缓存、HERMES_AGY_PROJECT_ID 或 loadCodeAssist 发现的项目 ID；
        仅成功的 loadCodeAssist 结果写入账号缓存，发现失败时抛出可读错误。
        """
        if acc.project_id:
            return acc.project_id
        configured = os.environ.get("HERMES_AGY_PROJECT_ID", "").strip()
        if configured:
            return configured
        payload = post_internal("loadCodeAssist", token, http=http, body={})
        raw = (payload or {}).get("cloudaicompanionProject")
        project = raw.get("id") if isinstance(raw, dict) else raw
        if project:
            acc.project_id = str(project)
            return acc.project_id
        raise NoAccountAvailable(
            "无法发现 Cloud Code 项目（loadCodeAssist 失败）："
            "请运行 hermes agy auth 重新登录，或检查代理设置。"
        )

    def _stream_once(
        self,
        envelope: dict,
        token: str,
        http: httpx.Client,
    ):
        """依次请求生产主机并累积一次 Cloud Code SSE 响应。

        优先打上次成功的主机，降低 Gemini 隐式缓存被双主机打散的概率。
        参数 envelope：Cloud Code 请求信封。
        参数 token：OAuth access token。
        参数 http：当前账号对应的客户端。
        返回：transport.Accumulated 响应累积结果。
        """
        last: Optional[UpstreamError] = None
        for host in preferred_hosts(last_good_host()):
            url = f"{host}/v1internal:streamGenerateContent?alt=sse"
            try:
                with http.stream(
                    "POST",
                    url,
                    json=envelope,
                    headers=request_headers(token),
                ) as response:
                    if response.status_code >= 400:
                        body = redact(response.read().decode("utf-8", "replace"))
                        error = UpstreamError(response.status_code, body)
                        if 400 <= response.status_code < 500:
                            raise error
                        last = error
                        continue
                    remember_good_host(host)
                    return accumulate(iter_sse_events(response.iter_lines()))
            except httpx.HTTPError as exc:
                message = redact(str(exc))
                last = UpstreamError(
                    0,
                    f"网络错误：{message}（国内网络需开启 TUN 或设置 HTTPS_PROXY）",
                )
                continue
        raise last or UpstreamError(0, "所有主机都失败")

    @staticmethod
    def _mark_auth_required(pool: P.Pool, acc: P.Account) -> None:
        """把凭证失效账号标记为需要重新授权。

        参数 pool：包含该账号的号池，保留此参数以明确变更所属聚合对象。
        参数 acc：待标记账号。
        返回：无。
        """
        del pool
        acc.auth_required = True

    def _create(
        self,
        *,
        model: Optional[str] = None,
        messages: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Any = None,
        stream: bool = False,
        reasoning_effort: Optional[str] = None,
        temperature: Any = None,
        top_p: Any = None,
        max_tokens: Any = None,
        stop: Any = None,
        **_: Any,
    ) -> Any:
        """完成一次选号、鉴权、项目发现、推理及有限重试。

        参数 model：模型 ID，用于选择 google/anthropic/openai 账号族。
        参数 messages：OpenAI chat messages。
        参数 tools：可选工具定义。
        参数 tool_choice：工具选择策略。
        参数 stream：是否返回 Hermes 兼容的伪流分块。
        参数 reasoning_effort：推理强度。
        参数 temperature、top_p、max_tokens、stop：标准生成参数。
        参数 _：忽略其余 OpenAI 兼容参数。
        返回：OpenAI 形态 completion 或伪流分块；临时刷新失败时前向换号，
        401 刷新重试仍失败时标记重新授权并前向换号；无 resetTime 且无备用账号
        的首次 429 会等待不超过 15 秒并同账号重试一次。
        """
        model_id = model or ""
        family = P.family_of(model_id)
        if family is None:
            raise ValueError(
                f"模型 {model_id!r} 无法归入 google/anthropic/openai 任一族"
            )

        pool = self._store.load()
        account = P.select_account(pool, family, self._now_ms())
        if account is None:
            raise NoAccountAvailable(_AUTH_HINT)

        quota_attempts = 0
        rate_limit_waited = False
        wait_retry_account_id: Optional[str] = None
        refreshed_after_401: set[str] = set()
        while True:
            http = self._http_factory(account.proxy_url)
            try:
                try:
                    token = self._access_token(account, http)
                except (OAuthError, httpx.HTTPError) as refresh_error:
                    logger.warning(
                        "agy-oauth: 账号 %s 刷新临时失败: %s",
                        account.id,
                        redact(str(refresh_error)),
                    )
                    next_account = P.advance(
                        pool,
                        family,
                        exclude_id=account.id,
                        now_ms=self._now_ms(),
                    )
                    if next_account is None:
                        raise
                    self._store.save(pool)
                    account = next_account
                    continue
                if token is None:
                    self._mark_auth_required(pool, account)
                    self._store.save(pool)
                    logger.warning(
                        "agy-oauth: 账号 %s 凭证失效，标记 authRequired",
                        account.id,
                    )
                    next_account = P.advance(
                        pool,
                        family,
                        exclude_id=account.id,
                        now_ms=self._now_ms(),
                    )
                    if next_account is None:
                        raise NoAccountAvailable(_AUTH_HINT)
                    account = next_account
                    continue

                project = self._project_id(account, token, http)
                envelope = build_envelope(
                    project_id=project,
                    model=model_id,
                    messages=messages or [],
                    tools=tools,
                    tool_choice=tool_choice,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                result = self._stream_once(envelope, token, http)
                account.last_used_at = self._now_ms()
                if wait_retry_account_id == account.id:
                    account.cooldowns.pop(family, None)
                    wait_retry_account_id = None
                self._store.save(pool)
                completion = to_completion(result, model=model_id)
                if stream:
                    return completion_to_stream_chunks(completion)
                return completion
            except UpstreamError as error:
                logger.warning(
                    "agy-oauth: 账号 %s 上游错误 %s: %s",
                    account.id,
                    error.status,
                    redact(error.body)[:300],
                )
                if error.is_auth() and account.id not in refreshed_after_401:
                    refreshed_after_401.add(account.id)
                    try:
                        token = self._refresh_token(account, http)
                    except (OAuthError, httpx.HTTPError) as refresh_error:
                        logger.warning(
                            "agy-oauth: 账号 %s 401 后刷新临时失败: %s",
                            account.id,
                            redact(str(refresh_error)),
                        )
                        next_account = P.advance(
                            pool,
                            family,
                            exclude_id=account.id,
                            now_ms=self._now_ms(),
                        )
                        if next_account is None:
                            raise
                        self._store.save(pool)
                        account = next_account
                        continue
                    if token is not None:
                        continue
                    self._mark_auth_required(pool, account)
                    next_account = P.advance(
                        pool,
                        family,
                        exclude_id=account.id,
                        now_ms=self._now_ms(),
                    )
                    self._store.save(pool)
                    if next_account is None:
                        raise NoAccountAvailable(_AUTH_HINT) from error
                    account = next_account
                    continue
                if error.is_quota():
                    reset_time = error.reset_time()
                    P.mark_cooldown(
                        pool,
                        account.id,
                        family,
                        reason=str(error.status or "quota"),
                        reset_time_iso=reset_time,
                        now_ms=self._now_ms(),
                    )
                    self._store.save(pool)
                    if quota_attempts >= 1:
                        raise
                    quota_attempts += 1
                    next_account = P.advance(
                        pool,
                        family,
                        exclude_id=account.id,
                        now_ms=self._now_ms(),
                    )
                    if next_account is None:
                        if reset_time is None and not rate_limit_waited:
                            rate_limit_waited = True
                            wait_seconds = min(
                                pool.rate_limit_cooldown_ms / 1000.0,
                                15.0,
                            )
                            self._sleep(wait_seconds)
                            wait_retry_account_id = account.id
                            continue
                        raise
                    account = next_account
                    self._store.save(pool)
                    continue
                if error.is_auth():
                    self._mark_auth_required(pool, account)
                    next_account = P.advance(
                        pool,
                        family,
                        exclude_id=account.id,
                        now_ms=self._now_ms(),
                    )
                    self._store.save(pool)
                    if next_account is None:
                        raise NoAccountAvailable(_AUTH_HINT) from error
                    account = next_account
                    continue
                raise
            finally:
                try:
                    http.close()
                except Exception:
                    logger.debug("agy-oauth: 关闭 HTTP 客户端失败", exc_info=True)
