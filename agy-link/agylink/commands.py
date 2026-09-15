"""`hermes agy <sub>` 与 `/agy <sub>` 共用的子命令实现，全部返回脱敏纯文本。"""
from __future__ import annotations

import time
import webbrowser
from pathlib import Path
from typing import Callable, Literal, Optional

import httpx

from agylink import oauth, quota
from agylink import pool as P
from agylink.client import default_http_factory
from agylink.paths import accounts_dir, dsh_accounts_dir, pool_file
from agylink.redact import redact
from agylink.token_store import is_expiring, read_token, write_token

HELP = """用法：hermes agy <子命令>   或   /agy <子命令>
  auth [--alias 名称] [--proxy URL]   浏览器 Google 登录，新增 Hermes 账号
  auth-code <授权码或回调URL>          回环失败 / 无浏览器时的粘贴兜底
  import-dsh                          导入 ~/.dsh/agy-accounts 号池（只读引用）
  status | pool                       账号、配额、冷却、当前 active
  models                              拉取可用模型列表
  remove <id>                         删除账号（dsh 来源只取消登记）
  refresh-quota [id]                  刷新配额缓存
  help                                本帮助"""

_AUTH_TIMEOUT_S = 300
TokenReason = Literal["ok", "missing", "invalid_grant", "transient"]


def _now_ms() -> int:
    """返回当前 Unix 毫秒时间戳。

    参数：无。
    返回：当前时间的整数毫秒值。
    """
    return int(time.time() * 1000)


def _pct(value: object) -> str:
    """把 0 到 1 的数值格式化为百分比。

    参数 value：配额比例；非数字值显示为短横线。
    返回：四舍五入后的百分比文本或 ``-``。
    """
    return f"{round(float(value) * 100)}%" if isinstance(value, (int, float)) else "-"


def _store() -> P.PoolStore:
    """构造 Hermes 号池存储。

    参数：无。
    返回：绑定当前 pool.json 的 ``PoolStore``。
    """
    return P.PoolStore(pool_file())


def cmd_status(*, current_ms: Optional[int] = None) -> str:
    """生成号池状态表。

    参数 current_ms：可选的当前毫秒时间，用于判断冷却状态。
    返回：可直接展示给用户的状态文本。
    """
    pool = _store().load()
    if not pool.accounts:
        return "号池为空。运行 `hermes agy auth` 登录，或 `hermes agy import-dsh` 导入 DSH 号池。"
    now = _now_ms() if current_ms is None else current_ms
    lines = [
        "id | alias | source | email | enabled | authRequired | "
        "google 5h/7d | anthropic 5h/7d | openai 5h/7d | cooldown"
    ]
    for account in pool.accounts:
        needs_login = account.auth_required
        if not account.system_home and not account.dir:
            needs_login = True
        cells = [
            account.id,
            account.alias,
            account.source,
            account.email or "-",
            "yes" if account.enabled else "no",
            "yes" if needs_login else "no",
        ]
        for family in P.FAMILIES:
            family_quota = account.quotas.get(family) or {}
            cells.append(
                f"{_pct(family_quota.get('remainingFraction'))}/"
                f"{_pct(family_quota.get('weeklyFraction'))}"
            )
        cooling = [family for family in P.FAMILIES if P.is_cooling(account, family, now)]
        cells.append(",".join(cooling) or "-")
        lines.append(" | ".join(cells))
    active = ", ".join(
        f"{family}={pool.active_account_ids.get(family, '-')}" for family in P.FAMILIES
    )
    lines.append(f"active: {active}")
    return "\n".join(lines)


def cmd_import_dsh() -> str:
    """把 DSH 号池只读引用导入 Hermes 号池。

    参数：无。
    返回：导入和跳过数量；此操作绝不写入 DSH 文件。
    """
    path = dsh_accounts_dir() / "pool.json"
    if not path.exists():
        return f"未找到 DSH 号池：{path}"
    store = _store()
    pool = store.load()
    imported, skipped = P.import_dsh(pool, path)
    store.save(pool)
    return f"导入 {imported} 个账号，跳过 {skipped} 个已存在。DSH 文件未改动。"


def cmd_remove(account_id: str) -> str:
    """删除账号登记，并仅删除安全范围内的 Hermes 来源目录。

    参数 account_id：待删除账号的唯一标识。
    返回：删除、取消登记或找不到账号的结果文本。
    """
    store = _store()
    pool = store.load()
    account = next((item for item in pool.accounts if item.id == account_id), None)
    if account is None:
        return f"找不到账号 {account_id}"
    pool.accounts.remove(account)
    for family, active_id in list(pool.active_account_ids.items()):
        if active_id == account.id:
            del pool.active_account_ids[family]
    store.save(pool)
    if account.source == "dsh":
        return f"已取消登记 {account.id}（DSH 目录与 token 未改动）"
    try:
        P.delete_account_dir(account, accounts_dir())
    except ValueError as exc:
        return f"已取消登记 {account.id}，但未删目录：{exc}"
    return f"已删除 {account.id} 及其目录"


def _token_for(
    account: P.Account,
    http: httpx.Client,
) -> tuple[Optional[str], TokenReason]:
    """读取账号令牌，并明确区分缺失、失效和暂时性失败。

    参数 account：待读取令牌的账号。
    参数 http：用于刷新 OAuth 令牌的 HTTP 客户端。
    返回：``(access_token, reason)``；reason 为 ``ok``、``missing``、
    ``invalid_grant`` 或 ``transient``。
    """
    try:
        token_path = account.token_path()
    except ValueError:
        return None, "missing"
    try:
        token_path.stat()
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "transient"
    try:
        token = read_token(token_path)
    except OSError:
        return None, "transient"
    if token is None:
        return None, "missing"
    if not is_expiring(token):
        return token.access_token, "ok"
    try:
        fresh = oauth.refresh_access_token(token.refresh_token, http=http)
        write_token(token_path, fresh)
    except oauth.InvalidGrant:
        return None, "invalid_grant"
    except (oauth.OAuthError, httpx.HTTPError, OSError):
        return None, "transient"
    return fresh.access_token, "ok"


def cmd_refresh_quota(
    account_id: Optional[str],
    *,
    http_factory: Callable = default_http_factory,
    current_ms: Optional[int] = None,
) -> str:
    """刷新一个或全部账号的配额缓存。

    参数 account_id：指定账号 id；``None`` 表示全部账号。
    参数 http_factory：按代理 URL 创建 HTTP 客户端的工厂。
    参数 current_ms：可选的当前毫秒时间，供配额缓存记录使用。
    返回：逐账号刷新结果文本。
    """
    store = _store()
    pool = store.load()
    targets = [account for account in pool.accounts if account_id in (None, account.id)]
    if not targets:
        return f"找不到账号 {account_id}" if account_id else "号池为空"
    lines: list[str] = []
    now = _now_ms() if current_ms is None else current_ms
    for account in targets:
        try:
            with http_factory(account.proxy_url) as http:
                token, token_reason = _token_for(account, http)
                if token is None:
                    if token_reason in ("missing", "invalid_grant"):
                        account.auth_required = True
                        lines.append(
                            f"{account.id}: 凭证不可用，已标记 authRequired"
                        )
                    else:
                        lines.append(
                            f"{account.id}: 拉取配额失败：{redact(token_reason)}"
                        )
                    continue
                summary = quota.fetch_quota_summary(token, http=http)
                available = quota.fetch_available_models(token, http=http)
        except (oauth.OAuthError, httpx.HTTPError, OSError, ValueError) as exc:
            lines.append(f"{account.id}: 拉取配额失败：{redact(str(exc))}")
            continue
        family_quota = quota.family_quotas(summary, available, now_ms=now)
        if family_quota:
            account.quotas = family_quota
            account.auth_required = False
            values = ", ".join(
                f"{family}={_pct(family_quota.get(family, {}).get('remainingFraction'))}"
                for family in P.FAMILIES
            )
            lines.append(f"{account.id}: {values}")
        else:
            lines.append(f"{account.id}: 拉取配额失败")
    store.save(pool)
    return "\n".join(lines)


def cmd_models(
    *,
    http_factory: Callable = default_http_factory,
    current_ms: Optional[int] = None,
) -> str:
    """拉取模型列表，无可用账号时输出本地兜底目录。

    参数 http_factory：按代理 URL 创建 HTTP 客户端的工厂。
    参数 current_ms：可选的当前毫秒时间，用于账号选择。
    返回：每行一个模型 id，或带说明的兜底模型目录。
    """
    pool = _store().load()
    now = _now_ms() if current_ms is None else current_ms
    account = P.select_account(pool, "google", now)
    if account is not None:
        try:
            with http_factory(account.proxy_url) as http:
                token, _ = _token_for(account, http)
                model_ids = (
                    quota.model_ids_from_available(
                        quota.fetch_available_models(token, http=http)
                    )
                    if token
                    else []
                )
        except (oauth.OAuthError, httpx.HTTPError, OSError, ValueError):
            model_ids = []
        if model_ids:
            return "\n".join(model_ids)
    return "（兜底目录，尚无可用账号或拉取失败）\n" + "\n".join(
        quota.fallback_model_ids()
    )


def cmd_auth(
    alias: Optional[str],
    proxy_url: Optional[str],
    *,
    open_browser: bool,
    http_factory: Callable = default_http_factory,
    current_ms: Optional[int] = None,
) -> str:
    """发起 PKCE 登录；无浏览器模式仅返回授权 URL。

    参数 alias：新账号显示名称。
    参数 proxy_url：该账号使用的可选代理 URL。
    参数 open_browser：是否打开浏览器并等待回环授权。
    参数 http_factory：按代理 URL 创建 HTTP 客户端的工厂。
    参数 current_ms：可选的当前毫秒时间，用于账号 id 和 pending 过期时间。
    返回：授权指引、登录结果或可读错误文本。
    """
    now = _now_ms() if current_ms is None else current_ms
    account_dir = accounts_dir() / P.new_account_id(now)
    pkce = oauth.generate_pkce()
    state = oauth.generate_pkce().verifier[:24]
    pending = oauth.PendingAuth(
        pkce.verifier,
        state,
        alias or account_dir.name,
        proxy_url,
        str(account_dir),
        expires_at_ms=now + _AUTH_TIMEOUT_S * 1000,
    )
    oauth.save_pending(pending)
    url = oauth.build_authorize_url(pkce.challenge, state)
    if not open_browser:
        return (
            "请在浏览器打开以下地址完成 Google 授权，然后运行 "
            "`hermes agy auth-code <回调URL或授权码>`：\n"
            f"{url}"
        )
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        code = oauth.wait_for_loopback_code(state, timeout_s=_AUTH_TIMEOUT_S)
    except oauth.LoopbackBusy:
        return (
            f"回环端口 {oauth.CALLBACK_PORT} 被占用。请在浏览器打开：\n{url}\n"
            "授权后运行 `hermes agy auth-code <回调URL>`。"
        )
    except oauth.LoopbackTimeout:
        oauth.clear_pending()
        return "等待授权超时（5 分钟）。请重新运行 `hermes agy auth`。"
    except oauth.OAuthError as exc:
        oauth.clear_pending()
        return f"授权失败：{redact(str(exc))}"
    return _complete_login(
        pending,
        code,
        http_factory=http_factory,
        created_at_ms=now,
    )


def cmd_auth_code(
    raw: str,
    *,
    http_factory: Callable = default_http_factory,
    current_ms: Optional[int] = None,
) -> str:
    """使用粘贴的授权码或回调 URL 完成进行中的登录。

    参数 raw：用户粘贴的原始授权码或完整回调 URL。
    参数 http_factory：按代理 URL 创建 HTTP 客户端的工厂。
    参数 current_ms：可选的当前毫秒时间，用于 pending 校验和账号创建时间。
    返回：登录结果或可读错误文本。
    """
    pending = oauth.load_pending(now_ms=current_ms)
    if pending is None:
        return "没有进行中的登录。请先运行 `hermes agy auth`。"
    try:
        code, state = oauth.parse_code_input(raw)
    except oauth.OAuthError as exc:
        return f"无法解析授权码：{redact(str(exc))}"
    if state and state != pending.state:
        return "state 不匹配，请重新运行 `hermes agy auth`。"
    created_at = _now_ms() if current_ms is None else current_ms
    return _complete_login(
        pending,
        code,
        http_factory=http_factory,
        created_at_ms=created_at,
    )


def _complete_login(
    pending: oauth.PendingAuth,
    code: str,
    *,
    http_factory: Callable,
    created_at_ms: int,
) -> str:
    """完成换票、令牌写入、账号登记和 pending 清理。

    参数 pending：此前保存的 PKCE 登录上下文。
    参数 code：Google 返回的授权码；不得写入输出或日志。
    参数 http_factory：按代理 URL 创建 HTTP 客户端的工厂。
    参数 created_at_ms：新账号的创建毫秒时间。
    返回：登录成功或脱敏后的换票失败文本。
    """
    account_dir = Path(pending.account_dir)
    try:
        with http_factory(pending.proxy_url) as http:
            token = oauth.exchange_code(code, pending.verifier, http=http)
            email = oauth.fetch_userinfo(token.access_token, http=http).get("email")
    except (oauth.OAuthError, httpx.HTTPError, OSError, ValueError) as exc:
        return f"换票失败：{redact(str(exc))}"
    account = P.Account(
        id=account_dir.name,
        alias=pending.alias,
        dir=str(account_dir),
        system_home=False,
        source="hermes",
        email=email,
        enabled=True,
        proxy_url=pending.proxy_url,
        auth_required=False,
        created_at=created_at_ms,
        last_used_at=0,
        project_id=None,
        cooldowns={},
        quotas={},
    )
    write_token(account.token_path(), token)
    store = _store()
    pool = store.load()
    pool.accounts.append(account)
    store.save(pool)
    oauth.clear_pending()
    return (
        f"登录成功：{account.id}（{email or '未知邮箱'}）。"
        "运行 `hermes agy status` 查看。"
    )


def _parse_auth_options(arguments: list[str]) -> tuple[Optional[str], Optional[str]]:
    """解析 auth 的 alias 与 proxy 选项。

    参数 arguments：auth 子命令后的原始参数列表。
    返回：``(alias, proxy_url)``；缺少选项值时对应值保持 ``None``。
    """
    alias: Optional[str] = None
    proxy: Optional[str] = None
    index = 0
    while index < len(arguments):
        if arguments[index] == "--alias" and index + 1 < len(arguments):
            alias = arguments[index + 1]
            index += 2
            continue
        if arguments[index] == "--proxy" and index + 1 < len(arguments):
            proxy = arguments[index + 1]
            index += 2
            continue
        index += 1
    return alias, proxy


def dispatch(
    argv: list[str],
    *,
    open_browser: bool = True,
    http_factory: Optional[Callable] = None,
    now_ms: Optional[Callable[[], int]] = None,
) -> str:
    """解析参数并执行 agy 子命令，确保命令边界不向用户抛 traceback。

    参数 argv：不含 ``agy`` 本身的命令行参数。
    参数 open_browser：auth 是否打开浏览器并等待回环。
    参数 http_factory：测试或调用方注入的 HTTP 客户端工厂。
    参数 now_ms：测试注入的当前毫秒时间函数。
    返回：经过敏感信息涂抹的纯文本命令结果。
    """
    factory = http_factory or default_http_factory
    arguments = [item for item in (argv or []) if item is not None]
    subcommand = arguments[0] if arguments else "help"
    rest = arguments[1:]
    current_ms = now_ms() if now_ms is not None else None
    try:
        if subcommand in ("help", "-h", "--help"):
            result = HELP
        elif subcommand in ("status", "pool"):
            result = cmd_status(current_ms=current_ms)
        elif subcommand == "import-dsh":
            result = cmd_import_dsh()
        elif subcommand == "remove":
            result = cmd_remove(rest[0]) if rest else "用法：remove <id>"
        elif subcommand == "refresh-quota":
            result = cmd_refresh_quota(
                rest[0] if rest else None,
                http_factory=factory,
                current_ms=current_ms,
            )
        elif subcommand == "models":
            result = cmd_models(http_factory=factory, current_ms=current_ms)
        elif subcommand == "auth":
            alias, proxy = _parse_auth_options(rest)
            result = cmd_auth(
                alias,
                proxy,
                open_browser=open_browser,
                http_factory=factory,
                current_ms=current_ms,
            )
        elif subcommand == "auth-code":
            result = (
                cmd_auth_code(
                    " ".join(rest),
                    http_factory=factory,
                    current_ms=current_ms,
                )
                if rest
                else "用法：auth-code <授权码或回调URL>"
            )
        else:
            result = f"未知子命令：{subcommand}\n{HELP}"
        return redact(result)
    except Exception as exc:
        return f"执行失败：{redact(str(exc))}"
