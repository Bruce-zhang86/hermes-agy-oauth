"""Google OAuth（Antigravity 消费者客户端）：PKCE、授权 URL、换票、刷新、pending、回环。

与 dsh-agy-link / 官方 agy 使用同一公开客户端与同一五条 scope。
注意：不得加入 openid scope，否则该客户端的同意页会挂起。
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from agylink.paths import pending_auth_file
from agylink.token_store import TokenSet, token_from_response

SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
)
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"
CALLBACK_PORT = 51121
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}/oauth-callback"

# 公开仓库不内置 Google OAuth 客户端明文（GitHub push protection 会拦截）。
# 与官方 agy CLI 相同的消费者客户端请用环境变量注入：
# HERMES_AGY_CLIENT_ID / HERMES_AGY_CLIENT_SECRET（或 AGY_CLIENT_ID / AGY_CLIENT_SECRET）。


class OAuthError(Exception):
    """OAuth 流程错误基类。"""


class InvalidGrant(OAuthError):
    """refresh token 已吊销或无效（Google 返回 invalid_grant）。"""


class LoopbackBusy(OAuthError):
    """回环端口被占用。"""


class LoopbackTimeout(OAuthError):
    """回环等待授权码超时。"""


@dataclass(frozen=True)
class Pkce:
    """PKCE 对：verifier 留本地，challenge 上送。"""

    verifier: str
    challenge: str


def client_credentials() -> tuple[str, str]:
    """返回 OAuth 客户端凭证。

    参数：无。
    返回：(client_id, client_secret)。
    读取顺序：HERMES_AGY_CLIENT_ID/SECRET，其次 AGY_CLIENT_ID/SECRET。
    均未配置时抛出 OAuthError，避免把客户端明文写进公开仓库。
    """
    client_id = (
        os.environ.get("HERMES_AGY_CLIENT_ID", "").strip()
        or os.environ.get("AGY_CLIENT_ID", "").strip()
    )
    client_secret = (
        os.environ.get("HERMES_AGY_CLIENT_SECRET", "").strip()
        or os.environ.get("AGY_CLIENT_SECRET", "").strip()
    )
    if not client_id or not client_secret:
        raise OAuthError(
            "未配置 Antigravity OAuth 客户端。请设置环境变量 "
            "HERMES_AGY_CLIENT_ID 与 HERMES_AGY_CLIENT_SECRET"
            "（与官方 agy CLI 使用同一套公开消费者客户端）。"
        )
    return client_id, client_secret


def generate_pkce() -> Pkce:
    """生成 S256 PKCE 对。

    参数：无。
    返回：Pkce(verifier, challenge)；verifier 为 32 字节随机数 base64url，challenge 为其 SHA-256 base64url（均去 = 填充）。
    """
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return Pkce(verifier, challenge)


def build_authorize_url(challenge: str, state: str) -> str:
    """拼接 Google 授权页 URL（access_type=offline、prompt=consent、S256）。

    参数 challenge：PKCE challenge。
    参数 state：防 CSRF 的随机 state，回调时原样返回。
    返回：完整的授权页 URL 字符串。
    """
    client_id, _ = client_credentials()
    query = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTHORIZE_URL}?{urlencode(query)}"


def parse_code_input(raw: str) -> tuple[str, Optional[str]]:
    """解析用户粘贴的授权码或回调 URL。

    参数 raw：原始授权码，或含 code= 参数的回调 URL（可省略 http:// 以 localhost 开头）。
    返回：(code, state)；粘贴的是裸授权码时 state 为 None。空输入或 URL 缺 code 抛 OAuthError。
    """
    text = (raw or "").strip()
    if not text:
        raise OAuthError("授权码为空")
    if "://" in text or text.startswith("localhost"):
        q = parse_qs(urlparse(text if "://" in text else "http://" + text).query)
        code = (q.get("code") or [""])[0]
        if not code:
            raise OAuthError("URL 中没有 code 参数")
        return code, (q.get("state") or [None])[0]
    return text, None


def _post_token_form(fields: dict, *, http: httpx.Client) -> dict:
    """向 Google token 端点 POST 表单，自动附上 client 凭证。

    参数 fields：grant_type 及其配套字段（code / refresh_token 等）。
    参数 http：httpx.Client（测试可注入 MockTransport）。
    返回：解析后的 JSON dict（必含 access_token）。
    异常：4xx/5xx 且 error 为 invalid_grant 抛 InvalidGrant；其他非 2xx 或缺 access_token 抛 OAuthError。
    """
    client_id, client_secret = client_credentials()
    body = {"client_id": client_id, "client_secret": client_secret, **fields}
    resp = http.post(TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if resp.status_code >= 400:
        err = payload.get("error")
        code = err if isinstance(err, str) else (err or {}).get("status") if isinstance(err, dict) else None
        desc = payload.get("error_description") or ""
        if code == "invalid_grant":
            raise InvalidGrant(f"invalid_grant: {desc}".strip())
        raise OAuthError(f"token endpoint {resp.status_code}: {code or ''} {desc}".strip())
    if not payload.get("access_token"):
        raise OAuthError("token endpoint 没有返回 access_token")
    return payload


def exchange_code(code: str, verifier: str, *, http: httpx.Client) -> TokenSet:
    """用授权码 + PKCE verifier 换取 TokenSet。

    参数 code：Google 返回的授权码（会 strip）。
    参数 verifier：与授权 URL 中 challenge 配对的 PKCE verifier。
    参数 http：httpx.Client。
    返回：TokenSet（含 access_token、refresh_token、expiry）。
    """
    payload = _post_token_form(
        {"grant_type": "authorization_code", "code": code.strip(), "redirect_uri": REDIRECT_URI, "code_verifier": verifier},
        http=http,
    )
    return token_from_response(payload)


def refresh_access_token(refresh_token: str, *, http: httpx.Client) -> TokenSet:
    """用 refresh token 换取新的 access token。

    参数 refresh_token：现有 refresh token。
    参数 http：httpx.Client。
    返回：TokenSet；Google 不回传 refresh_token 时沿用传入的旧值。吊销时抛 InvalidGrant。
    """
    payload = _post_token_form({"grant_type": "refresh_token", "refresh_token": refresh_token}, http=http)
    return token_from_response(payload, fallback_refresh=refresh_token)


def fetch_userinfo(access_token: str, *, http: httpx.Client) -> dict:
    """拉取 Google userinfo（含 email）。

    参数 access_token：OAuth 访问令牌。
    参数 http：httpx.Client。
    返回：userinfo dict；非 200、网络错误或 JSON 损坏时返回空 dict，不抛异常。
    """
    try:
        resp = http.get(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        return resp.json() if resp.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return {}


@dataclass(frozen=True)
class PendingAuth:
    """进行中的登录状态，落盘供 auth-code 兜底使用。"""

    verifier: str
    state: str
    alias: str
    proxy_url: Optional[str]
    account_dir: str
    expires_at_ms: int


def save_pending(p: PendingAuth) -> None:
    """写 .pending-auth.json（覆盖上一份），自动创建父目录。

    参数 p：待落盘的 PendingAuth。
    返回：无。
    """
    path = pending_auth_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(p)), encoding="utf-8")


def load_pending(*, now_ms: Optional[int] = None) -> Optional[PendingAuth]:
    """读取进行中的登录状态。

    参数 now_ms：当前毫秒时间戳，默认取系统时间；测试可注入。
    返回：PendingAuth；文件不存在、JSON/字段损坏、expires_at_ms 非整数或已过期时返回 None。
    """
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    try:
        data = json.loads(pending_auth_file().read_text(encoding="utf-8"))
        p = PendingAuth(**data)
        if not isinstance(p.expires_at_ms, int) or isinstance(p.expires_at_ms, bool):
            return None
    except (OSError, ValueError, TypeError):
        return None
    return p if p.expires_at_ms > now_ms else None


def clear_pending() -> None:
    """删除 pending 文件。

    参数：无。
    返回：无；文件不存在或删除失败（OSError）时静默忽略。
    """
    try:
        pending_auth_file().unlink()
    except OSError:
        pass


def wait_for_loopback_code(expected_state: str, *, timeout_s: float, port: int = CALLBACK_PORT) -> str:
    """在 127.0.0.1:port 监听一次 /oauth-callback 并取回授权码。

    参数 expected_state：授权 URL 中下发的 state；不匹配的请求被拒绝并继续等待。
    参数 timeout_s：最长等待秒数。
    参数 port：监听端口，默认 CALLBACK_PORT。
    返回：授权码字符串。
    异常：端口占用抛 LoopbackBusy；超时抛 LoopbackTimeout；Google 回传 error 参数抛 OAuthError。
    """
    result: dict[str, str] = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        """一次性回调处理器：只接受 /oauth-callback，其余路径 404。"""

        def log_message(self, *_):
            """静默：不输出默认访问日志。

            参数 *_：BaseHTTPRequestHandler 传入的格式串与参数，忽略。
            返回：无。
            """
            return

        def do_GET(self):  # noqa: N802
            """处理 /oauth-callback 的 GET 请求。

            参数：无（从 self.path 读取查询串）。
            返回：无；校验 state 后把 code 或 error 写入外层 result 并置位 done。
            """
            parsed = urlparse(self.path)
            q = parse_qs(parsed.query)
            if parsed.path != "/oauth-callback":
                self.send_response(404); self.end_headers(); return
            if (q.get("state") or [""])[0] != expected_state:
                self._reply(400, "State mismatch"); return
            if q.get("error"):
                self._reply(400, "Authorization failed"); result["error"] = q["error"][0]; done.set(); return
            if not q.get("code"):
                self._reply(400, "State mismatch"); return
            result["code"] = q["code"][0]
            self._reply(200, "Login complete. You can close this tab.")
            done.set()

        def _reply(self, status: int, text: str):
            """发送一段简短的 HTML 响应。

            参数 status：HTTP 状态码。
            参数 text：显示在页面上的提示文本。
            返回：无。
            """
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<h2>{text}</h2>".encode())

    try:
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise LoopbackBusy(f"端口 {port} 被占用：{exc}") from exc
    server.timeout = 0.5
    deadline = time.monotonic() + timeout_s
    try:
        while not done.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if "code" in result:
        return result["code"]
    if "error" in result:
        raise OAuthError(f"授权被拒绝：{result['error']}")
    raise LoopbackTimeout("等待浏览器授权超时")
