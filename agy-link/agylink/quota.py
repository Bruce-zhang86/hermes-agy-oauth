"""Cloud Code 内部接口：模型目录（fetchAvailableModels）与配额汇总（retrieveUserQuotaSummary）。

主机回退顺序与 dsh-agy-link 一致；只用生产主机。
"""
from __future__ import annotations

import os
from typing import Optional

import httpx

from agylink.pool import FAMILIES, family_of

HOSTS: tuple[str, ...] = (
    "https://daily-cloudcode-pa.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
)


def preferred_hosts(last_good: Optional[str] = None, *, hosts: Optional[tuple[str, ...]] = None) -> tuple[str, ...]:
    """把上次成功的 Cloud Code 主机排到最前，其余保持原顺序。

    Gemini 隐式缓存不跨主机；工具循环里每次都先打 daily 再打 prod，会把
    已经热起来的前缀缓存打散。粘滞到上次成功的主机，后续轮次更容易命中。
    参数 last_good：上次 streamGenerateContent 成功的完整主机 URL。
    参数 hosts：候选主机元组；默认使用模块常量 HOSTS。
    返回：重排后的主机 URL 元组；last_good 未知或不在列表里时返回原顺序。
    """
    ordered = hosts if hosts is not None else HOSTS
    if last_good and last_good in ordered:
        return (last_good,) + tuple(host for host in ordered if host != last_good)
    return tuple(ordered)

FALLBACK_MODELS: dict[str, tuple[str, ...]] = {
    "google": ("gemini-3.6-flash", "gemini-3.1-pro-high", "gemini-3-flash"),
    "anthropic": ("claude-sonnet-4-6", "claude-opus-4-6-thinking"),
    "openai": ("gpt-oss-120b-medium",),
}


def user_agent() -> str:
    """返回 Cloud Code 请求的 User-Agent 字符串。

    参数：无。
    返回：环境变量 HERMES_AGY_USER_AGENT 非空时取其值，否则 antigravity/1.2.2 windows/amd64。
    """
    return os.environ.get("HERMES_AGY_USER_AGENT", "").strip() or "antigravity/1.2.2 windows/amd64"


def fallback_model_ids() -> list[str]:
    """目录拉取失败时使用的兜底模型 id 列表。

    参数：无。
    返回：按 FAMILIES 顺序拼接 FALLBACK_MODELS 各族的模型 id。
    """
    return [m for fam in FAMILIES for m in FALLBACK_MODELS[fam]]


def post_internal(action: str, access_token: str, *, http: httpx.Client, body: Optional[dict] = None) -> Optional[dict]:
    """POST {host}/v1internal:{action}，按 HOSTS 依次尝试 Cloud Code 内部接口。

    参数 action：v1internal 动作名（如 fetchAvailableModels）。
    参数 access_token：OAuth 访问令牌。
    参数 http：httpx 客户端（测试注入 MockTransport）。
    参数 body：可选 JSON 请求体，默认空 dict。
    返回：2xx 时解析的 JSON dict；401/403 或全部主机失败时 None。
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": user_agent(),
    }
    for host in HOSTS:
        try:
            resp = http.post(f"{host}/v1internal:{action}", json=body or {}, headers=headers)
        except httpx.HTTPError:
            continue
        if resp.status_code in (401, 403):
            return None
        if 200 <= resp.status_code < 300:
            try:
                return resp.json()
            except ValueError:
                return None
    return None


def fetch_available_models(access_token: str, *, http: httpx.Client) -> Optional[dict]:
    """拉取 Cloud Code 模型目录（含各模型 quotaInfo）。

    参数 access_token：OAuth 访问令牌。
    参数 http：httpx 客户端。
    返回：fetchAvailableModels 响应 dict，失败时 None。
    """
    return post_internal("fetchAvailableModels", access_token, http=http)


def fetch_quota_summary(access_token: str, *, http: httpx.Client) -> Optional[dict]:
    """拉取用户 5h / 7d 双 bucket 配额汇总。

    参数 access_token：OAuth 访问令牌。
    参数 http：httpx 客户端。
    返回：retrieveUserQuotaSummary 响应 dict，失败时 None。
    """
    return post_internal("retrieveUserQuotaSummary", access_token, http=http)


def model_ids_from_available(payload: Optional[dict]) -> list[str]:
    """从 fetchAvailableModels 响应提取排序后的模型 id 列表。

    参数 payload：目录响应 dict，或 None。
    返回：models 键下所有 model id 的升序列表；无效 payload 时 []。
    """
    models = (payload or {}).get("models")
    if not isinstance(models, dict):
        return []
    return sorted(str(k) for k in models.keys())


def family_quotas(summary: Optional[dict], available: Optional[dict], *, now_ms: int) -> dict[str, dict]:
    """把配额汇总与模型目录聚合成按族的配额 dict。

    参数 summary：retrieveUserQuotaSummary 响应，或 None。
    参数 available：fetchAvailableModels 响应，或 None。
    参数 now_ms：写入 updatedAt 的毫秒时间戳。
    返回：族名 → 配额字段 dict（remainingFraction、models 等，对齐 DSH pool.json quotas）。
    """
    out: dict[str, dict] = {}
    for group in (summary or {}).get("groups", []) or []:
        name = f"{group.get('displayName', '')} {group.get('description', '')}".lower()
        if "gemini" in name:
            fams = ["google"]
        elif "claude" in name or "gpt" in name:
            fams = ["anthropic", "openai"]
        else:
            continue
        five: dict = {}
        weekly: dict = {}
        for b in group.get("buckets", []) or []:
            w = str(b.get("window") or b.get("bucketId") or "").lower()
            if "5h" in w:
                five = b
            elif "week" in w:
                weekly = b
        for fam in fams:
            out[fam] = {
                "remainingFraction": five.get("remainingFraction"),
                "resetTime": five.get("resetTime"),
                "weeklyFraction": weekly.get("remainingFraction"),
                "weeklyResetTime": weekly.get("resetTime"),
                "description": group.get("description"),
                "updatedAt": now_ms,
                "models": [],
            }
    for model_id, entry in ((available or {}).get("models") or {}).items():
        fam = family_of(model_id)
        if fam is None or not isinstance(entry, dict):
            continue
        qi = entry.get("quotaInfo") or {}
        rem = qi.get("remainingFraction")
        if not isinstance(rem, (int, float)):
            continue
        out.setdefault(fam, {"remainingFraction": rem, "resetTime": qi.get("resetTime"), "weeklyFraction": None,
                             "weeklyResetTime": None, "description": None, "updatedAt": now_ms, "models": []})
        out[fam]["models"].append({"modelId": model_id, "displayName": entry.get("displayName") or model_id,
                                   "remainingFraction": rem, "resetTime": qi.get("resetTime")})
    return out
