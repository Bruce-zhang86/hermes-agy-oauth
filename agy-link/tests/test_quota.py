"""quota 模块测试：主机回退、401 停止、模型列表、配额汇总、兜底目录。"""
import httpx

from agylink import quota as Q


def _http(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_post_internal_falls_back_to_second_host(monkeypatch):
    monkeypatch.delenv("HERMES_AGY_USER_AGENT", raising=False)
    calls = []

    def handler(req):
        calls.append(req.url.host)
        if req.url.host.startswith("daily"):
            return httpx.Response(500)
        assert req.headers["User-Agent"] == "antigravity/1.2.2 windows/amd64"
        assert req.headers["Authorization"] == "Bearer ya29.t"
        return httpx.Response(200, json={"ok": True})

    assert Q.post_internal("fetchAvailableModels", "ya29.t", http=_http(handler)) == {"ok": True}
    assert calls == ["daily-cloudcode-pa.googleapis.com", "cloudcode-pa.googleapis.com"]


def test_user_agent_default_and_env_override(monkeypatch):
    monkeypatch.delenv("HERMES_AGY_USER_AGENT", raising=False)
    assert Q.user_agent() == "antigravity/1.2.2 windows/amd64"
    monkeypatch.setenv("HERMES_AGY_USER_AGENT", "  custom-ua/9.9 linux/arm64  ")
    assert Q.user_agent() == "custom-ua/9.9 linux/arm64"
    monkeypatch.setenv("HERMES_AGY_USER_AGENT", "   ")
    assert Q.user_agent() == "antigravity/1.2.2 windows/amd64"


def test_post_internal_stops_on_401():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(401)

    assert Q.post_internal("fetchAvailableModels", "x", http=_http(handler)) is None
    assert len(calls) == 1


def test_model_ids_from_available_and_fallback():
    payload = {"models": {"gemini-3-flash": {"displayName": "G"}, "claude-sonnet-4-6": {}}}
    assert Q.model_ids_from_available(payload) == ["claude-sonnet-4-6", "gemini-3-flash"]
    assert Q.model_ids_from_available(None) == []
    ids = Q.fallback_model_ids()
    assert "gemini-3.6-flash" in ids and "claude-sonnet-4-6" in ids and "gpt-oss-120b-medium" in ids


def test_family_quotas_from_summary_and_models():
    summary = {"groups": [
        {"displayName": "Gemini", "description": "Gemini Flash, Gemini Pro",
         "buckets": [{"window": "5h", "remainingFraction": 0.7, "resetTime": "2026-09-15T05:00:00Z"},
                     {"window": "weekly", "remainingFraction": 0.6, "resetTime": "2026-09-18T00:00:00Z"}]},
        {"displayName": "Claude", "description": "Claude, GPT-OSS",
         "buckets": [{"window": "5h", "remainingFraction": 1.0, "resetTime": "2026-09-15T08:00:00Z"}]},
    ]}
    available = {"models": {"gemini-3-flash": {"quotaInfo": {"remainingFraction": 0.7, "resetTime": "x"}}}}
    q = Q.family_quotas(summary, available, now_ms=5)
    assert q["google"]["remainingFraction"] == 0.7 and q["google"]["weeklyFraction"] == 0.6
    assert q["anthropic"]["remainingFraction"] == 1.0 and q["openai"]["remainingFraction"] == 1.0
    assert q["google"]["models"][0]["modelId"] == "gemini-3-flash"
    assert q["google"]["updatedAt"] == 5
