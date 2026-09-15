"""transport 响应侧测试：SSE 解包、累积、finish/usage、tool_call 编码、错误分类。"""
import json

from agylink import transport as T


def _sse(*objs):
    return [f"data: {json.dumps(o)}" for o in objs] + [""]


def test_iter_sse_unwraps_response_and_skips_noise():
    lines = [": heartbeat", ""] + _sse({"response": {"a": 1}}, {"b": 2})
    assert list(T.iter_sse_events(lines)) == [{"a": 1}, {"b": 2}]


def test_accumulate_text_thought_and_function_call():
    events = [
        {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        {"candidates": [{"content": {"parts": [{"functionCall": {"name": "terminal", "args": {"cmd": "ls"}, "id": "fc1"},
                                               "thoughtSignature": "SIG"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15, "thoughtsTokenCount": 3}},
    ]
    acc = T.accumulate(events)
    assert acc.text == "Hello" and acc.reasoning == "thinking..."
    assert acc.tool_calls == [{"id": "call_fc1|SIG", "name": "terminal", "arguments": "{\"cmd\": \"ls\"}"}]
    assert acc.finish_reason == "tool_calls"
    assert acc.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "reasoning_tokens": 3}


def test_finish_reason_mapping_without_tools():
    assert T.accumulate([{"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]}]).finish_reason == "length"
    assert T.accumulate([{"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}]).finish_reason == "content_filter"
    assert T.accumulate([{"candidates": [{"finishReason": "OTHER", "content": {"parts": []}}]}]).finish_reason == "stop"


def test_function_call_without_id_or_sig_gets_generated_id():
    acc = T.accumulate([{"candidates": [{"content": {"parts": [{"functionCall": {"name": "f", "args": {}}}]}}]}])
    assert acc.tool_calls[0]["id"].startswith("call_") and len(acc.tool_calls[0]["id"]) > 10


def test_to_completion_shape():
    acc = T.Accumulated(text="hi", reasoning="r", finish_reason="tool_calls",
                        tool_calls=[{"id": "call_a|b", "name": "f", "arguments": "{}"}],
                        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "reasoning_tokens": 0})
    c = T.to_completion(acc, model="gemini-3-flash")
    msg = c.choices[0].message
    assert msg.content == "hi" and msg.reasoning_content == "r"
    assert msg.tool_calls[0].id == "call_a|b" and msg.tool_calls[0].function.name == "f"
    assert c.choices[0].finish_reason == "tool_calls" and c.usage.total_tokens == 3 and c.model == "gemini-3-flash"


def test_upstream_error_classification():
    e = T.UpstreamError(429, json.dumps({"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"resetTime": "2026-09-15T05:00:00Z"}]}}))
    assert e.is_quota() and not e.is_auth() and e.reset_time() == "2026-09-15T05:00:00Z"
    assert T.UpstreamError(401, "").is_auth()
    assert not T.UpstreamError(500, "boom").is_quota()


def test_upstream_error_message_is_redacted():
    err = T.UpstreamError(400, '{"error": {"message": "bad token ya29.SECRET"}}')
    msg = str(err)
    assert msg.startswith("upstream HTTP 400:")
    assert "<access-token>" in msg
    assert "ya29.SECRET" not in msg
