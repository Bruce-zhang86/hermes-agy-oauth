"""transport 请求映射测试：envelope、system、工具往返、thoughtSignature、thinking。"""
import json
import threading

from agylink import transport as T


def test_tool_call_id_roundtrip():
    tid = T.encode_tool_call_id("fc1", "sigABC")
    assert tid == "call_fc1|sigABC"
    assert T.decode_tool_call_id(tid) == ("fc1", "sigABC")
    assert T.decode_tool_call_id("call_plain") == ("plain", "")
    assert T.decode_tool_call_id("weird") == ("", "")


def test_headers(monkeypatch):
    monkeypatch.delenv("HERMES_AGY_USER_AGENT", raising=False)
    h = T.request_headers("ya29.t")
    assert h["Authorization"] == "Bearer ya29.t"
    assert h["Content-Type"] == "application/json"
    assert h["User-Agent"] == "antigravity/1.2.2 windows/amd64"
    assert h["X-Goog-Api-Client"] == "google-cloud-sdk vscode_cloudshelleditor/0.1"
    assert json.loads(h["Client-Metadata"]) == {"ideType": "ANTIGRAVITY", "platform": "WINDOWS", "pluginType": "GEMINI"}


def test_envelope_basic_shape_and_system():
    env = T.build_envelope(project_id="p1", model="gemini-3-flash", request_id="agent-x",
                           messages=[{"role": "system", "content": "S1"}, {"role": "system", "content": "S2"},
                                     {"role": "user", "content": "hi"}],
                           tools=None, tool_choice=None, reasoning_effort=None)
    assert env["project"] == "p1" and env["model"] == "gemini-3-flash"
    assert env["requestType"] == "agent" and env["userAgent"] == "antigravity" and env["requestId"] == "agent-x"
    req = env["request"]
    assert req["systemInstruction"] == {"parts": [{"text": "S1\n\nS2"}]}
    assert req["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert [s["category"] for s in req["safetySettings"]] == [
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    assert "tools" not in req and "thinkingConfig" not in req.get("generationConfig", {})


def test_envelope_default_system_when_missing():
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=[{"role": "user", "content": "x"}],
                           tools=None, tool_choice=None, reasoning_effort=None)
    assert env["request"]["systemInstruction"]["parts"][0]["text"] == "You are a helpful AI assistant."


def test_tools_and_tool_roundtrip_with_signature():
    tools = [{"type": "function", "function": {"name": "terminal", "description": "run",
              "parameters": {"$schema": "x", "type": "object", "additionalProperties": False,
                             "properties": {"cmd": {"type": "string"}}}}}]
    msgs = [
        {"role": "user", "content": "ls"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_fc9|SIG", "type": "function", "function": {"name": "terminal", "arguments": "{\"cmd\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "call_fc9|SIG", "content": "{\"out\": \"a\"}"},
        {"role": "tool", "tool_call_id": "call_fc9|SIG", "name": "terminal", "content": "plain text"},
    ]
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=msgs, tools=tools,
                           tool_choice="auto", reasoning_effort=None)
    req = env["request"]
    decl = req["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "terminal" and "$schema" not in decl["parameters"] and "additionalProperties" not in decl["parameters"]
    assert req["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}
    model_turn = req["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["functionCall"] == {"name": "terminal", "args": {"cmd": "ls"}, "id": "fc9"}
    assert model_turn["parts"][0]["thoughtSignature"] == "SIG"
    fn_turn = req["contents"][2]
    assert fn_turn["role"] == "user"
    assert fn_turn["parts"][0]["functionResponse"] == {"name": "terminal", "id": "fc9", "response": {"out": "a"}}
    # 连续 tool 消息合并进同一 user 轮
    assert len(req["contents"]) == 3
    assert fn_turn["parts"][1]["functionResponse"]["response"] == {"result": "plain text"}


def test_missing_signature_uses_skip_marker_and_tool_choice_modes():
    msgs = [{"role": "assistant", "tool_calls": [{"id": "call_x", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]}]
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=msgs, tools=[
        {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
        tool_choice={"type": "function", "function": {"name": "f"}}, reasoning_effort=None)
    assert env["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert env["request"]["toolConfig"] == {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["f"]}}
    env2 = T.build_envelope(project_id="p", model="gemini-3-flash", messages=[{"role": "user", "content": "x"}],
                            tools=None, tool_choice="none", reasoning_effort=None)
    assert env2["request"]["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"


def test_thinking_config_by_family():
    g = T.build_envelope(project_id="p", model="gemini-3.1-pro-high", messages=[{"role": "user", "content": "x"}],
                         tools=None, tool_choice=None, reasoning_effort="high", max_tokens=100, temperature=0.2)
    gc = g["request"]["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": "high", "includeThoughts": True}
    assert gc["maxOutputTokens"] == 100 and gc["temperature"] == 0.2
    c = T.build_envelope(project_id="p", model="claude-opus-4-6-thinking", messages=[{"role": "user", "content": "x"}],
                         tools=None, tool_choice=None, reasoning_effort="high")
    assert c["request"]["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}
    s = T.build_envelope(project_id="p", model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
                         tools=None, tool_choice=None, reasoning_effort="high")
    assert "thinkingConfig" not in s["request"].get("generationConfig", {})


def test_tool_result_non_dict_json_wrapped_as_text():
    msgs = [
        {"role": "tool", "tool_call_id": "call_x", "name": "f", "content": "[1, 2]"},
        {"role": "tool", "tool_call_id": "call_y", "name": "f", "content": "{\"a\": 1}"},
    ]
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=msgs,
                           tools=None, tool_choice=None, reasoning_effort=None)
    parts = env["request"]["contents"][0]["parts"]
    assert len(env["request"]["contents"]) == 1
    assert parts[0]["functionResponse"]["response"] == {"result": "[1, 2]"}
    assert parts[1]["functionResponse"]["response"] == {"a": 1}


def test_parallel_tool_results_merge_into_single_user_turn():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_a1|S1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "call_b2|S2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_a1|S1", "content": "{\"r\": 1}"},
        {"role": "tool", "tool_call_id": "call_b2|S2", "content": "{\"r\": 2}"},
        {"role": "user", "content": "next"},
    ]
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=msgs,
                           tools=None, tool_choice=None, reasoning_effort=None)
    contents = env["request"]["contents"]
    assert [c["role"] for c in contents] == ["user", "model", "user", "user"]
    tool_turn = contents[2]
    assert len(tool_turn["parts"]) == 2
    assert [p["functionResponse"]["id"] for p in tool_turn["parts"]] == ["a1", "b2"]
    assert [p["functionResponse"]["name"] for p in tool_turn["parts"]] == ["f1", "f2"]
    assert [p["functionResponse"]["response"] for p in tool_turn["parts"]] == [{"r": 1}, {"r": 2}]
    assert contents[3]["parts"] == [{"text": "next"}]


def test_models_prefix_is_stripped():
    env = T.build_envelope(project_id="p", model="models/gemini-3-flash",
                           messages=[{"role": "user", "content": "x"}],
                           tools=None, tool_choice=None, reasoning_effort="low")
    assert env["model"] == "gemini-3-flash"
    assert env["request"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low", "includeThoughts": True}


def test_tool_call_id_not_double_prefixed():
    events = [{"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "t", "args": {}, "id": "call_123"}, "thoughtSignature": "SIG"},
    ]}}]}]
    acc = T.accumulate(events)
    assert acc.tool_calls[0]["id"] == "call_123|SIG"
    events2 = [{"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "t", "args": {}, "id": "123"}, "thoughtSignature": "SIG"},
    ]}}]}]
    assert T.accumulate(events2).tool_calls[0]["id"] == "call_123|SIG"


def test_signature_cache_restores_lost_signature():
    T.clear_thought_signature_cache()
    events = [{"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "f", "args": {}, "id": "call_777"}, "thoughtSignature": "S777"},
    ]}}]}]
    T.to_completion(T.accumulate(events), model="gemini-3-flash")
    env = T.build_envelope(
        project_id="p", model="gemini-3-flash",
        messages=[{"role": "assistant", "tool_calls": [
            {"id": "call_777", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]}],
        tools=None, tool_choice=None, reasoning_effort=None,
    )
    assert env["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "S777"
    env2 = T.build_envelope(
        project_id="p", model="gemini-3-flash",
        messages=[{"role": "assistant", "tool_calls": [
            {"id": "call_000", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        ]}],
        tools=None, tool_choice=None, reasoning_effort=None,
    )
    assert env2["request"]["contents"][0]["parts"][0]["thoughtSignature"] == "skip_thought_signature_validator"


def test_signature_cache_is_bounded():
    T.clear_thought_signature_cache()
    for i in range(600):
        T.remember_thought_signature(f"fc{i}", f"sig{i}")
    assert T.lookup_thought_signature("fc0") is None
    assert T.lookup_thought_signature("fc599") == "sig599"
    assert len(T._SIGNATURE_ENTRIES) <= 512


def test_signature_cache_concurrent_writers_stay_bounded():
    T.clear_thought_signature_cache()
    errors: list[Exception] = []

    def writer(tid: int) -> None:
        try:
            for i in range(200):
                T.remember_thought_signature(f"t{tid}-fc{i}", f"sig{tid}-{i}")
                T.lookup_thought_signature(f"call_t{tid}-fc{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    assert len(T._SIGNATURE_ENTRIES) <= T._SIGNATURE_CACHE_MAX
    # 每个 fc_id 最多两把键（raw / call_ 前缀）
    assert len(T._SIGNATURE_CACHE) <= 2 * T._SIGNATURE_CACHE_MAX
    # 仍留在 entries 里的 id 必须可查
    fc_id, sig = next(reversed(T._SIGNATURE_ENTRIES.items()))
    assert T.lookup_thought_signature(f"call_{fc_id}") == sig


_NOTIFY_SCHEMA = {
    "description": "notify desc",
    "anyOf": [
        {"type": "boolean"},
        {"type": "array", "items": {"type": "string"}},
    ],
}


def _tool_with_properties(properties: dict) -> list[dict]:
    return [{"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": properties,
    }}}]


def test_type_array_flattened_for_all_families():
    tools = _tool_with_properties({"p": {"type": ["string", "null"]}})
    for model in ("gemini-3-flash", "claude-sonnet-4-6"):
        env = T.build_envelope(project_id="p", model=model, messages=[{"role": "user", "content": "x"}],
                               tools=tools, tool_choice=None, reasoning_effort=None)
        prop = env["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["p"]
        assert prop == {"type": "string", "nullable": True}


def test_anyof_flattened_for_anthropic_only():
    tools = _tool_with_properties({"notify": _NOTIFY_SCHEMA})
    claude = T.build_envelope(project_id="p", model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
                              tools=tools, tool_choice=None, reasoning_effort=None)
    notify = claude["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["notify"]
    assert "anyOf" not in notify and notify["type"] == "boolean"
    assert notify["description"] == "notify desc（也接受：array<string>）"
    gemini = T.build_envelope(project_id="p", model="gemini-3-flash", messages=[{"role": "user", "content": "x"}],
                              tools=tools, tool_choice=None, reasoning_effort=None)
    assert "anyOf" in gemini["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["notify"]


def test_nested_anyof_inside_items_flattened():
    tools = _tool_with_properties({"arr": {"type": "array", "items": {
        "anyOf": [{"type": "integer"}, {"type": "null"}],
    }}})
    env = T.build_envelope(project_id="p", model="claude-sonnet-4-6", messages=[{"role": "user", "content": "x"}],
                           tools=tools, tool_choice=None, reasoning_effort=None)
    items = env["request"]["tools"][0]["functionDeclarations"][0]["parameters"]["properties"]["arr"]["items"]
    assert items == {"type": "integer", "nullable": True}


_INT64_CONSTRAINT_KEYS = {"minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"}

# Hermes `clarify` 风格：questions 数组带 minItems/maxItems，items.properties.choices 再带 maxItems
_CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array", "description": "questions desc", "minItems": 1, "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "choices": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                    "priority": {"type": "integer", "minimum": 1, "description": "prio"},
                },
            },
        },
    },
}


def _collect_keys(node, found: set) -> set:
    if isinstance(node, dict):
        found.update(node.keys())
        for v in node.values():
            _collect_keys(v, found)
    elif isinstance(node, list):
        for v in node:
            _collect_keys(v, found)
    return found


def _clarify_tool() -> list[dict]:
    return [{"type": "function", "function": {"name": "clarify", "parameters": json.loads(json.dumps(_CLARIFY_SCHEMA))}}]


def test_openai_family_strips_int64_constraints_and_annotates_description():
    tools = _clarify_tool()
    params = T._function_declarations(tools, "openai")[0]["parameters"]
    assert not (_collect_keys(params, set()) & _INT64_CONSTRAINT_KEYS)
    questions = params["properties"]["questions"]
    assert questions["description"] == "questions desc (minItems=1, maxItems=5)"
    choices = questions["items"]["properties"]["choices"]
    assert choices["description"] == "(maxItems=4)"
    assert questions["items"]["properties"]["priority"] == {"type": "integer", "minimum": 1, "description": "prio"}
    # 调用方输入不被改动
    assert tools[0]["function"]["parameters"] == _CLARIFY_SCHEMA


def test_google_and_anthropic_keep_int64_constraints():
    for fam in ("google", "anthropic"):
        params = T._function_declarations(_clarify_tool(), fam)[0]["parameters"]
        questions = params["properties"]["questions"]
        assert questions["minItems"] == 1 and questions["maxItems"] == 5
        assert questions["items"]["properties"]["choices"]["maxItems"] == 4
        assert questions["description"] == "questions desc"


def test_image_data_url_becomes_inline_data():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "see"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
             {"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]
    env = T.build_envelope(project_id="p", model="gemini-3-flash", messages=msgs, tools=None,
                           tool_choice=None, reasoning_effort=None)
    parts = env["request"]["contents"][0]["parts"]
    assert parts == [{"text": "see"}, {"inlineData": {"mimeType": "image/png", "data": "AAAA"}}]
