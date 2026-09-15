"""OpenAI chat ↔ Cloud Code v1internal 的请求/响应映射与 SSE 解析。

请求侧：messages/tools → envelope；响应侧：SSE → 累积 → completion。
依据：opencode-antigravity-auth ANTIGRAVITY_API_SPEC.md、antigravity-proxy（2026 在线验证）。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Optional

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

from agylink.pool import family_of
from agylink.quota import user_agent
from agylink.redact import redact

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful AI assistant."
SKIP_SIGNATURE = "skip_thought_signature_validator"
_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)
_SCHEMA_DROP_KEYS = {"$schema", "additionalProperties", "strict"}
# 仅 openai 族需要剥离的 schema 键：Cloud Code 把我们的 JSON 解析进 Gemini `Schema` proto，
# 这些字段在 proto 里是 int64，按 proto3 JSON 约定会被重新序列化为字符串（"4"）再转发给
# OpenAI 侧校验器，后者要求 integer 而返回 400 INVALID_ARGUMENT。minimum/maximum 是 double，不受影响。
_OPENAI_INT64_CONSTRAINT_KEYS = (
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minProperties",
    "maxProperties",
)

_FINISH_MAP = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "LANGUAGE": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}

_SIGNATURE_CACHE: OrderedDict[str, str] = OrderedDict()
_SIGNATURE_ENTRIES: OrderedDict[str, str] = OrderedDict()
_SIGNATURE_CACHE_MAX = 512
# Hermes gateway 多线程并发请求；两个 OrderedDict 的写入/淘汰必须互斥，
# 否则并发 popitem 可能在空表上抛 KeyError 导致请求失败。
_SIGNATURE_LOCK = threading.Lock()


def _signature_cache_keys(fc_id: str) -> list[str]:
    """为 functionCall id 生成 thoughtSignature 缓存查找键（raw、call_ 前缀、去前缀）。

    参数 fc_id：Gemini functionCall 的 id。
    返回：去重后的键列表。
    """
    keys = [fc_id]
    if fc_id.startswith("call_"):
        keys.append(fc_id[len("call_"):])
    else:
        keys.append(f"call_{fc_id}")
    return list(dict.fromkeys(keys))


def remember_thought_signature(fc_id: str, signature: str) -> None:
    """缓存 functionCall id 与 thoughtSignature，供 Hermes 截断 id 后回放时恢复。

    参数 fc_id：Gemini functionCall 的 id（可含或不含 call_ 前缀）。
    参数 signature：thoughtSignature 字符串。
    返回：无。
    """
    if not fc_id or not signature:
        return
    with _SIGNATURE_LOCK:
        if fc_id in _SIGNATURE_ENTRIES:
            _SIGNATURE_ENTRIES.move_to_end(fc_id)
            _SIGNATURE_ENTRIES[fc_id] = signature
        else:
            _SIGNATURE_ENTRIES[fc_id] = signature
            while len(_SIGNATURE_ENTRIES) > _SIGNATURE_CACHE_MAX:
                evicted_fc_id, evicted_sig = _SIGNATURE_ENTRIES.popitem(last=False)
                for key in _signature_cache_keys(evicted_fc_id):
                    if _SIGNATURE_CACHE.get(key) == evicted_sig:
                        _SIGNATURE_CACHE.pop(key, None)
        for key in _signature_cache_keys(fc_id):
            _SIGNATURE_CACHE[key] = signature
            _SIGNATURE_CACHE.move_to_end(key)


def lookup_thought_signature(tool_call_id: str) -> Optional[str]:
    """按 tool_call id 及其变体查找缓存的 thoughtSignature。

    参数 tool_call_id：OpenAI tool_call.id（可含 |sig，也可已被 Hermes 截断）。
    返回：缓存的 signature，找不到返回 None。
    """
    if not tool_call_id:
        return None
    candidates = [tool_call_id]
    if "|" in tool_call_id:
        candidates.append(tool_call_id.split("|", 1)[0])
    with _SIGNATURE_LOCK:
        for cand in candidates:
            if cand in _SIGNATURE_CACHE:
                return _SIGNATURE_CACHE[cand]
            if cand.startswith("call_"):
                bare = cand[len("call_"):]
                if bare in _SIGNATURE_CACHE:
                    return _SIGNATURE_CACHE[bare]
            else:
                prefixed = f"call_{cand}"
                if prefixed in _SIGNATURE_CACHE:
                    return _SIGNATURE_CACHE[prefixed]
    return None


def clear_thought_signature_cache() -> None:
    """清空 thoughtSignature 缓存（测试用）。

    参数：无。
    返回：无。
    """
    with _SIGNATURE_LOCK:
        _SIGNATURE_CACHE.clear()
        _SIGNATURE_ENTRIES.clear()


def encode_tool_call_id(fc_id: str, signature: str) -> str:
    """把 Gemini functionCall.id 与 thoughtSignature 编进 OpenAI tool_call.id。

    参数 fc_id：Gemini functionCall 的 id（已含 call_ 时不再重复前缀）。
    参数 signature：thoughtSignature 字符串。
    返回：格式为 call_<id>|<sig> 的 OpenAI tool_call.id。
    """
    base = fc_id if fc_id.startswith("call_") else f"call_{fc_id}"
    return f"{base}|{signature}" if signature else base


def decode_tool_call_id(tool_call_id: str) -> tuple[str, str]:
    """拆开 OpenAI tool_call.id 为 Gemini id 与 thoughtSignature。

    参数 tool_call_id：OpenAI 格式的 tool_call.id（支持 call_X|sig 与 X|sig）。
    返回：(fc_id, signature)；无法识别时 ("", "")；无 | 时签名为空。
    """
    if not tool_call_id:
        return "", ""
    if "|" in tool_call_id:
        id_part, sig = tool_call_id.split("|", 1)
        if id_part.startswith("call_"):
            return id_part[len("call_"):], sig
        return id_part, sig
    if tool_call_id.startswith("call_"):
        return tool_call_id[len("call_"):], ""
    return "", ""


def resolve_function_call_id(tool_call_id: str) -> tuple[str, str]:
    """得到 Cloud Code functionCall.id（Claude 必填）与 thoughtSignature。

    decode_tool_call_id 只认我们自己编的 call_<id>|<sig>。Hermes/Grok 历史里常见
    call-<uuid>-N（连字符），按旧逻辑会丢掉 id，Cloud Code 转 Claude 就 400：
    ``tool_use.id: Field required``。认不出我们的编码时，原样保留 tool_call.id。
    参数 tool_call_id：OpenAI / Hermes 的 tool_call.id 或 tool_call_id。
    返回：(functionCall.id, thoughtSignature)；id 在完全为空时生成一个 call_ 前缀值。
    """
    fc_id, signature = decode_tool_call_id(tool_call_id)
    if fc_id:
        return fc_id, signature
    raw = str(tool_call_id or "").split("|", 1)[0].strip()
    if raw:
        return raw, signature
    return f"call_{uuid.uuid4().hex[:24]}", signature


def request_headers(access_token: str) -> dict[str, str]:
    """构造 Cloud Code 推理请求头。

    参数 access_token：OAuth 访问令牌。
    返回：含 Authorization、Content-Type、User-Agent 等键的请求头字典。
    """
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": user_agent(),
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
        "Client-Metadata": json.dumps(
            {"ideType": "ANTIGRAVITY", "platform": "WINDOWS", "pluginType": "GEMINI"},
            separators=(",", ":"),
        ),
    }


def _text_of(content: Any) -> str:
    """把字符串或 parts 数组里的文本拼成一个字符串。

    参数 content：字符串或 OpenAI content parts 列表。
    返回：拼接后的纯文本。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _parts_of(content: Any) -> list[dict]:
    """用户/助手内容 → Gemini parts（文本 + data URL 图片；http 图片丢弃并记 debug）。

    参数 content：字符串或 OpenAI content parts 列表。
    返回：Gemini parts 列表（text / inlineData）。
    """
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict] = []
    for p in content or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text" and p.get("text"):
            parts.append({"text": p["text"]})
        elif p.get("type") == "image_url":
            url = str((p.get("image_url") or {}).get("url") or "")
            if url.startswith("data:") and ";base64," in url:
                mime = url[5:url.index(";")]
                parts.append({"inlineData": {"mimeType": mime, "data": url.split(";base64,", 1)[1]}})
            else:
                logger.debug("agy transport: dropping non-data image url")
    return parts


def _branch_is_null_only(branch: Any) -> bool:
    """判断 JSON Schema 分支是否仅表示 null。

    参数 branch：anyOf/oneOf/allOf 中的一个分支。
    返回：True 表示该分支仅为 null 类型。
    """
    if not isinstance(branch, dict):
        return False
    t = branch.get("type")
    if t == "null":
        return True
    return isinstance(t, list) and set(t) <= {"null"}


def _schema_type_label(branch: dict) -> str:
    """把 schema 分支格式化为简短类型标签（用于 anyOf 备选说明）。

    参数 branch：JSON Schema 对象。
    返回：如 boolean、array<string> 的类型字符串。
    """
    t = branch.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else None
    if t == "array":
        items = branch.get("items")
        if isinstance(items, dict):
            return f"array<{_schema_type_label(items)}>"
        return "array"
    if isinstance(t, str):
        return t
    return "any"


def _normalize_type_array(schema: dict) -> dict:
    """把 type 数组降为单类型，null 成员转为 nullable。

    参数 schema：含 type 数组的 JSON Schema 节点。
    返回：规范化后的节点副本。
    """
    t = schema.get("type")
    if not isinstance(t, list):
        return schema
    out = dict(schema)
    non_null = [x for x in t if x != "null"]
    out["type"] = non_null[0] if non_null else "string"
    if "null" in t:
        out["nullable"] = True
    return out


def _flatten_composition(schema: dict, key: str) -> Optional[dict]:
    """展开 anyOf/oneOf：取首个非 null 分支并合并父级 description。

    参数 schema：含组合关键字的 JSON Schema 节点。
    参数 key：anyOf 或 oneOf。
    返回：展开后的节点；无法展开时 None。
    """
    branches = schema.get(key)
    if not isinstance(branches, list):
        return None
    null_present = any(_branch_is_null_only(b) for b in branches)
    chosen: Optional[dict] = None
    alternatives: list[str] = []
    for branch in branches:
        if not isinstance(branch, dict) or _branch_is_null_only(branch):
            continue
        if chosen is None:
            chosen = dict(branch)
        else:
            alternatives.append(_schema_type_label(branch))
    if chosen is None:
        return None
    if schema.get("description"):
        chosen["description"] = schema["description"]
    if null_present:
        chosen["nullable"] = True
    if alternatives:
        note = f"（也接受：{', '.join(alternatives)}）"
        chosen["description"] = (chosen.get("description") or "") + note
    for k, v in schema.items():
        if k in (key, "anyOf", "oneOf", "allOf", "description") or k in _SCHEMA_DROP_KEYS:
            continue
        if k not in chosen:
            chosen[k] = v
    return chosen


def _strip_int64_constraints(node: dict) -> dict:
    """剥离 openai 族无法通过 Cloud Code 转发的 int64 约束键，并把约束写进 description。

    参数 node：单层 schema 节点（不递归；子节点由 _clean_schema 递归处理）。
    返回：新字典。被剥离的键以 `(minItems=1, maxItems=5)` 形式追加到 description（无则新建，有则空格分隔）。
    """
    present = [k for k in _OPENAI_INT64_CONSTRAINT_KEYS if k in node]
    if not present:
        return node
    result = {k: v for k, v in node.items() if k not in _OPENAI_INT64_CONSTRAINT_KEYS}
    hint = "(" + ", ".join(f"{k}={node[k]}" for k in present) + ")"
    existing = result.get("description")
    result["description"] = f"{existing} {hint}" if existing else hint
    return result


def _clean_schema(schema: Any, family: Optional[str] = None) -> Any:
    """递归清洗 JSON Schema：剔除无效键、单类型化 type 数组、非 google 族展开 anyOf、openai 族剥离 int64 约束。

    参数 schema：JSON Schema 对象、数组或标量。
    参数 family：模型族（google 保留 anyOf；其他族展开；openai 额外剥离 minItems 等 int64 键）。
    返回：Cloud Code 可接受的 schema 副本。
    """
    if isinstance(schema, dict):
        node = {k: v for k, v in schema.items() if k not in _SCHEMA_DROP_KEYS}
        if family != "google":
            flattened: Optional[dict] = None
            for comp_key in ("anyOf", "oneOf"):
                if comp_key in node:
                    flattened = _flatten_composition(node, comp_key)
                    break
            if flattened is None and isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
                only = node["allOf"][0]
                if isinstance(only, dict):
                    flattened = _flatten_composition({**node, "anyOf": [only]}, "anyOf")
            if flattened is not None:
                node = flattened
        node = _normalize_type_array(node)
        if family == "openai":
            node = _strip_int64_constraints(node)
        result: dict[str, Any] = {}
        for k, v in node.items():
            if k == "properties" and isinstance(v, dict):
                result[k] = {pk: _clean_schema(pv, family) for pk, pv in v.items()}
            elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
                result[k] = [_clean_schema(b, family) for b in v]
            elif isinstance(v, dict):
                result[k] = _clean_schema(v, family)
            elif isinstance(v, list):
                result[k] = [_clean_schema(i, family) for i in v]
            else:
                result[k] = v
        return result
    if isinstance(schema, list):
        return [_clean_schema(v, family) for v in schema]
    return schema


def _function_declarations(tools: Optional[list[dict]], family: Optional[str]) -> list[dict]:
    """OpenAI tools 列表 → Gemini functionDeclarations。

    参数 tools：OpenAI 格式的 tools 列表，可为 None。
    参数 family：模型族，用于 schema 清洗策略。
    返回：按 name 排序后的 Gemini functionDeclaration 字典列表。
    排序原因：Gemini 隐式缓存要求 tools 前缀字节稳定；Hermes 的 tool_search
    可能打乱可见工具顺序，按 name 排序后同一集合会得到同一前缀。
    """
    decls = []
    for t in tools or []:
        fn = (t or {}).get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        d = {"name": fn["name"]}
        if fn.get("description"):
            d["description"] = fn["description"]
        params = fn.get("parameters")
        d["parameters"] = (
            _clean_schema(params, family) if isinstance(params, dict) else {"type": "object", "properties": {}}
        )
        decls.append(d)
    decls.sort(key=lambda item: str(item.get("name") or ""))
    return decls


def _tool_config(tool_choice: Any) -> Optional[dict]:
    """OpenAI tool_choice → Gemini toolConfig。

    参数 tool_choice：auto/none/required 或指定 function 的 dict。
    返回：Gemini toolConfig 字典，无法识别时默认 AUTO。
    """
    if tool_choice is None or tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        name = ((tool_choice.get("function") or {}).get("name"))
        cfg = {"mode": "ANY"}
        if name:
            cfg["allowedFunctionNames"] = [name]
        return {"functionCallingConfig": cfg}
    return {"functionCallingConfig": {"mode": "AUTO"}}


def _resolve_thought_signature(tool_call_id: str, decoded_sig: str) -> str:
    """解析 assistant tool_call 应回传的 thoughtSignature。

    参数 tool_call_id：原始 OpenAI tool_call.id。
    参数 decoded_sig：decode_tool_call_id 拆出的签名（可为空）。
    返回：thoughtSignature 字符串（含占位符）。
    """
    if decoded_sig:
        return decoded_sig
    cached = lookup_thought_signature(tool_call_id)
    if cached:
        return cached
    return SKIP_SIGNATURE


def _contents_of(messages: list[dict]) -> tuple[list[dict], str]:
    """OpenAI messages → Gemini contents 与 system 文本。

    工具结果轮次使用 role user（Cloud Code 对 role function 返回 400）；
    连续多条 tool 消息（并行工具调用的结果）合并进同一个 user content 的 parts，
    以维持 Gemini 要求的 model/user 交替。
    参数 messages：OpenAI chat messages 列表。
    返回：(contents, system_text) 元组。
    """
    system_chunks: list[str] = []
    contents: list[dict] = []
    name_by_call_id: dict[str, str] = {}
    # 指向正在累积 functionResponse 的 user content；遇到非 tool 消息即断开
    open_tool_turn: Optional[dict] = None
    for msg in messages or []:
        role = msg.get("role")
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                system_chunks.append(text)
            continue
        if role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            fc_id, _ = resolve_function_call_id(call_id)
            text = _text_of(msg.get("content"))
            try:
                parsed = json.loads(text) if text else {}
                response = parsed if isinstance(parsed, dict) else {"result": text}
            except ValueError:
                response = {"result": text}
            fr = {"name": msg.get("name") or name_by_call_id.get(call_id) or "tool", "response": response}
            if fc_id:
                fr["id"] = fc_id
            if open_tool_turn is None:
                open_tool_turn = {"role": "user", "parts": []}
                contents.append(open_tool_turn)
            open_tool_turn["parts"].append({"functionResponse": fr})
            continue
        open_tool_turn = None
        if role == "assistant":
            parts = _parts_of(msg.get("content"))
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"_raw": fn.get("arguments")}
                tool_id = str(tc.get("id") or "")
                fc_id, sig = resolve_function_call_id(tool_id)
                name_by_call_id[tool_id] = fn.get("name") or ""
                fc = {"name": fn.get("name"), "args": args if isinstance(args, dict) else {"_raw": args}}
                if fc_id:
                    fc["id"] = fc_id
                parts.append({"functionCall": fc, "thoughtSignature": _resolve_thought_signature(tool_id, sig)})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        parts = _parts_of(msg.get("content"))
        if parts:
            contents.append({"role": "user", "parts": parts})
    return contents, "\n\n".join(system_chunks)


def build_envelope(
    *,
    project_id: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    tool_choice: Any,
    reasoning_effort: Optional[str],
    temperature: Any = None,
    top_p: Any = None,
    max_tokens: Any = None,
    stop: Any = None,
    request_id: Optional[str] = None,
) -> dict:
    """把一次 OpenAI chat 请求译成 Cloud Code envelope（spec 第 7 节）。

    参数 project_id：GCP 项目 id。
    参数 model：模型 id。
    参数 messages：OpenAI chat messages。
    参数 tools：可选 OpenAI tools 列表。
    参数 tool_choice：工具调用策略。
    参数 reasoning_effort：推理强度（low/medium/high）。
    参数 temperature、top_p、max_tokens、stop：生成参数。
    参数 request_id：可选请求 id，缺省 agent-<hex>。
    返回：Cloud Code envelope 字典。
    """
    if model.startswith("models/"):
        model = model[len("models/"):]
    fam = family_of(model)
    contents, system_text = _contents_of(messages)
    request: dict[str, Any] = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": system_text or DEFAULT_SYSTEM_INSTRUCTION}]},
        "safetySettings": [{"category": c, "threshold": "BLOCK_NONE"} for c in _SAFETY_CATEGORIES],
    }
    decls = _function_declarations(tools, fam)
    if decls:
        request["tools"] = [{"functionDeclarations": decls}]
    cfg = _tool_config(tool_choice)
    if cfg and (decls or tool_choice == "none"):
        request["toolConfig"] = cfg
    gen: dict[str, Any] = {}
    if isinstance(temperature, (int, float)):
        gen["temperature"] = float(temperature)
    if isinstance(top_p, (int, float)):
        gen["topP"] = float(top_p)
    if isinstance(max_tokens, int) and max_tokens > 0:
        gen["maxOutputTokens"] = max_tokens
    if stop:
        gen["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    if fam == "google" and reasoning_effort in ("low", "medium", "high"):
        gen["thinkingConfig"] = {"thinkingLevel": reasoning_effort, "includeThoughts": True}
    elif model == "claude-opus-4-6-thinking":
        gen["thinkingConfig"] = {"includeThoughts": True}
    if gen:
        request["generationConfig"] = gen
    return {
        "project": project_id,
        "model": model,
        "requestType": "agent",
        "userAgent": "antigravity",
        "requestId": request_id or f"agent-{uuid.uuid4().hex}",
        "request": request,
    }


class UpstreamError(Exception):
    """Cloud Code 非 2xx 响应。"""

    def __init__(self, status: int, body: str):
        """构造上游 HTTP 错误。

        参数 status：HTTP 状态码。
        参数 body：响应体字符串（原始值存于 self.body；异常消息为脱敏并截断至 400 字符）。
        返回：无。
        """
        super().__init__(f"upstream HTTP {status}: {redact(str(body))[:400]}")
        self.status = status
        self.body = body or ""

    def is_quota(self) -> bool:
        """是否为限流/额度耗尽（429 或 RESOURCE_EXHAUSTED）。

        参数：无。
        返回：True 表示配额/限流错误。
        """
        return self.status == 429 or "RESOURCE_EXHAUSTED" in self.body

    def is_auth(self) -> bool:
        """是否为凭证失效（401）。

        参数：无。
        返回：True 表示认证错误。
        """
        return self.status == 401

    def reset_time(self) -> Optional[str]:
        """尽力从错误 body 里找 resetTime（ISO 字符串）。

        参数：无。
        返回：resetTime 字符串，找不到返回 None。
        """
        try:
            payload = json.loads(self.body)
        except ValueError:
            return None
        stack = [payload]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if isinstance(cur.get("resetTime"), str):
                    return cur["resetTime"]
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return None


def iter_sse_events(lines: Iterable[str]) -> Iterator[dict]:
    """逐行解析 SSE：取 data: 行的 JSON，解包 {"response": ...}，忽略注释/空行/非 JSON。

    参数 lines：SSE 文本行迭代器。
    返回：解包后的 Gemini 响应 dict 迭代器。
    """
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("response"), dict):
            obj = obj["response"]
        if isinstance(obj, dict):
            yield obj


@dataclass
class Accumulated:
    """一次流式响应累积后的结果。"""

    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
        }
    )


def _nonneg_int(value: Any) -> int:
    """把任意值尽量转成非负整数。

    参数 value：usageMetadata 里的计数字段（可能是 int、str 或无效值）。
    返回：成功时的非负整数；无法转换或为负数时返回 0。
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def cached_tokens_from_usage_metadata(usage_metadata: Optional[dict]) -> int:
    """从 Cloud Code / Gemini usageMetadata 提取隐式缓存命中 token 数。

    Gemini 官方字段是 cachedContentTokenCount；部分 Cloud Code 转发还会给
    cacheReadInputTokenCount（Claude 风格）或 cacheTokensDetails 明细。
    参数 usage_metadata：SSE 事件里的 usageMetadata 字典，允许 None。
    返回：缓存命中 token 数；没有任何可识别字段时返回 0。
    """
    if not isinstance(usage_metadata, dict):
        return 0
    for key in (
        "cachedContentTokenCount",
        "cached_content_token_count",
        "cacheReadInputTokenCount",
        "cache_read_input_tokens",
        "cachedTokens",
        "cached_tokens",
    ):
        count = _nonneg_int(usage_metadata.get(key))
        if count:
            return count
    details = usage_metadata.get("cacheTokensDetails") or usage_metadata.get("cache_tokens_details")
    if not isinstance(details, list):
        return 0
    total = 0
    for item in details:
        if isinstance(item, dict):
            total += _nonneg_int(item.get("tokenCount") or item.get("token_count"))
    return total


def accumulate(events: Iterable[dict]) -> Accumulated:
    """把解包后的 Gemini 响应事件累积成文本、thinking、tool_calls、finish_reason、usage。

    参数 events：iter_sse_events 产出的响应 dict 迭代器。
    返回：Accumulated 累积结果。
    """
    acc = Accumulated()
    raw_finish: Optional[str] = None
    for ev in events:
        cands = ev.get("candidates") or []
        for cand in cands:
            for part in ((cand.get("content") or {}).get("parts") or []):
                if not isinstance(part, dict):
                    continue
                if part.get("functionCall"):
                    fc = part["functionCall"]
                    fc_id = str(fc.get("id") or "")
                    sig = str(part.get("thoughtSignature") or part.get("thought_signature") or "")
                    if sig and fc_id:
                        remember_thought_signature(fc_id, sig)
                    tid = encode_tool_call_id(fc_id, sig) if (fc_id or sig) else f"call_{uuid.uuid4().hex[:24]}"
                    args = fc.get("args", {})
                    acc.tool_calls.append(
                        {
                            "id": tid,
                            "name": str(fc.get("name") or ""),
                            "arguments": json.dumps(args) if isinstance(args, (dict, list)) else str(args),
                        }
                    )
                elif part.get("text"):
                    if part.get("thought"):
                        acc.reasoning += part["text"]
                    else:
                        acc.text += part["text"]
            if cand.get("finishReason"):
                raw_finish = str(cand["finishReason"]).upper()
        um = ev.get("usageMetadata")
        if isinstance(um, dict):
            # 后续 SSE 分片可能只带部分 usage、漏掉缓存字段；已读到的 cached_tokens 要保留。
            cached_tokens = cached_tokens_from_usage_metadata(um) or int(acc.usage.get("cached_tokens") or 0)
            acc.usage = {
                "prompt_tokens": int(um.get("promptTokenCount") or 0),
                "completion_tokens": int(um.get("candidatesTokenCount") or um.get("completionTokenCount") or 0),
                "total_tokens": int(um.get("totalTokenCount") or 0),
                "reasoning_tokens": int(um.get("thoughtsTokenCount") or 0),
                "cached_tokens": cached_tokens,
            }
    if acc.tool_calls:
        acc.finish_reason = "tool_calls"
    else:
        acc.finish_reason = _FINISH_MAP.get(raw_finish or "", "stop")
    if not acc.usage["total_tokens"]:
        acc.usage["total_tokens"] = acc.usage["prompt_tokens"] + acc.usage["completion_tokens"]
    return acc


def to_completion(acc: Accumulated, *, model: str) -> SimpleNamespace:
    """组装成 Hermes 期望的 OpenAI 形态 completion（与 copilot_acp_client 一致）。

    参数 acc：accumulate 产出的累积结果。
    参数 model：模型 id。
    返回：SimpleNamespace 形态的 chat completion。
    """
    for tc in acc.tool_calls:
        tid = str(tc.get("id") or "")
        if "|" in tid:
            id_part, sig = tid.split("|", 1)
            if sig:
                fc_key = id_part[len("call_"):] if id_part.startswith("call_") else id_part
                remember_thought_signature(fc_key, sig)
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=tc["id"],
            type="function",
            function=Function(name=tc["name"], arguments=tc["arguments"]),
        )
        for tc in acc.tool_calls
    ] or None
    usage = SimpleNamespace(
        prompt_tokens=acc.usage["prompt_tokens"],
        completion_tokens=acc.usage["completion_tokens"],
        total_tokens=acc.usage["total_tokens"],
        prompt_tokens_details=SimpleNamespace(cached_tokens=int(acc.usage.get("cached_tokens") or 0)),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=acc.usage.get("reasoning_tokens", 0)),
    )
    message = SimpleNamespace(
        role="assistant",
        content=acc.text or None,
        tool_calls=tool_calls,
        reasoning=acc.reasoning or None,
        reasoning_content=acc.reasoning or None,
        reasoning_details=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason=acc.finish_reason or "stop")
    return SimpleNamespace(choices=[choice], usage=usage, model=model)
