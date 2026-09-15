# D:\hermes\plugins\agy-link\agylink\redact.py
"""日志涂抹：去掉 access token、refresh token、授权码与带 code= 的回调 URL。"""
from __future__ import annotations

import re

_PATTERNS = (
    (re.compile(r"code=[^&\s\"']+"), "code=<redacted>"),
    (re.compile(r"ya29\.[A-Za-z0-9._\-]+"), "<access-token>"),
    (re.compile(r"1//[A-Za-z0-9._\-]+"), "<refresh-token>"),
    # Google 授权码形如 4/0AbCd...（远长于 20 字符）；要求 ≥20 字符，避免误伤命令输出里的 4/5、4/10 之类普通分数。
    (re.compile(r"(?<![A-Za-z0-9])4/[A-Za-z0-9._\-]{20,}"), "<auth-code>"),
)


def redact(text: str) -> str:
    """返回涂抹后的文本。

    参数 text：任意字符串（None 视为空串）。
    返回：ya29.* → <access-token>，1//* → <refresh-token>，4/<≥20 字符> → <auth-code>，
    code=xxx → code=<redacted>。顺序先处理 code=，避免授权码规则先吃掉 URL。
    """
    out = text or ""
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out
