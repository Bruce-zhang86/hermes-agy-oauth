# D:\hermes\plugins\agy-link\tests\test_redact.py
"""redact 模块测试：四类秘密都必须被涂抹。"""
from agylink.redact import redact


def test_redacts_access_token():
    assert "ya29." not in redact("Bearer ya29.a0AbCdEf-ghi_jkl")
    assert "<access-token>" in redact("Bearer ya29.a0AbCdEf-ghi_jkl")


def test_redacts_refresh_token():
    assert redact("rt=1//06abcDEF_ghi-jkl") == "rt=<refresh-token>"


def test_redacts_auth_code_and_callback_url():
    out = redact("http://localhost:51121/oauth-callback?code=4/0AbcDEF&state=xyz")
    assert "4/0AbcDEF" not in out
    assert "code=<redacted>" in out


def test_leaves_plain_text_alone():
    assert redact("hello world 429") == "hello world 429"


def test_redacts_bare_auth_code():
    code = "4/0AbCdEfGhIjKlMnOpQrStUvWxYz-_.123"
    assert len(code) - 2 >= 20
    assert redact(f"code: {code}") == "code: <auth-code>"


def test_short_fractions_are_not_treated_as_auth_code():
    assert redact("已导入 4/10 个账号") == "已导入 4/10 个账号"
    assert redact("progress 4/5 done") == "progress 4/5 done"
