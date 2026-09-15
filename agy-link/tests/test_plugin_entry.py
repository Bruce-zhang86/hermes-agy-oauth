"""agy-link 插件入口测试：注册、CLI 参数透传与斜杠命令安全边界。"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

PLUGIN_ENTRY = Path(__file__).resolve().parent.parent / "__init__.py"


class FakeContext:
    """记录插件注册调用的最小 Hermes 上下文替身。"""

    def __init__(self):
        self.cli_calls = []
        self.slash_calls = []

    def register_cli_command(self, **kwargs):
        self.cli_calls.append(kwargs)

    def register_command(self, name, **kwargs):
        self.slash_calls.append((name, kwargs))


def _load_entry():
    spec = importlib.util.spec_from_file_location("agy_link_plugin_entry_test", PLUGIN_ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registers_cli_and_slash_commands():
    entry = _load_entry()
    context = FakeContext()

    entry.register(context)

    assert context.cli_calls[0]["name"] == "agy"
    assert context.cli_calls[0]["description"]
    assert context.slash_calls[0][0] == "agy"


def test_cli_parser_passes_options_through():
    entry = _load_entry()
    context = FakeContext()
    entry.register(context)
    parser = argparse.ArgumentParser()

    context.cli_calls[0]["setup_fn"](parser)

    parsed = parser.parse_args(["auth", "--alias", "X"])
    assert parsed.agy_args == ["auth", "--alias", "X"]


def test_slash_handler_returns_parse_error_for_unbalanced_quote():
    entry = _load_entry()

    result = entry._slash_handler('auth --alias "unbalanced')

    assert "参数解析失败" in result


def test_slash_handler_refuses_remove():
    entry = _load_entry()

    result = entry._slash_handler("remove acc_1")

    assert result == "出于安全考虑，/agy 不支持 remove；请在终端执行 hermes agy remove <id>"


def test_slash_handler_returns_help():
    entry = _load_entry()

    result = entry._slash_handler("help")

    assert "hermes agy" in result
