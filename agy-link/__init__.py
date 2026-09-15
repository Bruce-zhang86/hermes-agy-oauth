"""agy-link 插件入口：注册 ``hermes agy`` CLI 子命令与 ``/agy`` 斜杠命令。

所有业务逻辑位于 agylink 包；本模块只负责 Python 路径接线和 Hermes 命令注册。
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agylink.commands import dispatch  # noqa: E402


def _setup_argparse(subparser) -> None:
    """配置 ``hermes agy <sub> [args...]`` 的 argparse 参数。

    参数 subparser：Hermes 为 agy CLI 命令创建的 argparse 子解析器。
    返回：无；解析结果通过 ``agy_args`` 交给共享 dispatch。
    """
    subparser.add_argument(
        "agy_args",
        nargs=argparse.REMAINDER,
        help="子命令与参数，如：auth --alias 工作号 / status / import-dsh",
    )
    subparser.set_defaults(func=_cli_handler)


def _cli_handler(args) -> None:
    """执行终端 CLI 命令并打印共享 dispatch 的纯文本结果。

    参数 args：argparse 解析得到、包含 ``agy_args`` 的命名空间。
    返回：无；命令结果写到标准输出。
    """
    print(dispatch(list(getattr(args, "agy_args", []) or []), open_browser=True))


def _slash_handler(raw_args: str) -> str:
    """执行 ``/agy`` 斜杠命令，并禁用服务端浏览器操作。

    参数 raw_args：斜杠命令名称后的原始参数文本。
    返回：共享 dispatch 生成的纯文本结果。
    """
    try:
        raw_parts = shlex.split(raw_args or "", posix=False)
        parts = [
            part[1:-1]
            if len(part) >= 2 and part[0] == part[-1] and part[0] in ("'", '"')
            else part
            for part in raw_parts
        ]
    except ValueError as exc:
        return f"参数解析失败：{exc}"
    if parts and parts[0] == "remove":
        return "出于安全考虑，/agy 不支持 remove；请在终端执行 hermes agy remove <id>"
    return dispatch(parts, open_browser=False)


def register(ctx) -> None:
    """向 Hermes 注册 agy CLI 和斜杠命令。

    参数 ctx：Hermes 提供的插件注册上下文。
    返回：无；注册信息写入插件管理器。
    """
    ctx.register_cli_command(
        name="agy",
        help="Google Antigravity 登录、号池与模型（hermes agy help）",
        setup_fn=_setup_argparse,
        handler_fn=_cli_handler,
        description="管理 Antigravity 登录、账号池、配额与模型目录",
    )
    ctx.register_command(
        "agy",
        handler=_slash_handler,
        description="Antigravity 账号：auth / status / import-dsh / models / help",
        args_hint="<auth|auth-code|status|pool|import-dsh|models|refresh-quota|help>",
    )
