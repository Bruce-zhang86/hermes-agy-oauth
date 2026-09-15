# hermes-agy-oauth

Hermes 插件：用 Google Antigravity（AGY）OAuth 订阅，把 Gemini / Claude / GPT-OSS 当作 Hermes 的模型后端。工具循环仍由 Hermes 自己执行。

仓库里是**两个插件**，需要都装：

| 目录 | `kind` | 作用 |
| --- | --- | --- |
| `agy-link/` | standalone | `hermes agy` CLI、`/agy`、号池、OAuth |
| `agy-oauth/` | model-provider | `--provider agy-oauth`、`/model`、推理客户端 |

共享逻辑在 `agy-link/agylink/`。本仓库**不含**账号、token、号池文件。

## 安装

需要已安装 [Hermes](https://github.com/NousResearch/hermes-agent)，且本机 `git` 可用。

```powershell
hermes plugins install Bruce-zhang86/hermes-agy-oauth/agy-link
hermes plugins install Bruce-zhang86/hermes-agy-oauth/agy-oauth
```

在 `config.yaml` 的 `plugins.enabled` 里加上 `agy-link`（`agy-oauth` 由模型提供方发现，不必写进 enabled）：

```yaml
plugins:
  enabled:
    - agy-link
```

浏览器登录前，在 Hermes 环境里设置与官方 `agy` CLI **相同**的公开消费者客户端（不要把这两项提交进 git）：

```powershell
$env:HERMES_AGY_CLIENT_ID = "<官方 agy 的 client id>"
$env:HERMES_AGY_CLIENT_SECRET = "<官方 agy 的 client secret>"
```

也可写进 `$HERMES_HOME/.env`。本机已在用的 Hermes 插件里（`agylink/oauth.py`）能看到这两项。

然后**重启网关**（飞书 / Telegram 等长驻进程要重启后才会认出 `agy-oauth`）：

```powershell
hermes gateway restart
```

手工拷贝也可以：

- `agy-link/` → `$HERMES_HOME/plugins/agy-link/`
- `agy-oauth/` → `$HERMES_HOME/plugins/agy-oauth/`  
  （或 `$HERMES_HOME/plugins/model-providers/agy-oauth/`，与 Hermes 自带提供方同一层）

## 使用

```powershell
# 登录（浏览器 Google 授权）
hermes agy auth --alias 工作号

# 从 DSH 只读导入已有 AGY 号（不写回 DSH）
hermes agy import-dsh
hermes agy status
hermes agy models

# 对话
hermes -m gemini-3-flash --provider agy-oauth
```

飞书等会话里：

```
/model --provider agy-oauth gemini-3-flash
```

模型 ID 必须用 `hermes agy models` 打出来的 **Cloud Code 名称**（例如 `gemini-3.8-flash-tiered`），不要用 OpenRouter 短名（`gemini-3.8-flash` 会 404）。

## 测试

需要本机有 Hermes 源码（提供 `providers` 包）：

```powershell
$env:HERMES_AGENT = "D:\hermes\hermes-agent"
& $env:HERMES_AGENT\venv\Scripts\python.exe -m pytest D:\hermes-agy-oauth\agy-link\tests -q
```

## 安全

- Token 落在 `$HERMES_HOME/agy-accounts/`，不要提交。
- `/agy remove` 在网关斜杠命令里被禁用；删号请用终端 `hermes agy remove <id>`。
- DSH 来源账号只取消登记，不会删除 DSH 目录。
