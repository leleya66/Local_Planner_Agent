# LocalMate Agent 提交说明

## 1. 项目简介

LocalMate Agent 是一个面向上海本地周末出行的 AI 路线规划 Demo。用户可以输入出发地、目的地、预算、人数、时间、偏好和限制条件，系统会结合本地地点表、RAG 案例库、高德地图能力和 LLM 生成可视化路线方案。

项目支持两种模式：

- 单人模式：一个用户直接输入需求并生成路线。
- 多人协作模式：多人进入同一个房间聊天，最后通过 `@Agent 开始规划` 汇总群聊需求生成方案。

当前代码已经做了阶段性模块化：

- `core/`：配置和请求模型。
- `services/`：会话、房间、地图代理、预约/团购辅助逻辑。
- `routers/`：多人房间路由。
- `workflow/`：Agent workflow 的模块化入口和部分已迁移工具函数。
- `agent_workflow_improved.py`：保留为 legacy 兼容实现，保证演示稳定。

## 2. 演示视频链接



## 3. 环境要求

推荐环境：

- Windows 10/11
- PowerShell
- Conda 或 Miniconda
- 网络可访问 DashScope 和高德地图服务
- 多人协作模式需要安装并配置 `ngrok`

请先将提交包解压到任意本地目录，例如：

```text
C:\Users\YourName\Desktop\LocalMate_Agent
```


本项目已提供 `environment.yml`，可用于创建项目环境。请在解压后的项目目录中打开 PowerShell，然后执行一次：

```powershell
conda env create -f environment.yml
```

该命令会创建名为 `localmate_agent` 的 Conda 环境。后续启动脚本会自动查找这个环境。

如果环境已经创建过，可以跳过这一步。若需要手动检查环境是否存在，可以执行：

```powershell
conda env list
```

如果评委希望使用已有环境，也可以先激活自己的环境再运行启动脚本；脚本会优先使用当前已激活环境中的 Python。

## 4. 配置 .env

请在解压后的项目目录中复制一份：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入真实 Key：

```text
DASHSCOPE_API_KEY=你的DashScopeKey
AMAP_API_KEY=你的高德Web服务Key
```

没有 `DASHSCOPE_API_KEY` 时，后端会无法初始化 LLM/RAG 工作流。没有 `AMAP_API_KEY` 时，地图、距离和 POI 能力会受影响。

## 5. 单人模式启动

请在解压后的项目目录中打开 PowerShell，执行：

```powershell
.\start_demo.ps1 -Mode single
```

脚本会自动定位到自身所在目录，并自动查找 `localmate_agent` Conda 环境或当前已激活的 Python 环境，不需要手动修改代码路径。

默认访问地址：

```text
http://127.0.0.1:8041/v8
```

如果端口被占用，可以执行：

```powershell
.\start_demo.ps1 -Mode single -Restart
```

如果 PowerShell 提示脚本执行策略限制，可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_demo.ps1 -Mode single
```

## 6. 多人协作模式启动

多人模式需要先安装并配置 `ngrok`。

请在解压后的项目目录中打开 PowerShell，执行：

```powershell
.\start_demo.ps1 -Mode collab
```

脚本会启动本地服务并打印公网地址，例如：

```text
Public URL: https://xxxx.ngrok-free.dev/
```

打开该公网地址后，点击页面右上角的协作入口，输入昵称并生成邀请链接。邀请链接会包含 `?room=...`，其他用户打开该链接即可加入同一房间。

如果需要重启多人模式：

```powershell
.\start_demo.ps1 -Mode collab -Restart
```

如果 PowerShell 提示脚本执行策略限制，可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\start_demo.ps1 -Mode collab
```

注意：ngrok 地址每次重启可能变化，PowerShell 窗口关闭后公网链接会失效。

## 7. 常见问题

**问题：提示找不到 `DASHSCOPE_API_KEY`。**

请确认已经从 `.env.example` 复制出 `.env`，并填入真实 `DASHSCOPE_API_KEY`。

**问题：地图、距离或 POI 不准确。**

请确认 `.env` 中配置了 `AMAP_API_KEY`，并且该 Key 是高德 Web 服务 Key。

**问题：端口被占用。**

使用：

```powershell
.\start_demo.ps1 -Mode single -Restart
```

或：

```powershell
.\start_demo.ps1 -Mode collab -Restart
```

**问题：多人协作无法生成公网链接。**

请确认已安装 `ngrok`，并且执行过：

```powershell
ngrok config add-authtoken 你的ngrokToken
```

**问题：无法直接打开本地地址。**

`127.0.0.1` 只能在运行服务的电脑上访问。线上提交建议提供演示视频链接；如果需要在线体验，请部署到云服务器，或在演示时使用 ngrok 公网链接。


