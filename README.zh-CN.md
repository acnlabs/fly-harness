# fly-harness

[English](README.md) | [中文](README.zh-CN.md)

**v0.5.5** — 稀疏神经 **harness（对接层）**，不是果蝇脑仿真。

![3D NeuroMechFly 走路（FlyGym 身体；仅 docs/usage 展示）](https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/overview_video.gif)

这段 3D 走路是 **docs/usage 展示** — 不是内核功能。片段来自
[FlyGym / NeuroMechFly](https://github.com/NeLy-EPFL/flygym)。默认回路里的脑是
**24 神经元 LIF fixture**。腿仍由 FlyGym 的运动 **CPG** 驱动。`FlyHarness.step`
在该 CPG **之上**。不是 166,700 神经元大脑，不是意识/上传，不是 Google 托管，
也不比 FlyGym 自己的控制器走得更好。

公式：**Agent = Model + harness**。harness 通过稳定契约（**仅** `ModelBackend`）
对接一台**正在运行**的仿真。FlyWire / MaleCNS 文件是权重转储，不是该契约。
harness 不对接到某个特定模型。

公开契约是：

1. **`BrainState`** — 膜电位、稀疏突触权重（`scipy.sparse` CSR）、时间戳、JSON 保存/加载
2. **`FlyHarness.step(obs) -> action`** — encode → **`ModelBackend.tick`** → decode（默认 backend 是进程内 LIF/rate 回路）
3. **`Encoder` / `Decoder`** — `typing.Protocol` 契约（可选 `BaseEncoder` / `BaseDecoder` ABC；不做字符串类名检查）
4. **`load_connectome`** — 从本地稀疏边列表（`.npz` / `.csv`）加载**回路子集**到 `BrainState`
5. **`ModelBackend`** — 可替换的 Model 端口：`tick`（编码输入 → 神经输出）、`reset`、`n_neurons`、`model_id`。可选 `snapshot` / `restore`（与 `reset` 同类；不支持的 backend 必须抛错，绝不能静默空操作）。见 [docs/docking.md](docs/docking.md)。

24 神经元触觉反射回路是 **测试/演示 fixture**。它不是模型，也不是缩小版 FlyWire 大脑。

## 这不是什么

- **不是** 14 万神经元 FlyWire 连接组、上传、或稠密 140k×140k 矩阵
- **不是** **内核**里任意规模的全脑运行时（内存受已加载子集约束）。第三方 Model 如 [`flybrain`](https://pypi.org/project/flybrain/)（MaleCNS v1.0，**166,700 神经元**，不是 16 万「参数」）通过 `ModelBackend` 接入 — 这是在**使用** harness，不是 harness 功能。你在本地跑的转储，不是 Google 托管仿真，也不是 FlyWire 网站
- **不是** `caveclient` 或内置生物力学引擎 — FlyGym/NeuroMechFly 是**可选身体 extra**，不是核心产品。extra ≠ 架构
- **不是** MCP 微内核或 FastAPI 服务 — MCP 是**可选协议 extra**，不是核心产品。extra ≠ 架构
- **不是** OpenRouter 这家公司、市场、计费系统、托管云网关、或公开生物仿真目录 — `BioSimRouter` 是 harness 侧 **port**；**biorouter** 是可选的**本地** HTTP 进程（stdlib，默认 `127.0.0.1`）
- **不是** 托管的 OpenWorm / c302 API，不是 `fly-harness[c302]` / `fly-harness[openworm]`，也**不是**一个新的全脑 PyPI 包。列出 `c302.celegans` 是用法（部署一台仿真 + 列出一个 id）。默认 CI 使用 `FakeC302`（**302** 条雌雄同体神经元），不是 OpenWorm Docker
- **不是** 意识/上传宣称

## 安装

Python 3.10+，依赖 `numpy` 和 `scipy`。核心安装仍只有 numpy/scipy。
刚装完（PyPI）— 然后证明内核还活着：

```bash
pip install fly-harness
fly-harness-check
```

从克隆（跑测试）：

```bash
pip install -e ".[dev]"
```

FlyGym 是可选 extra（可选扩展，不是架构）：

```bash
pip install -e ".[flygym]"
```

MCP 是可选 extra（经 FastMCP 的 stdio 服务器）：

```bash
pip install -e ".[mcp]"
```

**biorouter** 是可选 extra，且**只有 stdlib**（无 FastAPI）。这个 extra 存在，是为了让它仍是本仓库的扩展，而不是第二个产品或 GitHub 仓库：

```bash
pip install -e ".[biorouter]"
```

**flybrain** 是可选 extra。该 extra 是适配器，好让 **biorouter** 能列出 `flybrain.malecns`，示例能用那个第三方 Model。harness 不对接到某个特定模型。核心保持与模型无关，不 import flybrain。默认测试**不会**下载 `~/fly-data`：

```bash
pip install -e ".[flybrain]"
```

## 用法导览

安装后，先证明内核还活着，再对接你已经在跑的 Model。
公开调用永远是 `FlyHarness.step(obs) -> action`。这不是聊天
shell、web UI 或插件市场。不是意识或上传宣称。
更长的 how-to：[docs/usage.md](docs/usage.md)。契约：
[docs/docking.md](docs/docking.md)。

### 1. Fixture 反射 — 刚装完，无需 extra

24 神经元触觉反射是 **测试 fixture**。它证明 encode → tick →
decode。它不是生物回路。

```bash
pip install fly-harness
fly-harness-check
# or: python -m fly_harness
```

打印 `model_id`、`n_neurons`、`timestamp`，以及进程内 LIF
（`fly-harness.in-process-lif`）上的一次 obs→action 步进。CI 跑同样的检查，
不装 extra。

### 2. 直接 `ModelBackend` — 你已经在跑一台仿真

harness 核心**不**绑定 flybrain、c302 或任何厂商 Model。
如果你已经有一台可 tick 的仿真，包装 `ModelBackend`（`tick` / `reset` /
`n_neurons` / `model_id`）。harness 只做 encode → tick → decode。
示例和适配器（`examples/flybrain_loop.py`、`fly_harness.flybrain`）
是 **usage，不是内核**。

```python
from fly_harness import FlyHarness

# sim is your already-running flybrain / c302 / other tickable Model
backend = MySimBackend(sim)  # duck-typed: tick, reset, n_neurons, model_id
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
result = harness.step(observation)
```

草稿：[docs/usage.md](docs/usage.md)。可运行的 Direct 示例：
`examples/flybrain_loop.py`（CI 里用 FakeFlyBrain；`--real` 仅当 MaleCNS
文件已在磁盘上 — 永不下载）。

### 3. biorouter model id — 同一回路，选一个 id

本地 stdlib HTTP（`127.0.0.1`）。同样是 `FlyHarness.step`。选择
`flybrain.malecns` 或 `c302.celegans`。未知 id 是 404 /
`UnknownModelError`。不是托管目录。

extra 胶水**仅**在第三方 API 还不是 `ModelBackend` 时才需要：
`fly-harness[flybrain]` 包装 flybrain，好让 biorouter 能列出
`flybrain.malecns`。没有 `fly-harness[c302]` extra。

```bash
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
```

## step 回路

```python
from fly_harness import BrainState, FlyHarness
from fly_harness.protocols import Decoder, Encoder

harness = FlyHarness(state, encoder, decoder)
result = harness.step(observation)
action = result.action
```

`step` 把观测编码成长度 `n_neurons` 的电流向量，把已对接的 **Model** 推进一个 tick，再从神经输出解码出动作。

默认构造仍包装进程内 LIF/rate 动力学（`InProcessLifBackend`）。`FlyHarness.step(obs) -> action` 仍是同一公开调用。

`BrainState` 可以独立于 harness 做快照：

```python
state.save("brain.json")
restored = BrainState.load("brain.json")
```

## 对接模式

两种方式接上正在运行的 Model。本包是 **port**，不是托管的仿真目录。

### 1. Direct — 每个厂商 / 已部署仿真一个 backend

```python
from fly_harness import DirectBioSimBackend, FlyHarness

backend = DirectBioSimBackend(n_neurons=8, model_id="vendor.fake")
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
result = harness.step(observation)
```

`DirectBioSimBackend` 对接**一台**正在运行的仿真。进程内 `FakeDeployedSim` 对测试足够。若你已经托管了一台仿真，可用薄 HTTP 客户端（`url=...`，请求体含 `model` 字段）。没有 FlyWire / CAVE / neuPrint 客户端，也没有 14 万稠密矩阵。

### 2. Router — 一个入口，按 model id 选择

```python
from fly_harness import BioSimRouter, DirectBioSimBackend, FlyHarness, UnknownModelError

router = BioSimRouter(
    {
        "vendor.a": DirectBioSimBackend(n_neurons=8, model_id="vendor.a"),
        "vendor.b": DirectBioSimBackend(n_neurons=8, model_id="vendor.b"),
    },
    model_id="vendor.a",
)
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=router)
harness.step(observation)
router.use("vendor.b")
harness.step(observation)

# unknown model_id raises UnknownModelError (this is a port, not a marketplace)
```

`BioSimRouter` 是 OpenRouter-**shaped**：一个入口、一个 `model` / `model_id` 字段，未知 id 明确失败。可选 `remote_url=` 把该 `model` 字段转发到 **biorouter**（`HttpModelBackend`）。本仓库不提供托管市场。

也可以按位置传 backend：`FlyHarness(backend, encoder, decoder)`。

## 对接你的 Model

harness 不对接到某个特定厂商。实现 `ModelBackend`（`tick` / `reset` /
`n_neurons` / `model_id` / `timestamp`）并交给 `FlyHarness.step`。要在本地
**biorouter** 上列出该 Model，注册一个 OpenRouter-shaped 的 `model` id。Direct 对接和
router 对接都是 **usage**，不是 harness 功能。

短指南：[docs/docking.md](docs/docking.md)。仓库内替身：`FakeDeployedSim`。
可运行 HTTP 路径：`examples/biorouter_loop.py`。`flybrain.malecns` 是**一个已列出的
model**（在 extra + 磁盘数据存在时），不是「harness 对接 flybrain」。
换物种是同一个 port：部署一台正在运行的仿真并列出一个 id。
`c302.celegans` 是**一个已列出的**线虫 Model（默认 CI：`FakeC302`，**302**
条雌雄同体神经元）— usage，不是 harness extra，不是托管 OpenWorm API。

## biorouter

OpenRouter 路由已有的 LLM。**biorouter** 路由已有的**已部署生物仿真模型**。同一思路（`model` id），不同基底。

这个 extra **不是** harness，**不是** FlyWire 转储，也**不是**市场。它是本地 HTTP 进程（默认 `127.0.0.1`），给已经部署的 harness 客户端调用：`HttpModelBackend`、`DirectBioSimBackend(url=...)`、`BioSimRouter(remote_url=...)`。只用 stdlib `http.server` — 无 FastAPI、无计费、无云网关。

```bash
pip install -e ".[biorouter]"   # empty extra; core install is enough to run the CLI
biorouter
# or
python -m fly_harness.router --host 127.0.0.1 --port 8765
```

端点（JSON；请求体/查询含 OpenRouter-shaped 的 `model` 字段）：

- `POST /tick` — `{"model": "...", "input": [...]}` → 神经输出
- `POST /reset` — `{"model": "...", "potentials": null | [...]}`
- `GET /status?model=...`

默认进程内注册表：24 神经元 LIF **fixture**（`fly-harness.in-process-lif`）、`FakeDeployedSim` id（`fake.deployed`、`fake.deployed.gain`），以及 `c302.celegans`（`FakeC302`，**302** 条雌雄同体神经元 — 一个已列出的 Model，不是 fly-harness extra，不是 OpenWorm Docker）。若已安装 `fly-harness[flybrain]` **并且** MaleCNS 文件已在磁盘上，biorouter 也会列出 `flybrain.malecns`（OpenRouter-shaped 的 provider id）。未知 / 缺少 flybrain → HTTP 404 / 跳过。永不下载 MaleCNS。永不启动 OpenWorm Docker。

```python
from fly_harness import DirectBioSimBackend, FlyHarness

backend = DirectBioSimBackend(
    url="http://127.0.0.1:8765",
    model_id="fake.deployed",
    n_neurons=8,
)
harness = FlyHarness(encoder=encoder, decoder=decoder, backend=backend)
harness.step(observation)
```

`FlyHarness.step(obs) -> action` 不变。这个 extra 不是 14 万 FlyWire 运行时，也不下载 MaleCNS。

## 示例：用第三方 flybrain Model 调用 FlyHarness.step

公式：**Agent = Model + harness**。[`flybrain`](https://pypi.org/project/flybrain/)（[源码](https://github.com/alextitonis/fly.ai)）是**第三方 Model**（MaleCNS v1.0 LIF，**166,700 神经元**，2560 万连接）。Google/Janelia 发布的是转储，不是托管仿真。把它接上是在**使用** harness（`ModelBackend`），不是 harness 功能 — 核心保持与模型无关。可选 extra 里放适配器，好让 **biorouter** 能列出 `flybrain.malecns`、本示例能跑。不是 Google 托管，不是 16 万参数，不是意识。

```python
from fly_harness import FlyHarness
from fly_harness.flybrain import (
    FlyBrainBackend,
    FlyBrainInjectEncoder,
    FlyBrainReadoutDecoder,
)

backend = FlyBrainBackend.from_installed(download=False)  # files must already be on disk
harness = FlyHarness(
    encoder=FlyBrainInjectEncoder(backend.n_neurons, channels={"loom": (0, 1)}),
    decoder=FlyBrainReadoutDecoder(backend.n_neurons),
    backend=backend,
)
result = harness.step({"loom": 0.8})
```

默认示例使用进程内 `FakeFlyBrain`（32 神经元）。`--real` 仅在 `flybrain` 可 import **并且** MaleCNS 文件已在磁盘时运行；不会下载它们：

```bash
python examples/flybrain_loop.py
python examples/flybrain_loop.py --real
# or
python -m fly_harness.demo.flybrain_loop
fly-harness-flybrain-demo
```

## 示例：经 biorouter 调用 FlyHarness.step

OpenRouter 路由已有 LLM；**biorouter** 路由已部署的生物仿真。本示例启动**本地** stdlib 服务器（或用 `--url` 挂上），对接 `HttpModelBackend` / `DirectBioSimBackend(url=...)`，再调用 `FlyHarness.step`。它用的是 8 神经元 `FakeDeployedSim` fixture — **不是** FlyWire / MaleCNS，**不是**市场。

```bash
python examples/biorouter_loop.py
# or, after install:
python -m fly_harness.demo.biorouter_loop
fly-harness-biorouter-demo
# attach to a process you already started:
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_loop.py --url http://127.0.0.1:8765
```

## 示例：biorouter 经 FlyHarness.step 路由 flybrain.malecns

OpenRouter 列出 provider id；**biorouter** 用同样方式列出 `flybrain.malecns`。本示例**使用** harness：本地 stdlib HTTP → `HttpModelBackend` / `DirectBioSimBackend(url=...)` → `FlyHarness.step({"loom": ...})`。这不是 harness 功能 — 核心保持与模型无关。默认进程内运行注册一个假的 flybrain-shaped backend（32 神经元），好让 CI 在没有 MaleCNS 时仍绿。`--real` 仅当 `fly-harness[flybrain]` 且文件已在磁盘（永不下载）。缺少 `flybrain.malecns` → HTTP 404。不是 Google 托管，不是 16 万参数，不是意识。

```bash
python examples/biorouter_flybrain_loop.py
python examples/biorouter_flybrain_loop.py --real
# or, after install:
python -m fly_harness.demo.biorouter_flybrain_loop
fly-harness-biorouter-flybrain-demo
# attach to a process you already started (404 if that process did not list the id):
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_flybrain_loop.py --url http://127.0.0.1:8765
```

## 示例：biorouter 经 FlyHarness.step 路由 c302.celegans

换物种是 **usage**，不是 harness 功能：部署一台正在运行的仿真 + 列出一个 OpenRouter-shaped id。**biorouter** 列出 `c302.celegans` 的方式和列出 `flybrain.malecns` 一样。**没有** `fly-harness[c302]` / `fly-harness[openworm]` extra（extra 胶水只用于还不是 `ModelBackend` 的第三方 API；这条线虫路径被否决了）。不是新的全脑 PyPI 包，不是托管 OpenWorm API。

默认进程内运行使用 `FakeC302`（**302** 条雌雄同体神经元），好让 CI 在没有 NEURON / Docker / OpenWorm 时仍绿。`--real` 仅当第三方 `c302` 已可 import **并且** 一台 tick/reset Model 已在运行（永不下载连接组，永不启动 OpenWorm Docker）。缺少 / 未知 id → HTTP 404 / `UnknownModelError`。不是意识。

```bash
python examples/biorouter_c302_loop.py
python examples/biorouter_c302_loop.py --real
# or, after install:
python -m fly_harness.demo.biorouter_c302_loop
fly-harness-biorouter-c302-demo
# attach to a process you already started (404 if that process did not list the id):
biorouter --host 127.0.0.1 --port 8765
python examples/biorouter_c302_loop.py --url http://127.0.0.1:8765
```

## 加载稀疏连接组

`load_connectome` 把全局神经元 id（任意整数）映射到连续局部下标，并构建形状为 `(n_subset, n_subset)` 的 CSR 权重矩阵。

```python
from fly_harness import FlyHarness, load_connectome

loaded = load_connectome("tests/fixtures/mini_circuit.npz")
# loaded.state.weights is scipy.sparse CSR, shape (n_subset, n_subset)
# loaded.neuron_ids maps local index -> global neuron id
# loaded.id_to_index maps global id -> local index

harness = FlyHarness(loaded.state, encoder=..., decoder=...)
result = harness.step(observation)
```

```bash
fly-harness-loader-demo
# or
python -m fly_harness.demo.loaded_circuit
```

支持的文件：

- **NPZ** — 数组 `pre`、`post`、`weight`（整数 id + 浮点权重）；可选 `neuron_ids` 表示回路子集
- **CSV** — 表头 `pre_id,post_id,weight`

用 `neuron_ids=[...]` 或 `neuron_ids_file="ids.txt"`（每行一个 id）传入显式子集。只保留 pre 和 post 都在子集内的突触。

## Fixture：24 神经元反射

随包装运，只为了让契约在没有外部连接组文件时也能执行。左侧触觉偏向右转（反之亦然）。不要把它当成生物回路。

```bash
fly-harness-reflex-demo
# or
python -m fly_harness.demo.reflex
```

```python
from fly_harness import FlyHarness
from fly_harness.demo import ReflexDecoder, ReflexEncoder, build_reflex_connectome
from fly_harness.demo.codec import TouchObservation

state = build_reflex_connectome()
harness = FlyHarness(state, ReflexEncoder(), ReflexDecoder())
result = harness.step(TouchObservation(touch_left=1.0))
print(result.action)  # ReflexAction(turn=1, forward=..., brake=...)
```

## 可选身体：FlyGym / NeuroMechFly

这是 **extra（扩展）**，不是核心产品，也不是架构。它不附带 14 万神经元大脑，不改写 FlyGym，不装 extra 就不跑 MuJoCo。FlyGym / CPG **不是** harness 内核：`FlyHarness.step` 是脑级的，在身体运动 CPG **之上**。

`FlyGymEncoder` / `FlyGymDecoder` 把 NeuroMechFly 观测（关节角、接触力）和动作（关节目标、每腿黏附、可选肌肉/肌腱命令）映射到 `FlyHarness.step`。它们在 dict 上工作 — 不 import MuJoCo。`FlyGymHarnessEnv` 是对**你**用 FlyGym 构造的 env 的薄包装。

```python
from fly_harness import FlyHarness
from fly_harness.demo.connectome import SENSORY_INDICES, build_reflex_connectome
from fly_harness.flygym import FlyGymDecoder, FlyGymEncoder, FlyGymHarnessEnv

state = build_reflex_connectome()  # fixture circuit, not FlyWire
harness = FlyHarness(
    state,
    FlyGymEncoder(state.n_neurons),
    FlyGymDecoder(state.n_neurons),
    sensory_indices=SENSORY_INDICES,
)
# env = your FlyGym / NeuroMechFly simulation (not constructed here)
body = FlyGymHarnessEnv(env, harness)
obs, info = body.reset()
result = body.step()
# result.action.as_env_dict() -> {"joints": ..., "adhesion": ...}
```

Gymnasium FlyGym（`flygym-gymnasium`，由 extra 安装；import 名 `flygym` 或 `flygym_gymnasium`）接受 `env.step({"joints", "adhesion"})`。FlyGym 2.x `Simulation` 在你传入 `actuator_type` 时通过 `set_actuator_inputs` / `set_leg_adhesion_states` 做 duck-type。

## 可选协议：MCP

这是 **extra 扩展包**，不是微内核。`BrainState` / `FlyHarness` 仍只有 numpy/scipy，不 import MCP。默认服务器步进的是 24 神经元反射 **fixture**，不是 14 万 FlyWire 大脑。

```bash
pip install -e ".[mcp]"
# stdio (default FastMCP transport) — point an MCP host at this command:
fly-harness-mcp
# or
python -m fly_harness.mcp
```

工具（单元测试 handler 不需要活客户端）：

- `harness_step(touch_left, touch_right)` — 一次 `FlyHarness.step`
- `harness_reset()` — 电位和时钟清零
- `harness_status()` — 神经元数量、时间戳、包版本

Handler 可在没有 FastMCP 时 import：

```python
from fly_harness.mcp import HarnessSession

session = HarnessSession()
print(session.step(touch_left=1.0)["action"])
```

## 示例：经 harness 把协议 + 身体接在一起

这证明可选 **MCP** 和 **FlyGym** extra 可以接到同一个 `FlyHarness.step`。它用的是 24 神经元反射 **fixture**。它**不是** 14 万 FlyWire 或约 16.6 万 MaleCNS 上传，也不改写 FlyGym 或 MCP 包。

```bash
python examples/mcp_flygym_loop.py
# or, after install:
python -m fly_harness.demo.composed
fly-harness-composed-demo
```

默认运行使用进程内身体 stub（无 MuJoCo）。`--try-flygym` 尝试真实 NeuroMechFly 仿真，失败则回退到 stub。`--serve` 在该组合 session 上启动可选 FastMCP stdio 服务器（`pip install -e ".[mcp]"`）：

```bash
python examples/mcp_flygym_loop.py --serve
```

因此，调用 `harness_step(touch_left, touch_right)` 的 MCP host 会步进 harness，并把解码出的关节/黏附命令应用到身体。

## 示例：FlyGym 身体经 biorouter 接到 flybrain.malecns

FlyGym obs → `FlyHarness.step` → 本地 **biorouter** → 已列出的 model id `flybrain.malecns` → 解码出 FlyGym 关节/黏附动作再回到身体。把 FlyGym / flybrain 接上是在**使用** harness，不是 harness 功能。核心保持与模型无关。

默认进程内运行：`StubFlyGymEnv`（无 MuJoCo）+ 列为 `flybrain.malecns` 的 `FakeFlyBrain`（32 神经元）。`--real` 仅当 `fly-harness[flybrain]` 且 MaleCNS 文件已在磁盘（永不下载）。`--try-flygym` 尝试真实 NeuroMechFly 仿真，失败则回退到 stub。缺少 / 未知 model id → HTTP 404。不是 Google 托管，不是 16 万参数，不是意识。

```bash
python examples/body_loop.py
python examples/body_loop.py --real
python examples/body_loop.py --try-flygym
# or, after install:
python -m fly_harness.demo.body_loop
fly-harness-body-loop-demo
```

## 示例：同一回路，替换 Model

同一 FlyGym 身体。同一条 `obs → FlyHarness.step → action` 回路。只替换
Model。这不宣称比 FlyGym 自己的控制器走得更好。
FlyGym / CPG 不是 harness 内核；`FlyHarness.step` 仍是脑级的，
在运动 CPG 之上。

- **A** — 默认玩具进程内 LIF（`fly-harness.in-process-lif`，24 神经元 fixture）
- **B** — biorouter id `flybrain.malecns`

复用 body-loop / biorouter 示例。核心不绑定厂商 Model。
默认 B 在该 id 下列出 FakeFlyBrain（32 神经元，**不是** 166,700）。
`--real` 仅当 `fly-harness[flybrain]` 且 MaleCNS 文件已在磁盘
（永不下载）。缺少 extra 或数据 → 诚实 **跳过 B**；从不伪造
16.6 万神经元大脑。

```bash
python examples/swap_brain.py
python examples/swap_brain.py --real
python examples/swap_brain.py --try-flygym
# or, after install:
python -m fly_harness.demo.swap_brain
fly-harness-swap-brain-demo
```

## 测试

GitHub Actions 在 `main` 和 pull request 上跑 `pip install -e ".[dev]"` 然后 `pytest` — 不装 FlyGym/MCP/flybrain extra，无 MuJoCo，**不下载 MaleCNS**，**不启动 OpenWorm Docker**。这些 extra 里的 skip/mock 测试仍会通过。**biorouter** extra 只有 stdlib，所以它的测试跑在同一套核心 CI 里。load-check（`fly-harness-check`）是核心：玩具进程内 LIF，无 extra。

默认本地套件**不**需要 MuJoCo。FlyGym smoke 测试在 `flygym` 无法 import 或 NeuroMechFly 仿真无法启动时 skip：

```bash
pytest
# load-check (just-installed path; no extras):
pytest tests/test_check.py
fly-harness-check
python -m fly_harness
# with the FlyGym extra, still skip-friendly if MuJoCo/display is missing:
pytest -q
# FlyGym smoke only:
pytest tests/test_flygym_adapter.py -k smoke
```

Model-backend 对接（默认 LIF、假已部署仿真、router `model_id` 切换、未知 id 报错）是核心，不需要 extra：

```bash
pytest tests/test_backend.py tests/test_harness.py
```

第三方对接指南只是 markdown（`docs/docking.md`）。内容测试检查 `ModelBackend` 契约和 biorouter 列出说明，不需要 extra：

```bash
pytest tests/test_docking_guide.py
```

可选 **biorouter** 进程启动 stdlib 服务器，经 `HttpModelBackend` / `FlyHarness` tick（仍无 FastAPI，不需要 extra）：

```bash
pytest tests/test_router.py
# after install:
biorouter --help
python -m fly_harness.router --help
```

biorouter 示例默认在进程内（没有活的 `biorouter` 守护进程）。挂到真实进程由 `BIOROUTER_URL` skip 门控：

```bash
pytest tests/test_biorouter_example.py
python examples/biorouter_loop.py
```

Router 测试可以在 `flybrain.malecns` 下注册一个**假的 flybrain-shaped** backend（无 extra，无下载）。没有 flybrain/数据的默认 CI 省略该 id（HTTP 404）。真实 flybrain smoke 仅在 extra 已安装 **并且** `~/fly-data` 已有文件时运行（CI 不下载）：

```bash
pytest tests/test_router.py tests/test_flybrain_adapter.py
python examples/flybrain_loop.py
# optional, only with extra + on-disk MaleCNS:
# python examples/flybrain_loop.py --real
```

biorouter → `flybrain.malecns` 示例默认在进程内（该 id 下列出假的 32 神经元 backend）。经 HTTP 的真实 MaleCNS 仅在 extra + 数据已在磁盘时运行，否则 skip：

```bash
pytest tests/test_biorouter_flybrain_example.py
python examples/biorouter_flybrain_loop.py
```

默认 CI 也把 `c302.celegans` 列为 `FakeC302`（302 神经元），好让测试能选两个 model id（假 `flybrain.malecns` + 假 `c302.celegans`），无需 NEURON / Docker / OpenWorm。真实 c302 仅当一台正在运行的 tick/reset Model 已可 import 时运行（永不下载，永不启动 OpenWorm Docker）：

```bash
pytest tests/test_biorouter_c302_example.py tests/test_router.py
python examples/biorouter_c302_loop.py
```

MCP 测试 mock FastMCP，**不**启动活的 MCP 客户端。没有 `fly-harness[mcp]` 也会通过。装了 extra 时，factory smoke 检查 FastMCP 能否构造（仍无 stdio 客户端）：

```bash
pip install -e ".[mcp]"
pytest tests/test_mcp_extension.py
```

组合 MCP+FlyGym 示例同样是 mock/skip：无 MuJoCo，无活 MCP 客户端。

```bash
pytest tests/test_composed_example.py
python examples/mcp_flygym_loop.py
```

body-loop 示例是 stub 身体 + 进程内假 `flybrain.malecns`（无 MuJoCo，不下载 MaleCNS）。真实 extra 仅在已经存在时运行，否则 skip：

```bash
pytest tests/test_body_loop_example.py
python examples/body_loop.py
```

swap-Model 示例是同一身体 + 同一回路，两个 `model_id`
（玩具 LIF，然后 `flybrain.malecns`）。默认 CI 使用假的已列出 id（不
下载 MaleCNS）。没有 extra/数据时 `--real` 会跳过 B：

```bash
pytest tests/test_swap_brain_example.py
python examples/swap_brain.py
```

## 许可证

MIT — 见 [LICENSE](LICENSE)。
