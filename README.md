# 巡检机返航 · 导航轨迹固定滞后复核

巡检机返航后，导航审查员需要把**延迟到达的定位观测**纳入既有惯导轨迹。
本系统实现一个二维（位置 + 速度，匀速模型）线性卡尔曼滤波器，并按
**固定滞后窗口（Fixed-Lag Smoothing）** 规则管理观测与轨迹投影：

* 窗口内迟到观测：从窗口边界上的**检查点**重算其后历史后缀，产生新修订；
* 检查点之前的**已封存位置逐位不变**，绝不改写；
* **首次回执不可变**：观测首次被接受时返回的状态、协方差对角线、残差及结论
  被永久冻结；之后窗口内迟到观测只更新当前轨迹投影。同标识同内容的重传
  （无论发生在重放前后还是服务重开后）一律返回该首次回执，而不是最新投影
  中同一时刻的结果；旧版本已被改写的持久化回执在重开时确定性恢复；
* 窗口外观测、同刻非递增顺序、同标识内容冲突等一律拒绝，不改变已发布轨迹；
* 观测日志**只追加、不可变**，修订号**单调递增**；
* 计算期间可继续录入；旧计算结果带代际令牌，提交时 CAS，过期结果一律丢弃，
  **不得覆盖最新页面**；
* 状态原子持久化（tmp + `os.replace`），**重开后恢复相同日志与轨迹**。

零第三方运行时依赖：服务端仅用 Python 3.11 标准库（自带极简稠密线性代数），
前端为原生 HTML/JS，无需构建。

## 目录

```
app/
  linalg.py        纯 Python 矩阵运算 / Cholesky 正定校验 / 2x2 求逆
  kf.py            二维 CV 卡尔曼滤波 + 检查点/后缀重放（FixedLagSmoother）
  store.py         不可变日志、单调修订号、窗口规则、CAS 旧结果抑制、持久化
  server.py        HTTP 服务（健康检查 / 页面 / 业务 API），端口可配置
  static/          复核页面（逐条展示接受/重放/拒绝、当前位置、协方差对角、残差）
tests/test_core.py 代码测试（26 个）
scripts/verify.py  可执行验收服务 verify
Dockerfile, compose.yaml
```

## 本地运行

```bash
python3 -m app.server --port 8080 --state ./data/state.json
# 或：PORT=8080 STATE_FILE=./data/state.json python3 -m app.server
```

打开 http://localhost:8080 ：

1. 建立二维位置/速度初值 `x0=[px,py,vx,vy]`、4×4 初始协方差 `P0`（须对称正定）、
   过程噪声 `q>0`、观测噪声 `r>0`、滞后长度 `lag≥0`（秒）；
2. 按接收顺序录入观测（稳定标识 `id`、采样时刻 `timestamp`、二维位置 `x/y`），
   最多 32 条；
3. 逐条查看 **ACCEPTED / REPLAYED / REJECTED** 结论、当前位置、
   协方差对角与创新残差，以及已发布轨迹与不可变日志。

## HTTP 接口

| 方法/路径 | 说明 |
| --- | --- |
| `GET /healthz` | 健康响应 `{"status":"ok"}` |
| `GET /`、`GET /static/app.js` | 可交付页面 |
| `GET /api/state` | 配置、修订号、检查点、日志、轨迹、当前位置 |
| `POST /api/config` | 建立初值/噪声/滞后；非法矩阵返回 422 并说明原因 |
| `POST /api/observations` | 录入一条观测，返回逐条结论 |
| `POST /api/reset` | 清空日志与轨迹 |

失败语义：非法噪声矩阵、不可逆创新矩阵、窗口外观测等都返回明确原因，
并**保留最近一次有效轨迹与修订号**。

`POST /api/observations` 的回执除本条结论外还返回：`snapshot`（该观测首次
接受时冻结的状态、协方差对角、残差）、`receipt_decision` / `receipt_revision`
（首次回执自带的结论与修订号）、`current` / `track_revision`（当前最新投影）。
因此 **REPLAYED** 响应里 `snapshot` 恒定为首次接受证据，而 `track_revision`
反映当前轨迹修订，二者天然可不同。

## 验收服务 verify

`scripts/verify.py` 是可执行验收服务，依次执行：

1. **代码测试**（`unittest`）：迟到观测窗口内重放后缀且封存位置不变、
   窗口外拒绝、同标识重放/冲突、同刻拒绝、**不可变首次回执（含被旧版本
   改写记录的重开恢复）**、**重开恢复**、**旧结果抑制**、
   非法噪声/奇异创新防护；
2. **构建检查**：全部 Python 源码字节编译，页面资产完整，
   `app.js` 通过 `node --check`（有 node 时）；
3. **HTTP 冒烟**：健康地址、页面静态资源、全部业务 API/HTTP 路径与三类结论；
4. **持久化重开验收**：在持久化数据卷上覆盖
   “保存首次回执 — 录入窗口内迟到观测 — 同标识重传 — **关闭重开后**再次重传”，
   核对历史回执恒定而当前后缀确已更新；并回归窗口外拒绝不改写轨迹、
   同标识异内容冲突、正常迟到重放（旧结果抑制由阶段 1 确定性覆盖）。

执行后退出，退出码即验收结论（0 通过 / 1 失败）：

```bash
# 本地：自行在临时端口拉起服务并冒烟
python3 scripts/verify.py

# 对已运行服务冒烟（Compose verify 服务即此模式）
BASE_URL=http://127.0.0.1:8080 VERIFY_RESET=1 python3 scripts/verify.py
```

## Compose

```bash
docker compose up web          # 启动应用（宿主端口可用 HOST_PORT 覆盖，默认 8080）
docker compose run verify      # 运行一次性验收服务，查看其退出码
docker compose up              # 同时启动；verify 等 web 健康后执行并退出
```

`web` 带 healthcheck 与持久化卷 `nav-data`；`verify` 挂载同一卷、依赖 `web`
健康通过，先对 `http://web:${PORT}` 做端到端冒烟，再在卷上用一次性进程真实
**关闭并重新打开**服务核对首次回执与恢复，完成即退出（`Exited (0)` 表示通过）。

## 关键规则的实现位置

| 需求 | 实现 |
| --- | --- |
| 仅修正固定滞后窗口内历史后缀 | `FixedLagSmoother.replay`：边界 `t_max-lag` 上的最后一个观测为检查点，前缀封存、后缀重算 |
| 已封存位置不改写 | 前缀由相同输入确定性重放，测试 `test_late_within_window_...` 逐位断言 |
| 首次回执不可变（重传/重开恒定） | 接受时冻结入 `Store._receipts`，`_replay_locked` 永远回放冻结回执（沿 REPLAYED 链找到首次接受）；重开由 `_heal_receipts_locked` 恢复（旧文件 `_rebuild_original_receipts_locked` 按接受顺序逐前缀重放重建） |
| 同标识同内容回放原结论 / 内容不同拒绝 | `Store._quick_decide_locked` 同标识分支 |
| 窗口外观测、同刻非递增拒绝 | 同文件窗口与同刻分支 |
| 不可变日志 / 单调修订号 | `Store` 中 `_log` 只追加；仅接受成功时 `_revision += 1` |
| 计算中继续录入、旧结果不覆盖最新页面 | 锁内校验 → **锁外重放** → 锁内 CAS（代际 + next_seq），失败重试；前端再按修订/日志长度抑制 |
| 重开恢复相同日志与轨迹 | 原子写 JSON；启动时 `_load`，测试 `test_reopen_restores_log_and_track` |
| 非法噪声 / 奇异创新 | `FilterConfig.validate`（Cholesky）、`_kf_step`（S 求逆+特征值），失败保留最近轨迹 |
