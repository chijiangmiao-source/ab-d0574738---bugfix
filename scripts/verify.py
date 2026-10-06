#!/usr/bin/env python3
"""可执行验收服务 verify。

执行顺序（任一阶段失败立即以非零退出码结束）：
  1. 代码测试：首次回执不可变 / 迟到观测重放 / 窗口规则、重开恢复（含
     历史 v1 卷自愈）、旧结果抑制等（unittest）。
  2. 构建检查：字节编译全部 Python 源码；校验可交付页面资产
     （index.html / app.js 存在且相互引用；若环境有 node，则做 JS 语法检查）。
  3. HTTP 冒烟：健康地址 + 页面静态资源 + 全部业务 API/HTTP 路径，
     并核对接受 / 重放 / 拒绝结论、单调修订号与窗口外不变性。
  4. 持久化卷重开场景（仅自托管模式）：保存首次回执 → 窗口内迟到观测 →
     同标识重传 → 关闭进程后在同一状态文件上重开 → 再次重传，
     核对历史回执恒定而当前后缀确已更新。

目标服务：
  * 设置环境变量 BASE_URL（如 http://web:8080）时对该地址冒烟；
    Compose 中 verify 服务即通过该方式对 web 做端到端验收。
  * 未设置时，本脚本自行在临时端口拉起服务、使用临时状态文件，验收后清理。
  * VERIFY_RESET=1 时，冒烟结束后调用 /api/reset 清掉验收产生的数据
    （Compose 一次性验收场景）。

退出码：0 = 验收通过；1 = 失败。
"""

from __future__ import annotations

import json
import os
import py_compile
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail and not ok else ""))
    return ok


# ---------------------------------------------------------------------------
# 阶段 1：代码测试
# ---------------------------------------------------------------------------

def run_code_tests() -> bool:
    print("== 阶段 1：代码测试（unittest） ==")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
    return check("代码测试全部通过", proc.returncode == 0, tail)


# ---------------------------------------------------------------------------
# 阶段 2：构建检查
# ---------------------------------------------------------------------------

def run_build_checks() -> bool:
    print("== 阶段 2：构建检查（可交付页面） ==")
    ok = True

    # 2a. Python 源码字节编译
    py_files = list((ROOT / "app").rglob("*.py")) + [ROOT / "scripts" / "verify.py"]
    compile_error = ""
    for f in py_files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            ok = False
            compile_error = str(exc)
            break
    ok = check("Python 源码字节编译通过", ok, compile_error) and ok

    # 2b. 页面资产完整且相互引用
    index = ROOT / "app" / "static" / "index.html"
    appjs = ROOT / "app" / "static" / "app.js"
    assets_ok = index.exists() and appjs.exists()
    if assets_ok:
        html = index.read_text(encoding="utf-8")
        js = appjs.read_text(encoding="utf-8")
        bundle = html + js
        assets_ok = (
            "/static/app.js" in html
            and "api/observations" in js
            and "api/state" in js
            and all(decision in bundle for decision in ("ACCEPTED", "REPLAYED", "REJECTED"))
        )
    ok = check("页面资产存在且引用业务 API 与三类结论", assets_ok) and ok

    # 2c. 若有 node，对页面脚本做语法检查
    node = shutil_which("node")
    if node:
        proc = subprocess.run([node, "--check", str(appjs)], capture_output=True, text=True)
        ok = check("app.js 通过 node --check 语法检查", proc.returncode == 0, proc.stderr) and ok
    else:
        print("[SKIP] 未找到 node，跳过 JS 语法检查（不影响交付）")

    return ok


def shutil_which(name: str):
    from shutil import which

    return which(name)


# ---------------------------------------------------------------------------
# 阶段 3：HTTP 冒烟
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, payload=None, timeout: float = 5.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def http_json(method: str, url: str, payload=None, timeout: float = 5.0):
    status, body = http(method, url, payload, timeout)
    try:
        return status, json.loads(body) if body else {}
    except json.JSONDecodeError:
        return status, {}


def wait_healthy(base: str, attempts: int = 50) -> bool:
    for _ in range(attempts):
        try:
            status, _ = http("GET", base + "/healthz", timeout=1.0)
            if status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


CONFIG = {
    "x0": [0, 0, 1, 0],
    "P0": [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
    "q": 0.5,
    "r": 1,
    "lag": 2,
}


def smoke(base: str) -> bool:
    print(f"== 阶段 3：HTTP 冒烟（{base}） ==")
    ok = True

    ok = check("健康地址 GET /healthz 返回 200/ok", wait_healthy(base)) and ok
    if not ok:
        return False  # 目标不可达：后续调用无意义，直接以 FAIL/退出码 1 结束

    # 保证验收可重复执行：先清空任何历史数据（持久化卷上的旧状态）
    http_json("POST", base + "/api/reset", {})

    status, body = http_json("GET", base + "/healthz")
    ok = check("健康响应内容 status=ok", status == 200 and body.get("status") == "ok") and ok

    status, html = http("GET", base + "/")
    ok = check(
        "交付页面 GET / 可访问且为 HTML",
        status == 200 and "<!DOCTYPE html>" in html and "观测" in html,
    ) and ok
    status_js, js_body = http("GET", base + "/static/app.js")
    ok = check(
        "页面脚本 GET /static/app.js 返回 200 且为业务脚本",
        status_js == 200 and "api/observations" in js_body,
    ) and ok
    status_404, _ = http_json("GET", base + "/no-such-path")
    ok = check("未知路径返回 404", status_404 == 404) and ok

    # 非法配置必须 422 且保留状态
    bad = dict(CONFIG, q=0)
    status, body = http_json("POST", base + "/api/config", bad)
    ok = check("非法噪声（q=0）返回 422 并说明原因", status == 422 and "q" in body.get("error", "")) and ok
    bad_p0 = dict(CONFIG)
    bad_p0["P0"] = [[1, 0, 0, 0], [0, 1, 2, 0], [2, 0, 1, 0], [0, 0, 0, 1]]
    status, body = http_json("POST", base + "/api/config", bad_p0)
    ok = check("非对称 P0 返回 422", status == 422) and ok

    status, state0 = http_json("POST", base + "/api/config", CONFIG)
    ok = check("合法配置 POST /api/config 返回 200", status == 200 and state0["revision"] == 0) and ok

    def obs(oid, t, x, y):
        return http_json("POST", base + "/api/observations", {"id": oid, "timestamp": t, "x": x, "y": y})

    status, a = obs("VRF-A", 1.0, 1.0, 0.0)
    ok = check("观测 A 接受，修订号=1", a.get("decision") == "ACCEPTED" and a.get("track_revision") == 1) and ok
    _, b = obs("VRF-B", 2.0, 2.0, 0.0)
    _, c = obs("VRF-C", 3.0, 3.0, 0.0)
    ok = check("顺序观测 B/C 接受且修订号单调 (2,3)",
               b.get("decision") == "ACCEPTED" and c.get("track_revision") == 3) and ok

    # 关键：窗口内迟到观测重放后缀
    _, late = obs("VRF-LATE", 1.5, 1.2, 0.5)
    _, state = http_json("GET", base + "/api/state")
    ts = [p["timestamp"] for p in state["track"]["points"]]
    ok = check(
        "窗口内迟到观测接受：修订号=4 且轨迹插入 1.5",
        late.get("decision") == "ACCEPTED" and late.get("track_revision") == 4 and ts == [1.0, 1.5, 2.0, 3.0],
        f"ts={ts}",
    ) and ok
    ok = check("自检查点重算后缀：anchor_seq 指向封存位置", state.get("anchor_seq") == 0) and ok

    # 窗口外拒绝且轨迹不变
    _, old = obs("VRF-OLD", 0.2, 9.0, 9.0)
    _, state_after = http_json("GET", base + "/api/state")
    ok = check(
        "窗口外观测拒绝且已发布轨迹/修订号不变",
        old.get("decision") == "REJECTED"
        and "窗口" in old.get("reason", "")
        and state_after["revision"] == 4
        and state_after["track"] == state["track"],
    ) and ok

    # 同标识同内容回放原结论
    _, replay = obs("VRF-A", 1.0, 1.0, 0.0)
    ok = check(
        "同标识同内容回放原结论（REPLAYED，修订号不变）",
        replay.get("decision") == "REPLAYED" and replay.get("track_revision") == 4,
    ) and ok

    # 同刻非递增拒绝
    _, samets = obs("VRF-D", 2.0, 2.1, 0.0)
    ok = check("同刻非递增顺序拒绝", samets.get("decision") == "REJECTED") and ok

    # 同标识内容不同拒绝
    _, conflict = obs("VRF-A", 1.0, 1.0, 0.9)
    ok = check("同标识内容不同拒绝", conflict.get("decision") == "REJECTED") and ok

    # 非法 JSON
    req = urllib.request.Request(
        base + "/api/observations", data=b"{not-json", headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=3)
        bad_json_ok = False
    except urllib.error.HTTPError as exc:
        bad_json_ok = exc.code == 400
    ok = check("非法 JSON 请求返回 400", bad_json_ok) and ok

    _, final_state = http_json("GET", base + "/api/state")
    decisions = [e["decision"] for e in final_state["log"]]
    ok = check(
        "不可变日志包含全部留痕（接受/重放/拒绝）且修订号停在 4",
        decisions.count("ACCEPTED") == 4
        and "REPLAYED" in decisions
        and decisions.count("REJECTED") >= 3
        and final_state["revision"] == 4,
        str(decisions),
    ) and ok
    return ok


# ---------------------------------------------------------------------------
# 阶段 4：持久化卷上的首次回执场景（保存首次回执 → 窗口内迟到观测
#          → 同标识重传 → 重开后再次重传）
# ---------------------------------------------------------------------------

RECEIPT_CONFIG = {
    "x0": [0, 0, 1, 0],
    "P0": [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
    "q": 0.5,
    "r": 1,
    "lag": 2,
}


def _receipt_key(body: dict) -> dict:
    snap = body.get("snapshot") or {}
    return {
        "timestamp": snap.get("timestamp"),
        "state": snap.get("state"),
        "P_diag": snap.get("P_diag"),
        "residual": snap.get("residual"),
    }


def prepare_receipt_scenario(base: str):
    """重置后在卷上建立：保存中间观测首次回执 → 录入窗口内迟到观测 →
    同标识重传。返回 (first_receipt, state_before_retransmit)。"""
    http_json("POST", base + "/api/reset", {})
    status, _ = http_json("POST", base + "/api/config", RECEIPT_CONFIG)
    assert status == 200

    def obs(oid, t, x, y):
        return http_json("POST", base + "/api/observations", {"id": oid, "timestamp": t, "x": x, "y": y})

    obs("RCP-A", 1.0, 1.0, 0.0)
    _, mid = obs("RCP-B", 2.0, 2.0, 0.0)  # 保存中间一条被接受时返回的回执
    obs("RCP-C", 3.0, 3.0, 0.0)
    # 采样时刻位于前两条之间、落在滞后窗口（lag=2）内的迟到观测
    _, late = obs("RCP-LATE", 1.5, 1.2, 0.5)
    return mid, late


def check_receipt_retransmit(base: str, first_receipt: dict, tag: str) -> bool:
    """同标识同内容重传：REPLAYED 且回执恒为首次接受证据。"""
    ok = True

    def obs(oid, t, x, y):
        return http_json("POST", base + "/api/observations", {"id": oid, "timestamp": t, "x": x, "y": y})

    _, state = http_json("GET", base + "/api/state")
    cur_t2 = next(p for p in state["track"]["points"] if p["timestamp"] == 2.0)
    ok = check(
        f"{tag}：窗口内迟到观测已修正当前后缀（t=2 最新投影≠首次回执）",
        cur_t2["state"] != first_receipt["state"],
    ) and ok
    ok = check(
        f"{tag}：当前轨迹含迟到观测时刻 1.5，修订号=4",
        [p["timestamp"] for p in state["track"]["points"]] == [1.0, 1.5, 2.0, 3.0]
        and state["revision"] == 4,
    ) and ok
    ok = check(
        f"{tag}：封存检查点指向最早观测（anchor_seq=0，lag=2 边界 t=1）",
        state.get("anchor_seq") == 0,
    ) and ok

    _, again = obs("RCP-B", 2.0, 2.0, 0.0)
    got = _receipt_key(again)
    ok = check(
        f"{tag}：同标识同内容重传返回 REPLAYED 且首次回执恒定（state/P_diag/residual）",
        again.get("decision") == "REPLAYED"
        and got == first_receipt
        and again.get("track_revision") == 4,
        f"got={got}",
    ) and ok

    # 回归：同标识异内容冲突
    _, conflict = obs("RCP-B", 2.0, 2.0, 0.9)
    ok = check(f"{tag}：同标识异内容冲突仍拒绝", conflict.get("decision") == "REJECTED") and ok
    return ok, state


def receipt_smoke(base: str) -> bool:
    print(f"== 阶段 4：持久化卷首次回执场景（{base}） ==")
    mid, late = prepare_receipt_scenario(base)
    ok = True
    ok = check(
        "保存首次回执：三条递增观测接受、中间观测回执含 state/P_diag/residual",
        mid.get("decision") == "ACCEPTED"
        and late.get("decision") == "ACCEPTED"
        and all(k in (mid.get("snapshot") or {}) for k in ("state", "P_diag", "residual")),
    ) and ok
    first_receipt = _receipt_key(mid)
    more, _ = check_receipt_retransmit(base, first_receipt, "重开前")
    return ok and more, first_receipt


def reopen_receipt_smoke(base: str, first_receipt: dict) -> bool:
    """服务以同一持久化状态文件重开后，再次重传仍返回原始首次回执，
    且当前轨迹/单调修订号/封存位置与重开前一致。"""
    print(f"== 阶段 5：关闭重开后在持久化卷上再次重传（{base}） ==")
    ok = check("重开后健康检查通过", wait_healthy(base))
    more, state = check_receipt_retransmit(base, first_receipt, "重开后")
    # 重开后：当前轨迹、修订号、封存位置依旧正确
    ok = check(
        "重开后当前轨迹仍含迟到后缀、修订号停在 4 且封存检查点保留",
        [p["timestamp"] for p in state["track"]["points"]] == [1.0, 1.5, 2.0, 3.0]
        and state["revision"] == 4
        and state["anchor_seq"] == 0,
    ) and ok
    return ok and more


def _spawn_server(port: int, state_file: str):
    return subprocess.Popen(
        [sys.executable, "-m", "app.server", "--port", str(port), "--state", state_file],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_server(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_receipt_restart_cycle() -> bool:
    """保存首次回执 → 窗口内迟到观测 → 同标识重传 → 关闭 → 在同一持久化
    状态文件上重开 → 再次重传。始终自管一对服务进程以真实经历“重开”：
    Compose 下状态文件位于挂载的 nav-data 卷（/data），本地为临时目录。"""
    state_dir = os.environ.get("VERIFY_STATE_DIR", "").strip()
    tmp_dir = None
    if not state_dir:
        tmp_dir = tempfile.TemporaryDirectory(prefix="nav-verify-vol-")
        state_dir = tmp_dir.name
    else:
        os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, "verify-receipt-state.json")
    if os.path.exists(state_file):
        os.unlink(state_file)  # 每次验收独立开始；场景内的重开才是被检验对象

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = _spawn_server(port, state_file)
    try:
        if not wait_healthy(base):
            return check("重开前验收服务启动（健康检查）", False)
        ok, first_receipt = receipt_smoke(base)
        if not ok:
            return False
    finally:
        _stop_server(proc)

    # 关闭服务并在同一持久化状态文件上重新启动
    proc = _spawn_server(port, state_file)
    try:
        ok = reopen_receipt_smoke(base, first_receipt)
    finally:
        _stop_server(proc)
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return ok


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    if not run_code_tests():
        return 1
    if not run_build_checks():
        return 1

    base = os.environ.get("BASE_URL", "").rstrip("/")
    spawned = None
    tmp_state = None
    try:
        if not base:
            port = _free_port()
            tmp_state = tempfile.NamedTemporaryFile(prefix="nav-verify-", suffix=".json", delete=False)
            tmp_state.close()
            spawned = _spawn_server(port, tmp_state.name)
            base = f"http://127.0.0.1:{port}"

        if not smoke(base):
            return 1

        # 持久化卷上的保存 → 迟到 → 重传 → 重开 → 再重传（自管服务对，
        # Compose 下落在 nav-data 卷；与上面的冒烟服务互不干扰）。
        if not run_receipt_restart_cycle():
            return 1

        if os.environ.get("VERIFY_RESET") == "1":
            http("POST", base + "/api/reset", {})
            print("[INFO] 已调用 /api/reset 清理验收数据")
        print("\n验收结果：全部通过 ✅")
        return 0
    finally:
        if spawned is not None:
            _stop_server(spawned)
        if tmp_state is not None:
            try:
                os.unlink(tmp_state.name)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
