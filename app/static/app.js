/* 导航轨迹固定滞后复核页面脚本（原生 JS，无构建依赖）。
 *
 * 旧结果抑制（客户端侧）：每次请求带单调 requestId；响应只在其
 * requestId 为已完成的最大序号、且服务端修订号/日志长度不小于当前展示时
 * 才允许刷新页面，确保迟到的旧修订结果永不覆盖最新页面。
 */
"use strict";

const $ = (id) => document.getElementById(id);
let shownRevision = -1;
let shownLogSize = -1;
let lastRequestId = 0;
let maxAppliedId = 0;

/* ---------- P0 默认对角矩阵 ---------- */
function buildP0Grid() {
  const grid = $("p0grid");
  const diag = [10, 10, 4, 4];
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "any";
      inp.dataset.i = i;
      inp.dataset.j = j;
      inp.value = i === j ? diag[i] : 0;
      grid.appendChild(inp);
    }
  }
}

function readP0() {
  const P0 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  document.querySelectorAll("#p0grid input").forEach((el) => {
    P0[+el.dataset.i][+el.dataset.j] = parseFloat(el.value);
  });
  return P0;
}

function num(id) {
  return parseFloat($(id).value);
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined) return "—";
  if (Math.abs(v) >= 1e6 || (v !== 0 && Math.abs(v) < 1e-4)) return v.toExponential(3);
  return Number(v).toFixed(digits);
}
function fmtArr(a) {
  return Array.isArray(a) ? "[" + a.map((v) => fmt(v)).join(", ") + "]" : "—";
}

function setMsg(el, kind, text) {
  el.className = "msg " + kind;
  el.textContent = text;
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    data = {};
  }
  return { ok: res.ok, status: res.status, data };
}

/* ---------- 渲染 ---------- */
function render(s) {
  const rev = s.revision ?? 0;
  const logSize = s.log_size ?? 0;
  // 旧结果抑制：更旧的投影/更短的日志不得覆盖页面
  if (rev < shownRevision || logSize < shownLogSize) {
    return false;
  }
  shownRevision = rev;
  shownLogSize = logSize;

  $("rev").textContent = rev;
  $("anchor").textContent = s.anchor_seq === null || s.anchor_seq === undefined ? "—" : s.anchor_seq;
  $("logsize").textContent = logSize;

  if (s.config) {
    $("cfgState").innerHTML =
      `q=<code>${fmt(s.config.q)}</code> r=<code>${fmt(s.config.r)}</code> ` +
      `lag=<code>${fmt(s.config.lag, 2)}s</code> x0=<code>${fmtArr(s.config.x0)}</code>`;
  } else {
    $("cfgState").textContent = "尚未建立配置";
  }

  // 当前位置
  const cur = $("current");
  if (s.current) {
    const c = s.current;
    cur.innerHTML =
      `<dt>时间戳</dt><dd>${c.timestamp === null ? "（先验，尚无观测）" : fmt(c.timestamp, 3)}</dd>` +
      `<dt>位置 (px, py)</dt><dd>${fmtArr(c.position)}</dd>` +
      `<dt>速度 (vx, vy)</dt><dd>${fmtArr(c.velocity)}</dd>` +
      `<dt>协方差对角</dt><dd>${fmtArr(c.P_diag)}</dd>` +
      `<dt>最近残差</dt><dd>${fmtArr(c.residual)}</dd>`;
  }

  // 轨迹
  const tb = $("trackBody");
  if (s.track && s.track.points.length) {
    tb.innerHTML = s.track.points
      .map(
        (p) => `<tr${p.seq === s.anchor_seq ? ' style="color:var(--warn)"' : ""}>
          <td>${p.seq}${p.seq === s.anchor_seq ? " ★检查点" : ""}</td>
          <td>${fmt(p.timestamp, 3)}</td>
          <td>${fmtArr(p.state.slice(0, 2))}</td>
          <td>${fmtArr(p.state.slice(2, 4))}</td>
          <td>${fmtArr(p.P_diag)}</td>
          <td>${fmtArr(p.residual)}</td></tr>`
      )
      .join("");
  } else {
    tb.innerHTML = '<tr><td colspan="6" class="muted">尚无已发布轨迹</td></tr>';
  }

  // 日志（倒序展示，最新在上）
  const lb = $("logBody");
  if (!s.log.length) {
    lb.innerHTML = '<tr><td colspan="7" class="muted">无记录</td></tr>';
  } else {
    lb.innerHTML = s.log
      .slice()
      .reverse()
      .map(
        (e) => `<tr>
          <td>${e.seq}</td><td>${e.id}</td><td>${fmt(e.timestamp, 3)}</td>
          <td>(${fmt(e.x)}, ${fmt(e.y)})</td>
          <td><span class="badge ${e.decision}">${e.decision}</span></td>
          <td>${e.revision === null || e.revision === undefined ? "—" : e.revision}</td>
          <td class="wrap">${e.reason || ""}</td></tr>`
      )
      .join("");
  }
  return true;
}

async function refresh(allowStaleNote) {
  const reqId = ++lastRequestId;
  const res = await fetch("/api/state");
  const s = await res.json();
  if (reqId < maxAppliedId) return; // 已有更新的请求完成
  const applied = render(s);
  if (!applied && allowStaleNote) {
    setMsg($("obsMsg"), "info", "收到旧修订结果，已抑制，不覆盖当前页面。");
  }
  if (applied) maxAppliedId = reqId;
}

/* ---------- 事件 ---------- */
$("btnConfig").addEventListener("click", async () => {
  const body = {
    x0: [num("x0_0"), num("x0_1"), num("x0_2"), num("x0_3")],
    P0: readP0(),
    q: num("q"),
    r: num("r"),
    lag: num("lag"),
  };
  const { ok, data } = await postJson("/api/config", body);
  if (ok) {
    setMsg($("cfgMsg"), "ok", "配置已建立并通过正定/正值校验。");
    render(data);
  } else {
    setMsg($("cfgMsg"), "err", `配置被拒绝（${data.status || ""}）：${data.error || "非法配置"}；保留最近有效轨迹。`);
  }
});

$("btnReset").addEventListener("click", async () => {
  await postJson("/api/reset", {});
  shownRevision = -1;
  shownLogSize = -1;
  setMsg($("cfgMsg"), "info", "已重置日志与轨迹。");
  refresh();
});

$("btnObs").addEventListener("click", async () => {
  const body = {
    id: $("obsId").value.trim(),
    timestamp: num("obsT"),
    x: num("obsX"),
    y: num("obsY"),
  };
  const reqId = ++lastRequestId;
  const { ok, data } = await postJson("/api/observations", body);
  // 即使响应到达时已有更新请求完成，也尝试刷新；render 内部会抑制旧内容。
  if (reqId >= maxAppliedId) {
    const applied = render(await (await fetch("/api/state")).json());
    if (applied) maxAppliedId = reqId;
  }
  if (data.decision === "ACCEPTED") {
    setMsg($("obsMsg"), "ok", `接受：发布修订 ${data.track_revision}，当前位置 ${fmtArr(data.current ? data.current.position : null)}`);
  } else if (data.decision === "REPLAYED") {
    const receiptRev = data.receipt_revision === null || data.receipt_revision === undefined
      ? ""
      : `（首次回执修订 ${data.receipt_revision}）`;
    setMsg($("obsMsg"), "info", `重放：${data.reason}${receiptRev}；回执为首次接受证据，当前轨迹仍为修订 ${data.track_revision}`);
  } else {
    setMsg($("obsMsg"), "err", `拒绝：${data.reason || data.error || "未说明"}（已发布轨迹保持不变）`);
  }
  $("lastDecision").innerHTML = data.decision
    ? `<span class="badge ${data.decision}">${data.decision}</span> <span class="muted">${data.reason || data.error || ""}</span>`
    : "—";
});

buildP0Grid();
refresh();
