"""导航复核状态存储。

职责
----
* 不可变观测日志：所有提交（接受 / 重放 / 拒绝）只追加、不修改、不删除，
  最多 32 条。
* 单调修订号：仅当一次纳入成功发布新轨迹时修订号 +1；任何拒绝/重放都不改变它。
* 固定滞后窗口规则：迟到观测只有落在 ``最新采样时刻 - lag`` 之内才会被接受，
  窗口外、同标识内容冲突、顺序违规等一律拒绝且保持已发布轨迹不变。
* 旧结果抑制：计算结果带“代际（generation）”令牌，提交时若已有更新的代际，
  旧结果一律丢弃，绝不覆盖最新页面。
* 持久化：每次提交原子写入 JSON 文件（tmp + os.replace）；重开后原样恢复。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import linalg as la
from .kf import (
    FilterConfig,
    FixedLagSmoother,
    KalmanError,
    Observation,
    STATE_DIM,
)

MAX_OBSERVATIONS = 32
ACCEPTED = "ACCEPTED"
REPLAYED = "REPLAYED"
REJECTED = "REJECTED"
_EPS = 1e-9


class StoreError(ValueError):
    """请求本身非法（字段缺失/格式错误）。"""


@dataclass
class Track:
    """已发布轨迹（某个修订号下的不可变投影）。"""

    revision: int
    anchor_seq: int
    points: List[Dict[str, Any]]  # 每个点含 seq/timestamp/state/P_diag/residual

    def to_dict(self) -> Dict[str, Any]:
        return {"revision": self.revision, "anchor_seq": self.anchor_seq, "points": self.points}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["Track"]:
        if d is None:
            return None
        return cls(revision=int(d["revision"]), anchor_seq=int(d["anchor_seq"]), points=list(d["points"]))


class Store:
    """线程安全的应用状态容器。"""

    def __init__(self, path: Optional[str] = None, config: Optional[FilterConfig] = None):
        self._lock = threading.RLock()
        self._path = path
        self._config: Optional[FilterConfig] = None
        self._smoother: Optional[FixedLagSmoother] = None
        self._log: List[Dict[str, Any]] = []
        self._next_seq = 0
        self._revision = 0
        self._track: Optional[Track] = None
        self._generation = 0  # 计算代际；每次发起纳入 +1，用于旧结果抑制
        # 测试钩子：重放前调用（例如注入延迟以制造并发交错）
        self.before_replay_hook = None

        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            self._load()
        elif config is not None:
            self.initialize_config(config)

    # ------------------------------------------------------------------ #
    # 配置
    # ------------------------------------------------------------------ #

    def initialize_config(self, config: FilterConfig) -> Dict[str, Any]:
        """建立初值/噪声/滞后。配置非法抛 KalmanError；日志非空时拒绝重建。"""
        with self._lock:
            config.validate()
            if self._log:
                raise StoreError("已有观测日志，不能重建配置（可先重置）")
            self._config = config
            self._smoother = FixedLagSmoother(config)
            self._revision = 0
            self._track = None
            self._persist_locked()
            return self.state_locked()

    def reset(self) -> Dict[str, Any]:
        """清空日志与轨迹（显式操作，持久化同步清空）。"""
        with self._lock:
            self._log = []
            self._next_seq = 0
            self._revision = 0
            self._track = None
            self._generation += 1
            self._persist_locked()
            return self.state_locked()

    @property
    def config(self) -> Optional[FilterConfig]:
        return self._config

    # ------------------------------------------------------------------ #
    # 观测录入
    # ------------------------------------------------------------------ #

    def submit_observation(self, payload: Any) -> Dict[str, Any]:
        """录入一条观测，返回该条结论（接受/重放/拒绝 + 当前位置等）。

        并发模型：所有 *校验与日志追加* 在锁内瞬时完成；唯一耗时的重放计算
        在锁外执行。计算期间允许继续录入其它观测；计算结果回交时做 CAS：
        若出发时的输入代际已过期，则旧结果丢弃（不覆盖最新页面），并用最新
        状态重新校验/重放后再提交。
        """
        fields = self._validate_payload(payload)
        # 重放为纯 CPU 计算且极快，CAS 最多重试少量次数即可收敛
        reserved_seq: Optional[int] = None
        for _attempt in range(MAX_OBSERVATIONS + 1):
            with self._lock:
                quick = self._quick_decide_locked(fields, reserved_seq)
                if quick is not None:
                    action, data = quick
                    if action == "response":
                        return data
                    # action == "compute"：记录快照后到锁外计算
                    accepted_snapshot = [
                        Observation(
                            seq=o.seq,
                            obs_id=o.obs_id,
                            timestamp=o.timestamp,
                            x=o.x,
                            y=o.y,
                        )
                        for o in data["accepted"]
                    ]
                    new_obs = data["new_obs"]
                    reserved_seq = new_obs.seq  # 重试时保持首次接收序号
                    base_revision = data["base_revision"]
                    base_next_seq = data["base_next_seq"]
                    base_generation = self._generation

            # ---- 锁外：耗时重放（不阻塞其它录入） ----
            if self.before_replay_hook is not None:
                self.before_replay_hook(new_obs)
            compute_error: Optional[str] = None
            try:
                result = self._smoother.replay(
                    accepted_snapshot + [new_obs], base_revision + 1
                )
            except KalmanError as exc:
                compute_error = str(exc)
                result = None

            with self._lock:
                # 旧结果抑制：输入快照过期则丢弃本次结果并重试整轮
                if base_generation != self._generation or base_next_seq != self._next_seq:
                    continue
                if compute_error is not None:
                    return self._rejection_locked(
                        fields,
                        append=True,
                        reason=f"重放失败：{compute_error}；保留最近有效轨迹",
                        obs_seq=new_obs.seq,
                    )
                # 提交新轨迹（单调修订号：恰好 +1）
                assert self._revision == base_revision
                assert result is not None
                self._track = self._build_track(result, new_obs)
                self._revision = base_revision + 1
                point = next(p for p in self._track.points if p["seq"] == new_obs.seq)
                entry = {
                    **new_obs.to_dict(),
                    "decision": ACCEPTED,
                    "reason": "纳入并自检查点重放后缀成功",
                    "revision": self._revision,
                    "snapshot": self._snapshot_of(point),
                }
                self._log.append(entry)
                self._synchronize_accepted_snapshots_locked()
                self._generation += 1
                self._persist_locked()
                return self._decision_response_locked(entry)
        # 理论不可达：持续被更新的修订打断
        with self._lock:
            return self._rejection_locked(
                fields, append=False, reason="系统繁忙：计算结果持续被更新修订抑制，请重试"
            )

    def _quick_decide_locked(
        self, fields: Dict[str, Any], reserved_seq: Optional[int]
    ) -> Optional[Tuple[str, Any]]:
        """锁内快速判定。返回 ("response", 直接响应) 或 ("compute", 计算计划)。

        ``reserved_seq`` 为 CAS 重试时保留的首次接收序号；None 表示首次进入。
        """

        def reject(reason: str) -> Tuple[str, Any]:
            return (
                "response",
                self._rejection_locked(
                    fields, append=True, reason=reason, obs_seq=reserved_seq
                ),
            )

        if len(self._log) >= MAX_OBSERVATIONS:
            return (
                "response",
                self._rejection_locked(
                    fields, append=False, reason=f"观测日志已满（最多 {MAX_OBSERVATIONS} 条）"
                ),
            )

        # 1) 同标识：同内容回放原结论，内容不同拒绝
        prior = next((e for e in self._log if e["id"] == fields["id"]), None)
        if prior is not None:
            same = (
                prior["timestamp"] == fields["timestamp"]
                and prior["x"] == fields["x"]
                and prior["y"] == fields["y"]
            )
            if same:
                if prior["decision"] == ACCEPTED or prior["decision"] == REPLAYED:
                    return ("response", self._replay_locked(prior, fields))
                return reject("同标识同内容曾被拒绝，维持原拒绝结论")
            return reject("同标识观测内容不同，拒绝（稳定标识不可复用）")

        # 2) 必须先建立配置
        if self._smoother is None or self._config is None:
            return reject("尚未建立滤波器配置")

        # 3) 同刻非递增顺序：相同采样时刻的已接受观测会造成无法严格排序
        if any(
            abs(e["timestamp"] - fields["timestamp"]) <= _EPS
            for e in self._log
            if e["decision"] == ACCEPTED
        ):
            return reject("同一采样时刻已有已接受观测（同刻非递增顺序），拒绝，已发布轨迹不变")

        # 4) 固定滞后窗口：迟到观测不得超出 lag
        accepted = self._accepted_obs_locked()
        t_max = max((o.timestamp for o in accepted), default=-math.inf)
        if math.isfinite(t_max) and t_max - fields["timestamp"] > self._config.lag + _EPS:
            return reject(
                f"观测落后最新时刻 {t_max:.6g} 已超过滞后窗口 {self._config.lag:g}s，"
                "属窗口外观测，不得改写已封存轨迹"
            )

        if reserved_seq is None:
            reserved_seq = self._next_seq
            self._next_seq += 1
        new_obs = Observation(
            seq=reserved_seq,
            obs_id=fields["id"],
            timestamp=fields["timestamp"],
            x=fields["x"],
            y=fields["y"],
        )
        return (
            "compute",
            {
                "accepted": accepted,
                "new_obs": new_obs,
                "base_revision": self._revision,
                "base_next_seq": self._next_seq,
            },
        )

    # ------------------------------------------------------------------ #
    # 轨迹构造（提交）
    # ------------------------------------------------------------------ #

    def _snapshot_of(self, point: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": point["timestamp"],
            "state": list(point["state"]),
            "P_diag": list(point["P_diag"]),
            "residual": list(point["residual"]),
            "projection_revision": self._revision,
        }

    def _synchronize_accepted_snapshots_locked(self) -> None:
        if self._track is None:
            return
        points_by_seq = {point["seq"]: point for point in self._track.points}
        for entry in self._log:
            if entry["decision"] != ACCEPTED:
                continue
            point = points_by_seq.get(entry["seq"])
            if point is None:
                continue
            entry["snapshot"] = self._snapshot_of(point)

    def _build_track(self, result: Any, new_obs: Observation) -> Track:
        """把重放结果与观测序号对齐，构造可发布轨迹。"""
        accepted_sorted = sorted(
            self._accepted_obs_locked() + [new_obs], key=lambda o: (o.timestamp, o.seq)
        )
        points: List[Dict[str, Any]] = []
        for obs, p in zip(accepted_sorted, result.points):
            points.append(
                {
                    "seq": obs.seq,
                    "timestamp": p.timestamp,
                    "state": [float(v) for v in p.state],
                    "P_diag": [float(v) for v in la.diag(p.cov)],
                    "residual": [float(v) for v in p.residual],
                }
            )
        return Track(revision=result.revision, anchor_seq=result.anchor_seq, points=points)

    # ------------------------------------------------------------------ #
    # 结论构造
    # ------------------------------------------------------------------ #

    def _replay_locked(self, original: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
        entry = {
            "seq": self._next_seq,
            "id": fields["id"],
            "timestamp": fields["timestamp"],
            "x": fields["x"],
            "y": fields["y"],
            "decision": REPLAYED,
            "reason": f"同标识同内容回放原结论（首次见 seq={original['seq']}，修订 {original['revision']}）",
            "revision": None,
            "snapshot": original["snapshot"],
            "replayed_from_seq": original["seq"],
        }
        self._next_seq += 1
        self._log.append(entry)
        self._persist_locked()
        return self._decision_response_locked(entry)

    def _rejection_locked(
        self,
        fields: Optional[Dict[str, Any]],
        append: bool,
        reason: str,
        obs_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        entry = None
        if append and fields is not None:
            entry = {
                "seq": obs_seq if obs_seq is not None else self._next_seq,
                "id": fields["id"],
                "timestamp": fields["timestamp"],
                "x": fields["x"],
                "y": fields["y"],
                "decision": REJECTED,
                "reason": reason,
                "revision": None,
                "snapshot": None,
            }
            if obs_seq is None:
                self._next_seq += 1
            self._log.append(entry)
            self._persist_locked()
        resp = self._decision_response_locked(entry) if entry is not None else {}
        if fields is not None:
            resp.update(
                {
                    "id": fields["id"],
                    "timestamp": fields["timestamp"],
                    "x": fields["x"],
                    "y": fields["y"],
                }
            )
        resp.update({"decision": REJECTED, "reason": reason, "seq": obs_seq})
        return resp

    def _decision_response_locked(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        state = self.state_locked()
        return {
            "seq": entry["seq"],
            "id": entry["id"],
            "timestamp": entry["timestamp"],
            "x": entry["x"],
            "y": entry["y"],
            "decision": entry["decision"],
            "reason": entry["reason"],
            "revision": entry.get("revision"),
            "snapshot": entry.get("snapshot"),
            "current": state["current"],
            "track_revision": state["revision"],
            "log_size": len(self._log),
        }

    # ------------------------------------------------------------------ #
    # 状态视图
    # ------------------------------------------------------------------ #

    def _accepted_obs_locked(self) -> List[Observation]:
        return [
            Observation(
                seq=e["seq"],
                obs_id=e["id"],
                timestamp=e["timestamp"],
                x=e["x"],
                y=e["y"],
            )
            for e in self._log
            if e["decision"] == ACCEPTED
        ]

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return self.state_locked()

    def state_locked(self) -> Dict[str, Any]:
        if self._track and self._track.points:
            last = self._track.points[-1]
            current = {
                "timestamp": last["timestamp"],
                "position": [last["state"][0], last["state"][1]],
                "velocity": [last["state"][2], last["state"][3]],
                "P_diag": last["P_diag"],
                "residual": last["residual"],
            }
        elif self._config is not None:
            current = {
                "timestamp": None,
                "position": [self._config.x0[0], self._config.x0[1]],
                "velocity": [self._config.x0[2], self._config.x0[3]],
                "P_diag": [float(self._config.P0[i][i]) for i in range(STATE_DIM)],
                "residual": [0.0, 0.0],
            }
        else:
            current = None
        return {
            "config": self._config.to_dict() if self._config else None,
            "revision": self._revision,
            "anchor_seq": self._track.anchor_seq if self._track else None,
            "log_size": len(self._log),
            "max_observations": MAX_OBSERVATIONS,
            "current": current,
            "log": list(self._log),
            "track": self._track.to_dict() if self._track else None,
        }

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_payload(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise StoreError("请求体必须为 JSON 对象")
        obs_id = payload.get("id")
        if not isinstance(obs_id, str) or not obs_id.strip():
            raise StoreError("观测标识 id 必须为非空字符串")
        try:
            ts = float(payload["timestamp"])
            x = float(payload["x"])
            y = float(payload["y"])
        except (KeyError, TypeError, ValueError):
            raise StoreError("timestamp/x/y 必须为有限数值") from None
        for name, val in (("timestamp", ts), ("x", x), ("y", y)):
            if not math.isfinite(val):
                raise StoreError(f"{name} 必须为有限数值")
        return {"id": obs_id.strip(), "timestamp": ts, "x": x, "y": y}

    @staticmethod
    def parse_config(payload: Any) -> FilterConfig:
        """把 HTTP 配置请求解析为 FilterConfig（非法时抛 StoreError/KalmanError）。"""
        if not isinstance(payload, dict):
            raise StoreError("配置必须为 JSON 对象")
        try:
            x0 = payload["x0"]
            P0 = payload["P0"]
            q = float(payload["q"])
            r = float(payload["r"])
            lag = float(payload["lag"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreError("缺少或非法的配置字段：x0/P0/q/r/lag") from exc
        if not isinstance(x0, list) or len(x0) != 4:
            raise StoreError("x0 必须为 4 元素数组 [px,py,vx,vy]")
        if not isinstance(P0, list) or len(P0) != 4 or any(
            not isinstance(row, list) or len(row) != 4 for row in P0
        ):
            raise StoreError("P0 必须为 4x4 矩阵")
        try:
            x0t = tuple(float(v) for v in x0)
            P0l = [[float(v) for v in row] for row in P0]
        except (TypeError, ValueError) as exc:
            raise StoreError("x0/P0 含非数值元素") from exc
        return FilterConfig(x0=x0t, P0=P0l, q=q, r=r, lag=lag)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #

    def _persist_locked(self) -> None:
        if not self._path:
            return
        data = {
            "version": 1,
            "config": self._config.to_dict() if self._config else None,
            "next_seq": self._next_seq,
            "revision": self._revision,
            "log": self._log,
            "track": self._track.to_dict() if self._track else None,
        }
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("config"):
            cfg = FilterConfig.from_dict(data["config"])
            cfg.validate()
            self._config = cfg
            self._smoother = FixedLagSmoother(cfg)
        self._log = list(data.get("log", []))
        self._next_seq = int(data.get("next_seq", len(self._log)))
        self._revision = int(data.get("revision", 0))
        self._track = Track.from_dict(data.get("track"))
