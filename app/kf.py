"""固定滞后平滑器（Fixed-Lag Smoother）核心。

状态向量 x = [px, py, vx, vy]，二维位置与匀速（CV）模型。

设计要点
--------
* 轨迹按观测的 *采样时刻* 排序重放；窗口 ``lag`` 内的迟到观测会修正
  历史后缀（自上一个检查点起重算），窗口外的观测被拒绝，绝不改写已封存位置。
* 观测日志不可变：每条观测进入日志后永不删除、不可覆盖；
  同标识同内容重放原结论，同标识不同内容拒绝。
* 修订号单调递增：每次成功纳入观测产生一个新修订；拒绝不产生新轨迹。
* 数值失败（非法噪声、创新矩阵奇异）抛出 :class:`KalmanError`，调用方保留
  最近一次有效轨迹。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import linalg as la

# ---------------------------------------------------------------------------
# 常量与异常
# ---------------------------------------------------------------------------

STATE_DIM = 4
OBS_DIM = 2
_EPS = 1e-12
_H = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
_I4 = la.identity(STATE_DIM)


class KalmanError(ValueError):
    """滤波无法继续：非法参数或不可逆创新矩阵。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """滤波器配置。

    初值 ``x0`` = [px, py, vx, vy]；``P0`` 为 4x4 初始协方差；
    ``q`` 为连续过程噪声功率谱密度（位置方差/时间），
    Q(dt) = q * [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]；
    ``r`` 为观测噪声方差（各向同性），R = r I2；``lag`` 为滞后窗口长度（秒）。
    """

    x0: Tuple[float, float, float, float]
    P0: List[List[float]]
    q: float
    r: float
    lag: float

    def validate(self) -> None:
        if not (math.isfinite(self.q) and self.q > 0):
            raise KalmanError("过程噪声 q 必须为正数")
        if not (math.isfinite(self.r) and self.r > 0):
            raise KalmanError("观测噪声 r 必须为正数")
        if not (math.isfinite(self.lag) and self.lag >= 0):
            raise KalmanError("滞后长度 lag 必须为非负数")
        if len(self.x0) != STATE_DIM or not la.is_finite_vector(self.x0):
            raise KalmanError("初值 x0 必须为 4 个有限数 [px,py,vx,vy]")
        P = self.P0
        if la.shape(P) != (STATE_DIM, STATE_DIM) or not la.is_finite_matrix(P):
            raise KalmanError("初始协方差 P0 必须为含有限数值的 4x4 矩阵")
        if not la.is_symmetric(P):
            raise KalmanError("初始协方差 P0 必须对称")
        if any(v <= 0.0 for v in la.diag(P)):
            raise KalmanError("初始协方差 P0 对角线必须为正")
        # Cholesky 是最严格的“合法噪声/协方差矩阵”检查：要求对称正定。
        try:
            la.cholesky(P)
        except la.LinAlgError:
            raise KalmanError("初始协方差 P0 必须对称正定") from None

    @property
    def P0_array(self) -> List[List[float]]:
        return la.copy_matrix(self.P0)

    def to_dict(self) -> dict:
        return {
            "x0": [float(v) for v in self.x0],
            "P0": [[float(v) for v in row] for row in self.P0],
            "q": float(self.q),
            "r": float(self.r),
            "lag": float(self.lag),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        return cls(
            x0=tuple(float(v) for v in d["x0"]),
            P0=[[float(v) for v in row] for row in d["P0"]],
            q=float(d["q"]),
            r=float(d["r"]),
            lag=float(d["lag"]),
        )


# ---------------------------------------------------------------------------
# 观测与轨迹点
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """一条定位观测：接收序号、稳定标识、采样时刻、二维位置。"""

    seq: int
    obs_id: str
    timestamp: float
    x: float
    y: float

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "id": self.obs_id,
            "timestamp": self.timestamp,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(
            seq=int(d["seq"]),
            obs_id=str(d["id"]),
            timestamp=float(d["timestamp"]),
            x=float(d["x"]),
            y=float(d["y"]),
        )


@dataclass
class TrackPoint:
    """重放后某观测时刻的滤波后状态。"""

    timestamp: float
    state: List[float]
    cov: List[List[float]]
    residual: List[float]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": [float(v) for v in self.state],
            "P_diag": [float(v) for v in la.diag(self.cov)],
            "residual": [float(v) for v in self.residual],
        }


# ---------------------------------------------------------------------------
# 线性卡尔曼滤波步骤（纯函数）
# ---------------------------------------------------------------------------


def _transition(dt: float) -> la.Matrix:
    F = la.identity(STATE_DIM)
    F[0][2] = dt
    F[1][3] = dt
    return F


def _process_noise(q: float, dt: float) -> la.Matrix:
    a = dt**3 / 3.0
    b = dt**2 / 2.0
    c = dt
    return [
        [q * a, 0.0, q * b, 0.0],
        [0.0, q * a, 0.0, q * b],
        [q * b, 0.0, q * c, 0.0],
        [0.0, q * b, 0.0, q * c],
    ]


def _kf_step(
    x: la.Vector,
    P: la.Matrix,
    z: la.Vector,
    dt: float,
    q: float,
    r: float,
) -> Tuple[la.Vector, la.Matrix, la.Vector]:
    """预测 + 更新，返回 (后验状态, 后验协方差, 创新残差)。

    创新协方差 S = H P_pred H^T + r I；若 S 不可逆或非正定抛 KalmanError。
    """
    F = _transition(dt)
    Q = _process_noise(q, dt)
    # x_pred = F x
    x_pred = la.matvec(F, x)
    # P_pred = F P F^T + Q
    P_pred = la.add(la.matmul(la.matmul(F, P), la.transpose(F)), Q)
    P_pred = la.symmetric_part(P_pred)

    # S = H P_pred H^T + r I2 （只取位置块）
    top = [P_pred[0][:2], P_pred[1][:2]]
    S = [[top[0][0] + r, top[0][1]], [top[1][0], top[1][1] + r]]
    S = la.symmetric_part(S)
    try:
        S_inv = la.inv2(S)
    except la.LinAlgError:
        raise KalmanError("创新矩阵 S 不可逆，无法更新") from None
    e1, _ = la.eigvalsh2(S)
    if e1 <= _EPS:
        raise KalmanError("创新矩阵 S 非正定，无法更新")

    # K = P_pred H^T S_inv；H^T = [[1,0],[0,1],[0,0],[0,0]]
    PHT = [
        [P_pred[0][0], P_pred[0][1]],
        [P_pred[1][0], P_pred[1][1]],
        [P_pred[2][0], P_pred[2][1]],
        [P_pred[3][0], P_pred[3][1]],
    ]
    K = la.matmul(PHT, S_inv)  # 4x2
    innov = [z[0] - x_pred[0], z[1] - x_pred[1]]
    x_upd = [x_pred[i] + K[i][0] * innov[0] + K[i][1] * innov[1] for i in range(STATE_DIM)]
    # P_upd = (I - K H) P_pred
    KH = la.matmul(K, _H)  # 4x4
    IKH = la.sub(_I4, KH)
    P_upd = la.matmul(IKH, P_pred)
    P_upd = la.symmetric_part(P_upd)
    return x_upd, P_upd, innov


# ---------------------------------------------------------------------------
# 固定滞后重放
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """一次成功纳入后的完整投影结果。"""

    revision: int
    points: List[TrackPoint]
    accepted_ids: List[str]  # 参与本次重放的观测 id（按采样时刻排序）
    anchor_seq: int  # 检查点观测的接收序号；-1 表示从初始先验重放


class FixedLagSmoother:
    """持有不可变配置，对一组已接受观测执行“检查点 + 后缀重放”。

    本类无内部可变状态：轨迹完全由配置与已接受观测集合决定，便于持久化恢复。
    """

    def __init__(self, config: FilterConfig):
        config.validate()
        self.config = config

    def replay(self, accepted_obs: Sequence[Observation], revision: int) -> ReplayResult:
        """按 (采样时刻, 接收序号) 排序重放全部已接受观测。

        选取严格封存边界（timestamp <= t_last - lag）上的最后一个观测作为
        检查点：检查点之前（含）的历史为已封存位置，重放必须原样复现它们；
        仅检查点之后的 *后缀* 被重新计算。无检查点时从初始先验重放。
        """
        cfg = self.config
        ordered = sorted(accepted_obs, key=lambda o: (o.timestamp, o.seq))

        points: List[TrackPoint] = []
        anchor_seq = -1

        if ordered:
            t_last = ordered[-1].timestamp
            anchor_idx = -1
            if cfg.lag > 0.0:
                boundary = t_last - cfg.lag
                for i in range(len(ordered) - 1, -1, -1):
                    if ordered[i].timestamp <= boundary + _EPS:
                        anchor_idx = i
                        break

            if anchor_idx >= 0:
                anchor_seq = ordered[anchor_idx].seq
                # 封存前缀：确定性重放，结果与既有发布逐位一致（不改写封存位置）。
                x, P, prefix_points = self._run(ordered[: anchor_idx + 1])
                suffix = ordered[anchor_idx + 1 :]
                _, _, suffix_points = self._run(suffix, x, P, prefix_points)
                points = prefix_points + suffix_points
            else:
                _, _, points = self._run(ordered)

        return ReplayResult(
            revision=revision,
            points=points,
            accepted_ids=[o.obs_id for o in ordered],
            anchor_seq=anchor_seq,
        )

    # -- 内部 -------------------------------------------------------------

    def _run(
        self,
        ordered: Sequence[Observation],
        x0: Optional[la.Vector] = None,
        P0: Optional[la.Matrix] = None,
        preceding: Optional[List[TrackPoint]] = None,
    ) -> Tuple[la.Vector, la.Matrix, List[TrackPoint]]:
        """从给定先验（默认配置先验）顺序执行 KF 步骤。"""
        cfg = self.config
        x = [float(v) for v in cfg.x0] if x0 is None else list(x0)
        P = cfg.P0_array if P0 is None else la.copy_matrix(P0)
        points: List[TrackPoint] = []
        last_t: Optional[float] = None
        if preceding:
            last_t = preceding[-1].timestamp
        for obs in ordered:
            dt = 0.0 if last_t is None else obs.timestamp - last_t
            z = [obs.x, obs.y]
            x, P, residual = _kf_step(x, P, z, dt, cfg.q, cfg.r)
            points.append(TrackPoint(obs.timestamp, list(x), la.copy_matrix(P), list(residual)))
            last_t = obs.timestamp
        return x, P, points
