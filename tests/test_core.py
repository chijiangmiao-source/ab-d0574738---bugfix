"""核心规则测试：重放、窗口、修订、持久化与旧结果抑制。"""

import math
import os
import tempfile
import threading
import unittest

from app.kf import FilterConfig, FixedLagSmoother, KalmanError, Observation
from app import linalg as la
from app.store import (
    ACCEPTED,
    REJECTED,
    REPLAYED,
    MAX_OBSERVATIONS,
    Store,
)


def cfg(lag=2.0, q=0.5, r=1.0, p0=None):
    return FilterConfig(
        x0=(0.0, 0.0, 1.0, 0.0),
        P0=p0 or [[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]],
        q=q,
        r=r,
        lag=lag,
    )


def obs(seq, oid, t, x, y):
    return Observation(seq=seq, obs_id=oid, timestamp=float(t), x=x, y=y)


class TestConfigValidation(unittest.TestCase):
    def test_valid(self):
        cfg().validate()

    def test_negative_noise(self):
        with self.assertRaises(KalmanError):
            cfg(q=-1).validate()
        with self.assertRaises(KalmanError):
            cfg(r=0).validate()

    def test_bad_lag(self):
        with self.assertRaises(KalmanError):
            cfg(lag=-0.1).validate()

    def test_nonsymmetric_p0(self):
        p0 = [[10, 1, 0, 0], [0, 10, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]
        with self.assertRaises(KalmanError):
            cfg(p0=p0).validate()

    def test_indefinite_p0(self):
        p0 = [[-1, 0, 0, 0], [0, 10, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]
        with self.assertRaises(KalmanError):
            cfg(p0=p0).validate()

    def test_singular_inv2(self):
        with self.assertRaises(la.LinAlgError):
            la.inv2([[1.0, 2.0], [2.0, 4.0]])


class TestSmokeFilter(unittest.TestCase):
    def test_observation_reduces_uncertainty(self):
        s = FixedLagSmoother(cfg())
        r = s.replay([obs(0, "a", 1.0, 1.1, 0.2)], 1)
        self.assertEqual(len(r.points), 1)
        self.assertLess(r.points[0].cov[0][0], 10.0 + 1.0)  # 更新后位置方差下降
        self.assertTrue(la.is_symmetric(r.points[0].cov))

    def test_singular_innovation_raises_and_is_explained(self):
        # r=0 且位置先验方差≈0 时，创新矩阵 S 奇异：必须抛 KalmanError
        # （正常配置 r>0 不可触发，此处直接验证数值防护与失败语义）。
        s = FixedLagSmoother(cfg(r=1.0))
        zero_p0 = [[1e-30, 0, 0, 0], [0, 1e-30, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4]]
        object.__setattr__(s.config, "r", 0.0)
        object.__setattr__(s.config, "P0", zero_p0)
        with self.assertRaises(KalmanError):
            s.replay([obs(0, "a", 1.0, 1.0, 0.0)], 1)

    def test_invalid_config_via_store_keeps_state(self):
        store = Store(config=cfg())
        store.submit_observation({"id": "A", "timestamp": 1.0, "x": 1.0, "y": 0.0})
        before = store.state()
        bad = FilterConfig(
            x0=(0, 0, 0, 0),
            P0=[[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            q=0.5,
            r=1.0,
            lag=2.0,
        )
        with self.assertRaises(KalmanError):
            store.initialize_config(bad)
        self.assertEqual(store.state()["track"], before["track"])


class TestStoreRules(unittest.TestCase):
    def setUp(self):
        self.store = Store(config=cfg(lag=2.0))

    def submit(self, oid, t, x, y):
        return self.store.submit_observation(
            {"id": oid, "timestamp": t, "x": x, "y": y}
        )

    def test_basic_accept_monotonic_revision(self):
        r1 = self.submit("A", 1.0, 1.1, 0.2)
        r2 = self.submit("B", 2.0, 2.0, 0.1)
        self.assertEqual(r1["decision"], ACCEPTED)
        self.assertEqual(r2["decision"], ACCEPTED)
        self.assertEqual(r1["track_revision"], 1)
        self.assertEqual(r2["track_revision"], 2)
        st = self.store.state()
        self.assertEqual(len(st["track"]["points"]), 2)
        # 速度先验 vx=1：滤波位置应接近匀速假设
        self.assertGreater(st["current"]["position"][0], 0.5)

    def test_late_within_window_recomputes_suffix_keeps_anchor(self):
        self.submit("A", 1.0, 1.0, 0.0)
        self.submit("B", 2.0, 2.0, 0.0)
        self.submit("C", 3.0, 3.0, 0.0)
        before = {
            p["timestamp"]: list(p["state"])
            for p in self.store.state()["track"]["points"]
        }
        # 迟到观测 t=1.5，lag=2，tmax=3 -> 3-1.5=1.5 <= 2，窗口内，接受
        late = self.submit("LATE", 1.5, 1.2, 0.5)
        self.assertEqual(late["decision"], ACCEPTED)
        self.assertEqual(late["track_revision"], 4)
        st = self.store.state()
        pts = st["track"]["points"]
        self.assertEqual([p["timestamp"] for p in pts], [1.0, 1.5, 2.0, 3.0])
        # 检查点（t=1，seq=0）为封存位置：状态必须与此前逐位一致
        self.assertEqual(st["anchor_seq"], 0)
        self.assertEqual(pts[0]["state"], before[1.0])
        # 后缀（t=2,3）因迟到观测被修正：状态必须发生变化
        self.assertNotEqual(
            [p for p in pts if p["timestamp"] == 2.0][0]["state"], before[2.0]
        )
        self.assertNotEqual(
            [p for p in pts if p["timestamp"] == 3.0][0]["state"], before[3.0]
        )

    def test_out_of_window_rejected_keeps_published_track(self):
        self.submit("A", 1.0, 1.0, 0.0)
        self.submit("B", 2.0, 2.0, 0.0)
        r3 = self.submit("C", 3.0, 3.0, 0.0)
        snapshot_before = self.store.state()["track"]
        # lag=2：t=0.5 距最新 3.0 为 2.5 > 2 -> 窗口外，拒绝
        rej = self.submit("OLD", 0.5, 9.0, 9.0)
        self.assertEqual(rej["decision"], REJECTED)
        self.assertIn("窗口", rej["reason"])
        st = self.store.state()
        self.assertEqual(st["revision"], 3)  # 修订号不变
        self.assertEqual(st["track"], snapshot_before)  # 已发布轨迹逐字节不变

    def test_replay_same_id_same_content(self):
        first = self.submit("A", 1.0, 1.1, 0.2)
        rev_after_first = self.store.state()["revision"]
        again = self.submit("A", 1.0, 1.1, 0.2)
        self.assertEqual(again["decision"], REPLAYED)
        self.assertEqual(again["track_revision"], rev_after_first)
        # 回放原结论：快照位置与首次接受一致
        self.assertEqual(
            [round(v, 12) for v in again["snapshot"]["state"]],
            [round(v, 12) for v in first["snapshot"]["state"]],
        )
        self.assertEqual(self.store.state()["revision"], rev_after_first)

    def test_same_id_different_content_rejected(self):
        self.submit("A", 1.0, 1.1, 0.2)
        rej = self.submit("A", 1.0, 9.9, 0.2)
        self.assertEqual(rej["decision"], REJECTED)
        rej2 = self.submit("A", 2.0, 1.1, 0.2)
        self.assertEqual(rej2["decision"], REJECTED)

    def test_same_timestamp_non_increasing_rejected(self):
        self.submit("A", 1.0, 1.0, 0.0)
        rej = self.submit("B", 1.0, 1.05, 0.0)
        self.assertEqual(rej["decision"], REJECTED)
        self.assertIn("同刻", rej["reason"])
        self.assertEqual(self.store.state()["revision"], 1)

    def test_log_is_immutable_append_only(self):
        self.submit("A", 1.0, 1.0, 0.0)
        self.submit("B", 9.0, 9.0, 9.0)  # 窗口外拒绝
        log0 = self.store.state()["log"][0]
        self.submit("A", 1.0, 1.0, 0.0)  # 重放
        log = self.store.state()["log"]
        self.assertEqual(log[0], log0)  # 首条永不被改写
        self.assertEqual([e["seq"] for e in log], [0, 1, 2])

    def test_max_32_observations(self):
        for i in range(MAX_OBSERVATIONS):
            r = self.submit(f"O{i}", float(i + 1), float(i + 1), 0.0)
            self.assertEqual(r["decision"], ACCEPTED)
        overflow = self.submit("X", 40.0, 1.0, 1.0)
        self.assertEqual(overflow["decision"], REJECTED)
        self.assertIn("32", overflow["reason"])
        self.assertEqual(len(self.store.state()["log"]), MAX_OBSERVATIONS)

    def test_residuals_returned(self):
        r = self.submit("A", 1.0, 5.0, 5.0)
        self.assertEqual(len(r["snapshot"]["residual"]), 2)


class TestPersistence(unittest.TestCase):
    def test_reopen_restores_log_and_track(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            s1 = Store(path=path, config=cfg(lag=5.0))
            s1.submit_observation({"id": "A", "timestamp": 1.0, "x": 1.1, "y": 0.2})
            s1.submit_observation({"id": "B", "timestamp": 2.0, "x": 2.3, "y": -0.1})
            s1.submit_observation({"id": "L", "timestamp": 1.4, "x": 1.2, "y": 0.3})
            s1.submit_observation({"id": "A", "timestamp": 1.0, "x": 1.1, "y": 0.2})  # 重放
            s1.submit_observation({"id": "OLD", "timestamp": -9, "x": 0, "y": 0})  # 窗口外拒绝
            before = s1.state()

            # “重开”：新实例从持久化存储恢复
            s2 = Store(path=path)
            after = s2.state()
            self.assertEqual(after["revision"], before["revision"])
            self.assertEqual(after["log"], before["log"])
            self.assertEqual(after["track"], before["track"])
            self.assertEqual(after["config"], before["config"])
            # 恢复后修订号继续单调，重放仍命中
            replay = s2.submit_observation({"id": "B", "timestamp": 2.0, "x": 2.3, "y": -0.1})
            self.assertEqual(replay["decision"], REPLAYED)
        finally:
            os.unlink(path)


class TestStaleResultSuppression(unittest.TestCase):
    def test_old_computation_does_not_overwrite_newest(self):
        store = Store(config=cfg(lag=5.0))
        released = threading.Event()
        proceed = threading.Event()

        def hook(new_obs):
            # 仅在第一条观测的重放期间挂起，并插入一条并发完成的观测
            if new_obs.obs_id == "A":
                released.set()
                proceed.wait(5)

        store.before_replay_hook = hook
        results = {}

        def first():
            results["A"] = store.submit_observation(
                {"id": "A", "timestamp": 1.0, "x": 1.0, "y": 0.0}
            )

        t = threading.Thread(target=first)
        t.start()
        self.assertTrue(released.wait(5))
        # 计算继续录入：B 在 A 的重放进行中完整提交
        results["B"] = store.submit_observation(
            {"id": "B", "timestamp": 2.0, "x": 2.0, "y": 0.0}
        )
        self.assertEqual(results["B"]["decision"], ACCEPTED)
        proceed.set()
        t.join(5)
        self.assertFalse(t.is_alive())

        st = store.state()
        # A 的旧结果被抑制并重试：最终两条观测都在，修订连续，无覆盖/回退
        self.assertEqual(results["A"]["decision"], ACCEPTED)
        self.assertEqual(st["revision"], 2)
        ids = {p["seq"] for p in st["track"]["points"]}
        self.assertEqual(len(ids), 2)
        seqs = sorted(e["seq"] for e in st["log"] if e["decision"] == ACCEPTED)
        self.assertEqual(seqs, [0, 1])


if __name__ == "__main__":
    unittest.main()
