"""toio + ATOM Mate for toio の 2D シミュレータ。

モデル化しているもの:
- 差動二輪の運動学 + 一次遅れのモーター応答 + 不感帯 + 速度飽和 + スリップ
- VL53L0X ToF センサー (視野角内の複数レイの最短距離, ノイズ・欠測・誤検出)
- Position ID による位置・角度計測 (ノイズ・読み取り失敗)
- Colab ⇔ MQTT ⇔ ATOM ⇔ BLE ⇔ toio の通信遅延・ジッタ・パケットロス
- ファームウェアの指令タイムアウト停止と ToF 安全停止 (core.apply_safety_shield)

障害物は円柱と回転した直方体 (上から見て円と四角形) で表す。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import EnvConfig
from .core import apply_safety_shield

TOF_PHYSICAL_MAX_MM = 1200.0  # VL53L0X (デフォルトモード) の実用的な最大距離


# ---------------------------------------------------------------- 障害物
@dataclass
class Obstacles:
    circles: np.ndarray  # (C, 3) [cx, cy, r]
    boxes: np.ndarray  # (B, 4, 2) 四隅の座標 (周回順)

    def __post_init__(self):
        self.circles = np.asarray(self.circles, dtype=np.float64).reshape(-1, 3)
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4, 2)
        self.seg_a = self.boxes.reshape(-1, 2)
        self.seg_b = np.roll(self.boxes, -1, axis=1).reshape(-1, 2)

    @staticmethod
    def empty() -> "Obstacles":
        return Obstacles(np.zeros((0, 3)), np.zeros((0, 4, 2)))

    @staticmethod
    def make_box(cx: float, cy: float, w: float, h: float, angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        local = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) * 0.5
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([cx, cy])


def raycast(origin: np.ndarray, angles: np.ndarray, obs: Obstacles, max_range: float) -> np.ndarray:
    """各レイが最初に障害物に当たる距離 (当たらなければ max_range)。"""
    d = np.column_stack([np.cos(angles), np.sin(angles)])  # (R, 2)
    best = np.full(len(angles), max_range)

    if len(obs.circles):
        oc = origin[None, :] - obs.circles[:, :2]  # (C, 2)
        b = d @ oc.T  # (R, C)
        c = (oc**2).sum(axis=1) - obs.circles[:, 2] ** 2  # (C,)
        disc = b**2 - c[None, :]
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        t = np.where(c[None, :] < 0, 0.0, t)  # センサーが円の内側
        t = np.where((disc >= 0) & (t >= 0), t, np.inf)
        best = np.minimum(best, t.min(axis=1))

    if len(obs.seg_a):
        e = obs.seg_b - obs.seg_a  # (S, 2)
        diff = obs.seg_a - origin[None, :]  # (S, 2)
        denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # (R, S)
        num_t = diff[:, 0] * e[:, 1] - diff[:, 1] * e[:, 0]  # (S,)
        num_s = diff[None, :, 0] * d[:, 1:2] - diff[None, :, 1] * d[:, 0:1]  # (R, S)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num_t[None, :] / denom
            s = num_s / denom
        ok = (np.abs(denom) > 1e-9) & (t >= 0) & (s >= 0) & (s <= 1)
        best = np.minimum(best, np.where(ok, t, np.inf).min(axis=1))
    return best


def clearance(p: np.ndarray, obs: Obstacles) -> float:
    """点 p から最も近い障害物表面までの距離 (内部なら負)。"""
    best = math.inf
    if len(obs.circles):
        best = float((np.linalg.norm(obs.circles[:, :2] - p, axis=1) - obs.circles[:, 2]).min())
    if len(obs.seg_a):
        e = obs.seg_b - obs.seg_a
        ap = p - obs.seg_a
        u = np.clip((ap * e).sum(axis=1) / (e * e).sum(axis=1), 0.0, 1.0)
        dist = float(np.linalg.norm(ap - u[:, None] * e, axis=1).min())
        # 凸四角形の内側判定: すべての辺に対して同じ側にあれば内側
        cross = (e[:, 0] * ap[:, 1] - e[:, 1] * ap[:, 0]).reshape(-1, 4)
        inside = np.any(np.all(cross > 0, axis=1) | np.all(cross < 0, axis=1))
        best = min(best, -dist if inside else dist)
    return best


# ---------------------------------------------------------------- パラメータ
@dataclass
class SimParams:
    """1 エピソード分の物理・センサー・通信パラメータ (DomainRandomization からサンプリング)。"""

    gain_left: float
    gain_right: float
    deadzone: float
    motor_tau_s: float
    slip_noise: float
    latency_s: float
    uplink_ratio: float  # 遅延のうち観測側 (ATOM→Colab) に割り振る割合
    latency_jitter_s: float
    cmd_drop_prob: float
    tof_noise_rel: float
    tof_noise_abs_mm: float
    tof_scale: float
    tof_bias_mm: float
    tof_dropout_prob: float
    tof_ghost_prob: float
    tof_fov_deg: float
    tof_angle_deg: float
    pose_noise_mm: float
    pose_noise_rad: float
    pose_dropout_prob: float
    radius_mm: float


def nominal_params(cfg: EnvConfig) -> SimParams:
    """ランダム化なし (公称値・ノイズなし・小さな遅延) のパラメータ。"""
    r = cfg.robot
    return SimParams(
        gain_left=1.0, gain_right=1.0, deadzone=r.deadzone, motor_tau_s=r.motor_tau_s, slip_noise=0.0,
        latency_s=0.15, uplink_ratio=0.5, latency_jitter_s=0.0, cmd_drop_prob=0.0,
        tof_noise_rel=0.0, tof_noise_abs_mm=0.0, tof_scale=1.0, tof_bias_mm=0.0,
        tof_dropout_prob=0.0, tof_ghost_prob=0.0, tof_fov_deg=r.tof_fov_deg, tof_angle_deg=r.tof_angle_deg,
        pose_noise_mm=0.0, pose_noise_rad=0.0, pose_dropout_prob=0.0, radius_mm=r.radius_mm,
    )  # fmt: skip


def sample_params(rng: np.random.Generator, cfg: EnvConfig, difficulty: float = 1.0) -> SimParams:
    """DR の範囲からパラメータを一様サンプリングする。

    difficulty < 1 のときは各範囲を中央値に向かって縮める (カリキュラム学習用)。
    """
    dr = cfg.dr
    s = 0.25 + 0.75 * float(np.clip(difficulty, 0.0, 1.0))

    def u(rng_range: tuple[float, float]) -> float:
        lo, hi = rng_range
        mid = 0.5 * (lo + hi)
        return float(rng.uniform(mid - (mid - lo) * s, mid + (hi - mid) * s))

    gain = u(dr.motor_gain)
    asym = u(dr.wheel_asymmetry)
    return SimParams(
        gain_left=gain * math.sqrt(asym),
        gain_right=gain / math.sqrt(asym),
        deadzone=u(dr.deadzone),
        motor_tau_s=u(dr.motor_tau_s),
        slip_noise=u(dr.slip_noise),
        latency_s=u(dr.latency_s),
        uplink_ratio=float(rng.uniform(0.3, 0.7)),
        latency_jitter_s=dr.latency_jitter_s * s,
        cmd_drop_prob=u(dr.cmd_drop_prob),
        tof_noise_rel=u(dr.tof_noise_rel),
        tof_noise_abs_mm=u(dr.tof_noise_abs_mm),
        tof_scale=u(dr.tof_scale),
        tof_bias_mm=u(dr.tof_bias_mm),
        tof_dropout_prob=u(dr.tof_dropout_prob),
        tof_ghost_prob=u(dr.tof_ghost_prob),
        tof_fov_deg=u(dr.tof_fov_deg),
        tof_angle_deg=cfg.robot.tof_angle_deg + u(dr.tof_angle_offset_deg),
        pose_noise_mm=u(dr.pose_noise_mm),
        pose_noise_rad=math.radians(u(dr.pose_noise_deg)),
        pose_dropout_prob=u(dr.pose_dropout_prob),
        radius_mm=u(dr.robot_radius_mm),
    )


# ---------------------------------------------------------------- レイアウト生成
def generate_layout(
    rng: np.random.Generator, cfg: EnvConfig, n_obstacles: int, robot_radius: float
) -> tuple[Obstacles, tuple[float, float, float]]:
    """障害物とロボット初期姿勢をランダムに配置する。"""
    W, H = cfg.mat_spec.width_mm, cfg.mat_spec.height_mm
    circles, boxes, bounds = [], [], []
    for _ in range(n_obstacles):
        for _try in range(50):
            if rng.random() < 0.5:
                r = rng.uniform(*cfg.obstacle_circle_radius_mm)
                cx, cy = rng.uniform(r, W - r), rng.uniform(r, H - r)
                cand = ("c", (cx, cy, r), r)
            else:
                w, h = rng.uniform(*cfg.obstacle_box_side_mm, size=2)
                br = 0.5 * math.hypot(w, h)
                cx, cy = rng.uniform(br, W - br), rng.uniform(br, H - br)
                cand = ("b", Obstacles.make_box(cx, cy, w, h, rng.uniform(0, math.pi)), br)
            if all(math.hypot(cx - bx, cy - by) - br - cand[2] >= cfg.obstacle_min_gap_mm for bx, by, br in bounds):
                (circles if cand[0] == "c" else boxes).append(cand[1])
                bounds.append((cx, cy, cand[2]))
                break
    obstacles = Obstacles(np.array(circles).reshape(-1, 3), np.array(boxes).reshape(-1, 4, 2))
    pose = sample_free_point(rng, cfg, obstacles, robot_radius + 25.0)
    return obstacles, (pose[0], pose[1], float(rng.uniform(-math.pi, math.pi)))


def sample_free_point(
    rng: np.random.Generator,
    cfg: EnvConfig,
    obstacles: Obstacles,
    min_clearance: float,
    avoid: Optional[tuple[float, float]] = None,
    min_dist: float = 0.0,
) -> tuple[float, float]:
    W, H = cfg.mat_spec.width_mm, cfg.mat_spec.height_mm
    m = cfg.spawn_margin_mm
    p = (W / 2, H / 2)
    for _ in range(500):
        p = (float(rng.uniform(m, W - m)), float(rng.uniform(m, H - m)))
        if clearance(np.array(p), obstacles) < min_clearance:
            continue
        if avoid is not None and math.hypot(p[0] - avoid[0], p[1] - avoid[1]) < min_dist:
            continue
        return p
    return p


# ---------------------------------------------------------------- 本体
@dataclass
class Measurement:
    tof_mm: Optional[float]  # None = 測定範囲外 / 欠測
    pose: Optional[tuple[float, float, float]]  # None = Position ID 読み取り失敗
    time_s: float


class ToioSim:
    """1 台の toio を時間発展させるシミュレータ (gym には依存しない)。

    env.py (学習) と virtual_robot.py (実機の代わりに MQTT で動く仮想ロボット) の両方で使う。
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.dt = cfg.control_dt_s / cfg.physics_substeps
        self.rng = np.random.default_rng()
        self.params = nominal_params(cfg)
        self.obstacles = Obstacles.empty()

    # ---- 初期化 ----
    def reset(
        self,
        rng: np.random.Generator,
        params: SimParams,
        obstacles: Obstacles,
        pose: tuple[float, float, float],
    ) -> None:
        self.rng, self.params, self.obstacles = rng, params, obstacles
        self.t = 0.0
        self.x, self.y, self.th = pose
        self.v_left = self.v_right = 0.0
        self.pending: deque[tuple[float, int, int]] = deque()  # (到着時刻, left, right)
        self.last_arrival = 0.0
        self.active_cmd = (0, 0)
        self.active_since = -math.inf
        self.applied_cmd = (0, 0)
        self.shield_count = 0
        self.shield_tof: Optional[float] = None
        hist_len = int(2.0 / self.dt) + 2  # 2 秒分 (最大遅延より十分長く)
        self.history: deque[tuple[float, float, float, float]] = deque(maxlen=hist_len)
        self.history.append((self.t, self.x, self.y, self.th))
        half = math.radians(params.tof_fov_deg) / 2
        self._ray_offsets = math.radians(params.tof_angle_deg) + np.linspace(-half, half, self.cfg.tof_rays)
        self.update_clearance()

    def update_clearance(self) -> float:
        """ロボット外周から最寄りの障害物までの距離 (0 以下で衝突)。"""
        self.clearance_mm = clearance(np.array([self.x, self.y]), self.obstacles) - self.params.radius_mm
        return self.clearance_mm

    # ---- 通信 ----
    def submit_command(self, left: int, right: int, delay_s: Optional[float] = None) -> None:
        """方策の指令を送る。delay_s=None なら DR の下り遅延 + ジッタ後に到着 (ロスあり)。"""
        p = self.params
        if delay_s is None:
            if self.rng.random() < p.cmd_drop_prob:
                return
            delay_s = p.latency_s * (1.0 - p.uplink_ratio) + self.rng.uniform(0.0, p.latency_jitter_s)
        arrival = max(self.t + delay_s, self.last_arrival)  # MQTT(TCP) は順序を保つ
        self.last_arrival = arrival
        self.pending.append((arrival, int(left), int(right)))

    def measure(self, delay_s: Optional[float] = None) -> Measurement:
        """センサー値を取得する。delay_s=None なら DR の上り遅延だけ過去の状態を観測する。"""
        p = self.params
        if delay_s is None:
            delay_s = p.latency_s * p.uplink_ratio + self.rng.uniform(0.0, p.latency_jitter_s)
        t_obs = self.t - delay_s
        rec = self.history[0]
        for rec_i in reversed(self.history):
            if rec_i[0] <= t_obs + 1e-9:
                rec = rec_i
                break
        _, x, y, th = rec

        pose = None
        if self.rng.random() >= p.pose_dropout_prob:
            pose = (
                x + self.rng.normal(0.0, p.pose_noise_mm),
                y + self.rng.normal(0.0, p.pose_noise_mm),
                th + self.rng.normal(0.0, p.pose_noise_rad),
            )
        return Measurement(self.read_tof(x, y, th), pose, rec[0])

    # ---- センサー ----
    def true_tof(self, x: float, y: float, th: float) -> float:
        """ノイズなしの ToF 距離 (視野内のレイの最短距離)。"""
        off = self.cfg.robot.tof_offset_mm
        origin = np.array([x + off * math.cos(th), y + off * math.sin(th)])
        return float(raycast(origin, th + self._ray_offsets, self.obstacles, TOF_PHYSICAL_MAX_MM).min())

    def read_tof(self, x: float, y: float, th: float) -> Optional[float]:
        p = self.params
        if self.rng.random() < p.tof_ghost_prob:
            return float(self.rng.uniform(30.0, self.cfg.tof_max_range_mm))
        d = self.true_tof(x, y, th)
        if d >= TOF_PHYSICAL_MAX_MM or self.rng.random() < p.tof_dropout_prob:
            return None
        d = d * p.tof_scale + p.tof_bias_mm + self.rng.normal(0.0, p.tof_noise_rel * d + p.tof_noise_abs_mm)
        return max(d, 0.0)

    # ---- 物理 ----
    def _unit_to_speed(self, unit: int, gain: float) -> float:
        a = abs(unit)
        if a < self.params.deadzone:
            return 0.0
        a = min(a, self.cfg.robot.max_unit)
        return math.copysign(a * self.cfg.robot.mm_per_s_per_unit * gain, unit)

    def advance(self, duration_s: float) -> int:
        """duration_s だけ時間を進める。戻り値はこの間に安全停止が働いたサブステップ数。"""
        cfg, p = self.cfg, self.params
        n = max(1, int(round(duration_s / self.dt)))
        alpha = 1.0 - math.exp(-self.dt / max(p.motor_tau_s, 1e-3))
        # 障害物の近くにいるときだけ、ファームウェアの安全停止をサブステップ単位で再現する。
        # ファームウェアが見る ToF は約 33ms ごとに更新されるので、2 サブステップ (40ms) ごとに読み直す
        near = self.clearance_mm < cfg.safety_stop_mm + 45.0
        shield_steps = 0
        for i in range(n):
            self.t += self.dt
            while self.pending and self.pending[0][0] <= self.t + 1e-9:
                _, l, r = self.pending.popleft()
                self.active_cmd, self.active_since = (l, r), self.t
            if self.t - self.active_since <= cfg.cmd_timeout_s:
                left, right = self.active_cmd
            else:
                left, right = 0, 0
            if near and left + right > 0:
                if i % 2 == 0 or self.shield_tof is None:
                    self.shield_tof = self.read_tof(self.x, self.y, self.th)
                left, right, shielded = apply_safety_shield(left, right, self.shield_tof, cfg.safety_stop_mm)
                shield_steps += shielded
            self.applied_cmd = (left, right)

            tl = self._unit_to_speed(left, p.gain_left)
            tr = self._unit_to_speed(right, p.gain_right)
            self.v_left += (tl - self.v_left) * alpha
            self.v_right += (tr - self.v_right) * alpha
            vl, vr = self.v_left, self.v_right
            if p.slip_noise > 0:
                vl *= 1.0 + self.rng.normal(0.0, p.slip_noise)
                vr *= 1.0 + self.rng.normal(0.0, p.slip_noise)
            v = 0.5 * (vl + vr)
            w = (vl - vr) / cfg.robot.wheel_base_mm
            th_mid = self.th + 0.5 * w * self.dt
            self.x += v * math.cos(th_mid) * self.dt
            self.y += v * math.sin(th_mid) * self.dt
            self.th = (self.th + w * self.dt + math.pi) % (2 * math.pi) - math.pi
            self.history.append((self.t, self.x, self.y, self.th))
        self.shield_tof = None
        self.shield_count += shield_steps
        self.update_clearance()
        return shield_steps

