"""シミュレータと実機推論の両方から使う共通ロジック。

ここにある関数を sim (env.py) と実機 (real_robot.py) が同じように呼ぶことで、
「学習時と推論時で観測・行動の定義がずれる」という sim2real でありがちな不具合を防ぐ。

座標系はすべて toio マットに合わせる:
  x: 右向き, y: 下向き (mm, マット左上が原点)
  heading: x 軸正方向が 0、時計回り (= y 正方向) が正 [rad]
  → 左車輪の方が速いと heading が増える (右旋回)
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

import numpy as np

from .config import EnvConfig

# 1 フレームあたりの観測特徴量
FRAME_FEATURES = (
    "tof",  # ToF 距離 / tof_max_range  (0: 接触, 1: 障害物なし)
    "goal_dist",  # ゴールまでの距離 / goal_dist_scale (上限 1)
    "goal_sin",  # ロボットから見たゴール方位 (右が正)
    "goal_cos",
    "pos_x",  # マット上の位置 [-1, 1] (マット外へ出ないために使う)
    "pos_y",
    "head_cos",  # マット上での向き
    "head_sin",
    "prev_fwd",  # 前ステップの行動
    "prev_turn",
)
FRAME_DIM = len(FRAME_FEATURES)
F = {name: i for i, name in enumerate(FRAME_FEATURES)}


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ---------------------------------------------------------------- 行動
def action_to_wheel_units(action: Sequence[float], cfg: EnvConfig) -> tuple[int, int]:
    """方策の出力 [fwd, turn] ∈ [-1, 1]^2 を toio の左右速度指示値 (整数) に変換する。

    fwd  -1 → 後退 (最大の reverse_ratio 倍), +1 → 前進最大
    turn +1 → 右旋回 (左輪が速い)
    """
    fwd_a = _clip(float(action[0]), -1.0, 1.0)
    turn_a = _clip(float(action[1]), -1.0, 1.0)
    rr = cfg.reverse_ratio
    fwd = ((fwd_a + 1.0) * 0.5 * (1.0 + rr) - rr) * cfg.max_forward_unit
    turn = turn_a * cfg.max_turn_unit
    lim = cfg.max_wheel_unit
    left = int(_clip(round(fwd + turn), -lim, lim))
    right = int(_clip(round(fwd - turn), -lim, lim))
    return left, right


def forward_ratio_to_action(ratio: float, cfg: EnvConfig) -> float:
    """前進率 (-reverse_ratio〜1) を action[0] に変換する (ルールベース制御用)。"""
    rr = cfg.reverse_ratio
    ratio = _clip(float(ratio), -rr, 1.0)
    return 2.0 * (ratio + rr) / (1.0 + rr) - 1.0


def apply_safety_shield(left: int, right: int, tof_mm: Optional[float], safety_stop_mm: float) -> tuple[int, int, bool]:
    """ToF が近すぎるときに前進成分だけを打ち消す (旋回・後退は許可)。

    ファームウェア (atom_toio_bridge.ino の applySafetyShield) と同一ロジック。
    実機の安全機構をシミュレータでも再現しておくことで、学習時と実機で挙動が変わらない。
    """
    if tof_mm is not None and 0 <= tof_mm < safety_stop_mm and left + right > 0:
        w = int((left - right) / 2)  # C++ の整数除算と同じく 0 方向へ丸める
        return w, -w, True
    return left, right, False


# ---------------------------------------------------------------- 観測
class ObstacleMemory:
    """ToF が検出した点をマット座標で一定時間覚えておく。

    単眼の ToF は視野が約 25° と狭く、回避中に障害物が視野から外れると横からぶつかりやすい。
    Position ID で自己位置が正確に分かるので、検出点を地図に残し、
    ロボットから見た周囲の方向ごとの最寄り距離を観測に加える。
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        n = max(cfg.memory_points, 1)
        self.xy = np.zeros((n, 2))
        self.step_added = np.full(n, -(10**9))
        self.step = 0
        self.head = 0
        self.edges = np.linspace(-math.pi, math.pi, cfg.memory_sectors + 1)

    def reset(self) -> None:
        self.step_added[:] = -(10**9)
        self.step = 0

    def add(self, tof_mm: Optional[float], pose: Optional[tuple[float, float, float]]) -> None:
        self.step += 1
        if pose is None or tof_mm is None or not (0 <= tof_mm < self.cfg.tof_max_range_mm):
            return
        x, y, th = pose
        r = self.cfg.robot
        a = th + math.radians(r.tof_angle_deg)
        self.xy[self.head] = (
            x + r.tof_offset_mm * math.cos(th) + tof_mm * math.cos(a),
            y + r.tof_offset_mm * math.sin(th) + tof_mm * math.sin(a),
        )
        self.step_added[self.head] = self.step
        self.head = (self.head + 1) % len(self.xy)

    def sectors(self, pose: tuple[float, float, float]) -> np.ndarray:
        """ロボット正面を中心に等分した各方向の、記憶点までの距離 / memory_range (1 = なし)。"""
        cfg = self.cfg
        out = np.ones(cfg.memory_sectors, dtype=np.float32)
        live = self.step - self.step_added < cfg.memory_steps
        if not live.any():
            return out
        x, y, th = pose
        rel = self.xy[live] - (x, y)
        dist = np.hypot(rel[:, 0], rel[:, 1])
        near = dist < cfg.memory_range_mm
        if not near.any():
            return out
        # 正面方向がセクタの境目にならないよう半セクタずらす
        half = math.pi / cfg.memory_sectors
        bearing = (np.arctan2(rel[near, 1], rel[near, 0]) - th + math.pi + half) % (2 * math.pi) - math.pi
        idx = np.clip(np.searchsorted(self.edges, bearing, side="right") - 1, 0, cfg.memory_sectors - 1)
        np.minimum.at(out, idx, (dist[near] / cfg.memory_range_mm).astype(np.float32))
        return out


class ObservationBuilder:
    """センサー値の履歴から方策への入力ベクトルを作る。

    観測 = 直近 history_len フレーム (FRAME_FEATURES) + 障害物メモリ (memory_sectors 方向)
    単眼の ToF は視野が狭い (約 25°) ため、フレームを積み重ねたうえで
    検出点の地図 (ObstacleMemory) も使い、視野外の障害物を覚えていられるようにしている。
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.mat = cfg.mat_spec
        self.frames: deque[np.ndarray] = deque(maxlen=cfg.history_len)
        self.last_pose: Optional[tuple[float, float, float]] = None
        self.memory = ObstacleMemory(cfg) if cfg.memory_sectors > 0 else None

    @property
    def obs_dim(self) -> int:
        return FRAME_DIM * self.cfg.history_len + self.cfg.memory_sectors

    def reset(self) -> None:
        self.frames.clear()
        self.last_pose = None
        if self.memory is not None:
            self.memory.reset()

    def update(
        self,
        tof_mm: Optional[float],
        pose: Optional[tuple[float, float, float]],
        goal_mm: tuple[float, float],
        prev_action: Sequence[float],
    ) -> np.ndarray:
        """1 制御周期分の計測値を追加して観測ベクトルを返す。

        tof_mm: None または負値なら「障害物なし (測定範囲外)」
        pose:   (x_mm, y_mm, heading_rad)。None なら直前の有効値を使う (Position ID 読み取り失敗)
        """
        if pose is not None:
            self.last_pose = pose
        if self.last_pose is None:
            self.last_pose = (self.mat.width_mm / 2, self.mat.height_mm / 2, 0.0)
        x, y, th = self.last_pose

        cfg = self.cfg
        if tof_mm is None or tof_mm < 0:
            tof_mm = cfg.tof_max_range_mm
        tof_n = min(max(tof_mm, 0.0), cfg.tof_max_range_mm) / cfg.tof_max_range_mm

        dx, dy = goal_mm[0] - x, goal_mm[1] - y
        dist = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - th)

        frame = np.array(
            [
                tof_n,
                min(dist / cfg.goal_dist_scale_mm, 1.0),
                math.sin(bearing),
                math.cos(bearing),
                _clip(2.0 * x / self.mat.width_mm - 1.0, -1.0, 1.0),
                _clip(2.0 * y / self.mat.height_mm - 1.0, -1.0, 1.0),
                math.cos(th),
                math.sin(th),
                _clip(float(prev_action[0]), -1.0, 1.0),
                _clip(float(prev_action[1]), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        if not self.frames:
            self.frames.extend([frame] * cfg.history_len)
        self.frames.appendleft(frame)  # 先頭が最新
        if self.memory is None:
            return np.concatenate(self.frames)
        self.memory.add(None if tof_mm >= cfg.tof_max_range_mm else tof_mm, pose)
        return np.concatenate([*self.frames, self.memory.sectors(self.last_pose)])
