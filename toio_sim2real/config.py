"""シミュレータ・学習・実機推論で共有する設定。

数値の出典:
- 座標スケール / 車輪間隔 / 速度係数 / モーター時定数:
  toio SDK for Unity の CubeSimulator.cs, Mat.cs, CubeHandle.cs
- 速度指示値の不感帯 (8 未満は停止) と飽和 (115):
  toio コア キューブ技術仕様「モーター」
- マット座標範囲: toio コア キューブ技術仕様「Position ID」

学習済みモデルと一緒に env_config.json として保存し、Colab での実機推論時に
同じ設定を読み込むことで、観測・行動の定義がずれないようにしている。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

# toio SDK for Unity: Mat.DotPerM = 411 / 0.560 (dot/m)
MM_PER_DOT = 560.0 / 411.0  # ≒ 1.36 mm/dot


@dataclass(frozen=True)
class MatSpec:
    """toio のプレイマット。座標は Position ID の dot 単位。"""

    name: str
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def width_mm(self) -> float:
        return (self.x_max - self.x_min) * MM_PER_DOT

    @property
    def height_mm(self) -> float:
        return (self.y_max - self.y_min) * MM_PER_DOT

    def dot_to_mm(self, x_dot: float, y_dot: float) -> tuple[float, float]:
        """マット座標 (dot) → シミュレータ座標 (mm, マット左上が原点, y は下向き)。"""
        return (x_dot - self.x_min) * MM_PER_DOT, (y_dot - self.y_min) * MM_PER_DOT

    def mm_to_dot(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return x_mm / MM_PER_DOT + self.x_min, y_mm / MM_PER_DOT + self.y_min


MAT_PRESETS: dict[str, MatSpec] = {
    # トイオ・コレクション付属マット(土俵の面) 約 56cm 四方
    "toio_collection": MatSpec("toio_collection", 45, 455, 45, 455),
    # toio コア キューブ(単体)付属の簡易プレイマット 約 41 x 29 cm
    "simple": MatSpec("simple", 98, 402, 142, 358),
}


@dataclass
class RobotConfig:
    """toio コア キューブ + ATOM Mate for toio の公称値。"""

    wheel_base_mm: float = 26.6
    # 4.3 rpm/指示値 × 直径 12.5mm の車輪 → 約 2.81 mm/s/指示値
    mm_per_s_per_unit: float = 4.3 * math.pi * 12.5 / 60.0
    deadzone: int = 8  # これ未満の速度指示値ではモーターが回らない
    max_unit: int = 115  # これ以上は速度が飽和する
    motor_tau_s: float = 0.04
    # 衝突判定用の外接円半径 (キューブ 31.8mm 角 + ATOM Mate のはみ出し分)
    radius_mm: float = 22.0
    # ToF センサー (VL53L0X) の取り付け位置: キューブ中心から前方へのオフセットと向き
    tof_offset_mm: float = 16.0
    tof_angle_deg: float = 0.0
    tof_fov_deg: float = 25.0


@dataclass
class DomainRandomization:
    """エピソードごとにサンプリングする物理・センサー・通信パラメータの範囲。

    sim2real の要。実機キャリブレーション (calibration.py) の結果が
    この範囲に収まっていることを確認し、外れていたら範囲を広げて再学習する。
    """

    enabled: bool = True
    motor_gain: tuple[float, float] = (0.85, 1.15)  # 速度係数の個体差・電池残量
    wheel_asymmetry: tuple[float, float] = (0.93, 1.07)  # 左右輪の差 (直進性)
    deadzone: tuple[float, float] = (7.0, 11.0)
    motor_tau_s: tuple[float, float] = (0.03, 0.12)
    slip_noise: tuple[float, float] = (0.0, 0.08)  # サブステップごとの車輪速度ノイズ (相対)
    # 観測取得から指令反映までの往復遅延。Colab ⇔ MQTT ブローカー ⇔ ATOM ⇔ BLE ⇔ toio
    latency_s: tuple[float, float] = (0.08, 0.45)
    latency_jitter_s: float = 0.04
    cmd_drop_prob: tuple[float, float] = (0.0, 0.05)
    tof_noise_rel: tuple[float, float] = (0.01, 0.05)
    tof_noise_abs_mm: tuple[float, float] = (1.0, 6.0)
    tof_scale: tuple[float, float] = (0.95, 1.05)
    tof_bias_mm: tuple[float, float] = (-10.0, 10.0)
    tof_dropout_prob: tuple[float, float] = (0.0, 0.05)  # 反射が弱く「障害物なし」になる確率
    tof_ghost_prob: tuple[float, float] = (0.0, 0.01)  # 何もないのに近距離を返す確率
    tof_fov_deg: tuple[float, float] = (15.0, 27.0)
    tof_angle_offset_deg: tuple[float, float] = (-4.0, 4.0)  # 取り付け誤差
    pose_noise_mm: tuple[float, float] = (0.5, 3.0)
    pose_noise_deg: tuple[float, float] = (0.5, 3.0)
    pose_dropout_prob: tuple[float, float] = (0.0, 0.03)
    robot_radius_mm: tuple[float, float] = (20.0, 24.0)


@dataclass
class RewardConfig:
    progress_per_mm: float = 0.01
    goal_bonus: float = 5.0
    collision_penalty: float = 10.0
    out_of_bounds_penalty: float = 10.0
    proximity_penalty: float = 0.05
    proximity_margin_mm: float = 40.0
    time_penalty: float = 0.005
    smoothness_penalty: float = 0.02
    reverse_penalty: float = 0.02
    shield_penalty: float = 0.05


@dataclass
class EnvConfig:
    mat: str = "toio_collection"
    robot: RobotConfig = field(default_factory=RobotConfig)
    dr: DomainRandomization = field(default_factory=DomainRandomization)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # --- 制御 ---
    control_dt_s: float = 0.1  # 方策の実行周期 (Colab 側の制御ループと一致させる)
    physics_substeps: int = 5
    max_steps: int = 600  # 1 エピソード 60 秒
    max_forward_unit: int = 45  # 前進成分の最大速度指示値 (約 126 mm/s)
    max_turn_unit: int = 25  # 旋回成分の最大速度指示値
    max_wheel_unit: int = 80  # 左右輪それぞれの上限 (ファームウェア側でもクランプ)
    reverse_ratio: float = 0.3  # 後退は前進最大の 30% まで (後方は見えないため)
    cmd_timeout_s: float = 0.5  # この時間指令が来なければロボット側で停止
    safety_stop_mm: float = 30.0  # ToF がこれ未満なら前進成分をカット (ファームウェアと同一ロジック)

    # --- 観測 ---
    history_len: int = 8  # 直近何フレームを観測に積むか (0.8 秒分)
    tof_max_range_mm: float = 400.0  # これ以上は「障害物なし」として扱う
    tof_rays: int = 7  # FOV 内を何本のレイで近似するか
    goal_dist_scale_mm: float = 300.0
    # 障害物メモリ: ToF が検出した点をマット座標で覚えておき、周囲 memory_sectors 方向の
    # 最寄り距離を観測に加える (視野外になった障害物に横からぶつからないため)。0 で無効
    memory_sectors: int = 8
    memory_range_mm: float = 200.0
    memory_steps: int = 100  # 何ステップ (10 秒) 覚えておくか
    memory_points: int = 100

    # --- タスク ---
    goal_tolerance_mm: float = 25.0
    goal_min_start_dist_mm: float = 150.0
    edge_margin_mm: float = 10.0  # 中心がマット端からこの距離より外に出たら失敗
    spawn_margin_mm: float = 45.0
    obstacle_count: tuple[int, int] = (1, 5)
    obstacle_circle_radius_mm: tuple[float, float] = (15.0, 35.0)
    obstacle_box_side_mm: tuple[float, float] = (25.0, 70.0)
    obstacle_min_gap_mm: float = 55.0  # 障害物同士の最小すき間 (ロボットが通れる幅)

    @property
    def mat_spec(self) -> MatSpec:
        return MAT_PRESETS[self.mat]

    # ---- JSON 入出力 ----
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return _from_dict(cls, d)

    @classmethod
    def load(cls, path: str | Path) -> "EnvConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _from_dict(cls, d: dict):
    """ネストした dataclass を復元する (JSON で list になった tuple も戻す)。"""
    defaults = cls()
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        default = getattr(defaults, f.name)
        if is_dataclass(default):
            v = _from_dict(type(default), v)
        elif isinstance(default, tuple):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)
