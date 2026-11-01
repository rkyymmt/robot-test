"""方策の読み込みと、比較用のルールベース制御。

どちらも stable-baselines3 と同じ predict(obs, deterministic=True) -> (action, state) を持つので、
評価 (evaluate.py) や実機推論 (real_robot.py) でそのまま差し替えられる。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .core import F, ObservationBuilder, forward_ratio_to_action

MODEL_FILE = "model.zip"
CONFIG_FILE = "env_config.json"


class RuleBasedPolicy:
    """ToF が近ければ右へ旋回、そうでなければゴール方向へ進むだけの単純な制御。

    学習なしで実機の配線・通信を確かめるときや、RL 方策との比較に使う。
    """

    def __init__(self, cfg: EnvConfig, avoid_mm: float = 150.0, turn_gain: float = 0.5):
        self.cfg = cfg
        self.avoid_mm = avoid_mm
        self.turn_gain = turn_gain

    def _act(self, obs: np.ndarray) -> np.ndarray:
        tof_mm = float(obs[F["tof"]]) * self.cfg.tof_max_range_mm
        bearing = math.atan2(float(obs[F["goal_sin"]]), float(obs[F["goal_cos"]]))
        if tof_mm < self.avoid_mm:
            return np.array([forward_ratio_to_action(0.0, self.cfg), 0.8], dtype=np.float32)
        slow = min(1.0, (tof_mm - self.avoid_mm) / 150.0 + 0.3)
        fwd = max(0.0, math.cos(bearing)) * slow
        turn = float(np.clip(self.turn_gain * bearing, -1.0, 1.0))
        return np.array([forward_ratio_to_action(fwd, self.cfg), turn], dtype=np.float32)

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs)
        if obs.ndim == 2:
            return np.stack([self._act(o) for o in obs]), None
        return self._act(obs), None


def load_trained(model_dir: str | Path, device: str = "cpu"):
    """学習済みモデルと、学習時の EnvConfig を読み込む。"""
    from stable_baselines3 import PPO

    model_dir = Path(model_dir)
    cfg = EnvConfig.load(model_dir / CONFIG_FILE)
    model = PPO.load(model_dir / MODEL_FILE, device=device)
    expected = ObservationBuilder(cfg).obs_dim
    if model.observation_space.shape != (expected,):
        raise ValueError(
            f"モデルの観測次元 {model.observation_space.shape} が {CONFIG_FILE} から作る観測 ({expected},) と一致しません。"
            "モデルと設定ファイルの組み合わせを確認してください"
        )
    return model, cfg
