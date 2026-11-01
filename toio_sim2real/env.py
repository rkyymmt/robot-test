"""強化学習用の gymnasium 環境: マット上のゴールへ障害物を避けながら移動する。

- 観測: 直近 history_len フレーム分の [ToF, ゴール距離・方位, 位置, 向き, 前回行動]
- 行動: [前進, 旋回] ∈ [-1, 1]^2 → core.action_to_wheel_units で左右の速度指示値へ
- ゴールに着くと次のゴールが出現し、衝突・マット外・時間切れで終了
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .core import ObservationBuilder, action_to_wheel_units
from .sim import ToioSim, generate_layout, nominal_params, sample_free_point, sample_params


class ToioAvoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[EnvConfig] = None, difficulty: float = 1.0):
        self.cfg = cfg or EnvConfig()
        self.sim = ToioSim(self.cfg)
        self.builder = ObservationBuilder(self.cfg)
        self.observation_space = spaces.Box(-1.0, 1.0, (self.builder.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.difficulty = float(difficulty)

    def set_difficulty(self, difficulty: float) -> None:
        """カリキュラム学習用: 0 で障害物少なめ・DR 狭め、1 で設定どおり。"""
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        """options:
        params: SimParams のフィールドを上書きする dict (例: {"latency_s": 0.3})
        layout: (Obstacles, (x, y, heading)) で配置を固定
        n_obstacles: 障害物の数を固定
        """
        super().reset(seed=seed)
        options = options or {}
        cfg, rng = self.cfg, self.np_random

        params = sample_params(rng, cfg, self.difficulty) if cfg.dr.enabled else nominal_params(cfg)
        if "params" in options:
            params = replace(params, **options["params"])
        if "layout" in options:
            obstacles, pose = options["layout"]
        else:
            lo, hi = cfg.obstacle_count
            n = options.get("n_obstacles", int(rng.integers(lo, lo + round((hi - lo) * self.difficulty) + 1)))
            obstacles, pose = generate_layout(rng, cfg, n, params.radius_mm)
        self.sim.reset(rng, params, obstacles, pose)

        self.goal = self._sample_goal()
        self.steps = 0
        self.goals_reached = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_goal_dist = self._goal_dist()
        self.builder.reset()
        return self._observe(), self._info(collided=False, out_of_bounds=False, shield=0)

    def step(self, action):
        cfg, rw, sim = self.cfg, self.cfg.reward, self.sim
        a = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        left, right = action_to_wheel_units(a, cfg)
        sim.submit_command(left, right)
        shield = sim.advance(cfg.control_dt_s)
        self.steps += 1

        clr = sim.clearance_mm
        W, H, m = cfg.mat_spec.width_mm, cfg.mat_spec.height_mm, cfg.edge_margin_mm
        collided = clr <= 0.0
        out_of_bounds = not (m <= sim.x <= W - m and m <= sim.y <= H - m)
        dist = self._goal_dist()

        reward = rw.progress_per_mm * (self.prev_goal_dist - dist)
        reward -= rw.time_penalty
        reward -= rw.smoothness_penalty * float(np.sum((a - self.prev_action) ** 2))
        reward -= rw.reverse_penalty * max(0.0, -(left + right) / (2.0 * cfg.max_forward_unit))
        reward -= rw.proximity_penalty * max(0.0, 1.0 - clr / rw.proximity_margin_mm)
        reward -= rw.shield_penalty * shield

        terminated = False
        if collided:
            reward -= rw.collision_penalty
            terminated = True
        elif out_of_bounds:
            reward -= rw.out_of_bounds_penalty
            terminated = True
        elif dist < cfg.goal_tolerance_mm:
            reward += rw.goal_bonus
            self.goals_reached += 1
            self.goal = self._sample_goal()
            dist = self._goal_dist()
        self.prev_goal_dist = dist
        truncated = self.steps >= cfg.max_steps and not terminated

        self.prev_action = a
        obs = self._observe()
        return obs, float(reward), terminated, truncated, self._info(collided, out_of_bounds, shield)

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        m = self.sim.measure()
        self.last_tof = m.tof_mm
        return self.builder.update(m.tof_mm, m.pose, self.goal, self.prev_action)

    def _goal_dist(self) -> float:
        return math.hypot(self.goal[0] - self.sim.x, self.goal[1] - self.sim.y)

    def _sample_goal(self) -> tuple[float, float]:
        sim = self.sim
        return sample_free_point(
            self.np_random, self.cfg, sim.obstacles, sim.params.radius_mm + 20.0,
            avoid=(sim.x, sim.y), min_dist=self.cfg.goal_min_start_dist_mm,
        )  # fmt: skip

    def _info(self, collided: bool, out_of_bounds: bool, shield: int) -> dict[str, Any]:
        return {
            "goals_reached": self.goals_reached,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "shield": shield,
            "is_success": not (collided or out_of_bounds),
        }

    def snapshot(self) -> dict[str, Any]:
        """可視化用の真の状態。"""
        s = self.sim
        return {
            "t": s.t, "x": s.x, "y": s.y, "th": s.th, "goal": tuple(self.goal),
            "tof": self.last_tof, "cmd": s.applied_cmd, "radius": s.params.radius_mm,
            "tof_fov_deg": s.params.tof_fov_deg, "tof_angle_deg": s.params.tof_angle_deg,
        }  # fmt: skip
