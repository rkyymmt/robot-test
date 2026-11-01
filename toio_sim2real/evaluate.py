"""シミュレータ上での評価: 成功率・衝突率・ゴール到達数、遅延に対する頑健性。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .config import EnvConfig
from .env import ToioAvoidEnv
from .sim import Obstacles


@dataclass
class EpisodeResult:
    goals: int
    collided: bool
    out_of_bounds: bool
    steps: int
    shield_steps: int
    obstacles: Obstacles
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_episode(
    env: ToioAvoidEnv,
    policy,
    seed: Optional[int] = None,
    options: Optional[dict] = None,
    record: bool = False,
    deterministic: bool = True,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed, options=options)
    trace = [env.snapshot()] if record else []
    shield = 0
    while True:
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        shield += info["shield"]
        if record:
            trace.append(env.snapshot())
        if terminated or truncated:
            break
    return EpisodeResult(
        goals=info["goals_reached"], collided=info["collided"], out_of_bounds=info["out_of_bounds"],
        steps=env.steps, shield_steps=shield, obstacles=env.sim.obstacles, trace=trace,
    )  # fmt: skip


def evaluate(
    policy,
    cfg: EnvConfig,
    n_episodes: int = 30,
    seed: int = 10_000,
    dr: bool = True,
    params: Optional[dict] = None,
) -> dict[str, float]:
    """params: SimParams の一部を固定する dict (例: {"latency_s": 0.3})。"""
    cfg = copy.deepcopy(cfg)
    cfg.dr.enabled = dr
    env = ToioAvoidEnv(cfg)
    options = {"params": params} if params else None
    results = [run_episode(env, policy, seed=seed + i, options=options) for i in range(n_episodes)]
    minutes = sum(r.steps for r in results) * cfg.control_dt_s / 60.0
    return {
        "episodes": n_episodes,
        "goals_per_min": sum(r.goals for r in results) / max(minutes, 1e-9),
        "goals_per_episode": float(np.mean([r.goals for r in results])),
        "collision_rate": float(np.mean([r.collided for r in results])),
        "out_of_bounds_rate": float(np.mean([r.out_of_bounds for r in results])),
        "survival_rate": float(np.mean([not (r.collided or r.out_of_bounds) for r in results])),
        "shield_steps_per_episode": float(np.mean([r.shield_steps for r in results])),
    }


def latency_sweep(
    policy,
    cfg: EnvConfig,
    latencies: Sequence[float] = (0.05, 0.15, 0.3, 0.45, 0.6, 0.8),
    n_episodes: int = 20,
    seed: int = 20_000,
) -> list[dict[str, float]]:
    """通信遅延だけを固定値にして評価する。学習時の DR 範囲外での性能低下を確認する。"""
    rows = []
    for lat in latencies:
        row = evaluate(policy, cfg, n_episodes, seed, dr=True, params={"latency_s": lat})
        rows.append({"latency_s": lat, **row})
    return rows
