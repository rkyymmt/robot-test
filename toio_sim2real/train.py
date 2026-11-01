"""PPO (stable-baselines3) による学習。"""
from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from .config import EnvConfig
from .env import ToioAvoidEnv
from .policies import CONFIG_FILE, MODEL_FILE

DEFAULT_PPO_KWARGS = dict(
    n_steps=1024,
    batch_size=512,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.003,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


def build_vec_env(cfg: EnvConfig, n_envs: int, seed: int, difficulty: float = 1.0, subproc: bool = False):
    def make(rank: int):
        def _init():
            env = ToioAvoidEnv(cfg, difficulty=difficulty)
            env.reset(seed=seed + rank)
            return env

        return _init

    fns = [make(i) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if subproc else DummyVecEnv(fns)
    return VecMonitor(vec)


class ToioTrainingCallback(BaseCallback):
    """カリキュラム (難易度を徐々に上げる) と、タスク固有の指標の記録。"""

    def __init__(self, total_timesteps: int, start: float, ramp_fraction: float):
        super().__init__()
        self.total = total_timesteps
        self.start = start
        self.ramp = max(ramp_fraction, 1e-6)
        self.difficulty = start
        self.goals: deque[float] = deque(maxlen=200)
        self.collided: deque[float] = deque(maxlen=200)
        self.oob: deque[float] = deque(maxlen=200)

    def _on_rollout_start(self) -> None:
        progress = self.num_timesteps / (self.total * self.ramp)
        self.difficulty = float(min(1.0, self.start + (1.0 - self.start) * progress))
        self.training_env.env_method("set_difficulty", self.difficulty)

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.goals.append(info["goals_reached"])
                self.collided.append(info["collided"])
                self.oob.append(info["out_of_bounds"])
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("toio/difficulty", self.difficulty)
        if self.goals:
            self.logger.record("toio/goals_per_episode", float(np.mean(self.goals)))
            self.logger.record("toio/collision_rate", float(np.mean(self.collided)))
            self.logger.record("toio/out_of_bounds_rate", float(np.mean(self.oob)))


def train(
    cfg: Optional[EnvConfig] = None,
    out_dir: str | Path = "runs/toio_avoid",
    total_timesteps: int = 1_500_000,
    n_envs: int = 8,
    seed: int = 0,
    subproc: bool = False,
    curriculum_start: float = 0.3,
    curriculum_ramp_fraction: float = 0.5,
    checkpoint_every: int = 250_000,
    resume_from: Optional[str | Path] = None,
    ppo_kwargs: Optional[dict] = None,
    progress_bar: bool = True,
    verbose: int = 0,
) -> PPO:
    """学習して out_dir に model.zip と env_config.json を保存する。

    resume_from: 既存の model.zip から追加学習する (実機キャリブレーション後に
                 DR の範囲を変えて微調整する場合など)。その場合はカリキュラムを省略する。
    """
    cfg = cfg or EnvConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / CONFIG_FILE)  # 推論時に同じ観測定義を使うため最初に保存

    if resume_from is not None:
        curriculum_start = 1.0
    env = build_vec_env(cfg, n_envs, seed, difficulty=curriculum_start, subproc=subproc)
    tb_dir = str(out / "tb") if importlib.util.find_spec("tensorboard") else None
    if resume_from is not None:
        model = PPO.load(resume_from, env=env, device="cpu", tensorboard_log=tb_dir)
    else:
        kwargs = {**DEFAULT_PPO_KWARGS, **(ppo_kwargs or {})}
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", tensorboard_log=tb_dir, verbose=verbose, **kwargs)

    callbacks = CallbackList([
        ToioTrainingCallback(total_timesteps, curriculum_start, curriculum_ramp_fraction),
        CheckpointCallback(max(checkpoint_every // n_envs, 1), str(out / "checkpoints"), name_prefix="ppo"),
    ])  # fmt: skip
    use_bar = progress_bar and all(importlib.util.find_spec(m) for m in ("tqdm", "rich"))
    model.learn(
        total_timesteps,
        callback=callbacks,
        progress_bar=use_bar,
        reset_num_timesteps=resume_from is None,
        tb_log_name="ppo",
    )
    model.save(out / MODEL_FILE)
    env.close()
    return model
