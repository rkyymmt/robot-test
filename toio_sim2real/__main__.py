"""コマンドライン: python -m toio_sim2real {train,eval,run,virtual} ...

Colab ではノートブックから関数を直接呼ぶが、ローカル PC で試すときに使える。
"""
from __future__ import annotations

import argparse
import json
import time


def _mqtt_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="broker.emqx.io")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--prefix", required=True, help="ファームウェアの TOPIC_PREFIX と同じ値")
    p.add_argument("--user")
    p.add_argument("--password")
    p.add_argument("--tls", action="store_true")
    p.add_argument("--websockets", action="store_true")


def _mqtt_kwargs(a) -> dict:
    return dict(
        host=a.host, port=a.port, topic_prefix=a.prefix, username=a.user, password=a.password, tls=a.tls,
        transport="websockets" if a.websockets else "tcp",
    )  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m toio_sim2real")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="シミュレータで学習")
    p.add_argument("--out", default="runs/toio_avoid")
    p.add_argument("--steps", type=int, default=1_500_000)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--mat", default="toio_collection")
    p.add_argument("--config", help="EnvConfig の JSON (calibration.suggest_config の出力など)")
    p.add_argument("--resume", help="追加学習する model.zip")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval", help="シミュレータで評価")
    p.add_argument("--model", help="学習済みモデルのディレクトリ (省略時はルールベース)")
    p.add_argument("--episodes", type=int, default=30)

    p = sub.add_parser("run", help="実機 (または仮想ロボット) を方策で動かす")
    p.add_argument("--model", help="学習済みモデルのディレクトリ (省略時はルールベース)")
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--goals", help='マット座標 (dot) の JSON 例: "[[150,150],[350,350]]" (省略時はマットの四隅付近を巡回)')
    p.add_argument("--log", help="記録を保存する .npz")
    _mqtt_args(p)

    p = sub.add_parser("virtual", help="MQTT で動く仮想ロボットを起動")
    p.add_argument("--mat", default="toio_collection")
    p.add_argument("--obstacles", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    _mqtt_args(p)

    a = ap.parse_args()
    from .config import EnvConfig

    if a.cmd == "train":
        from .train import train

        cfg = EnvConfig.load(a.config) if a.config else EnvConfig(mat=a.mat)
        train(cfg, a.out, a.steps, n_envs=a.envs, seed=a.seed, resume_from=a.resume, verbose=1)
        return

    if a.cmd in ("eval", "run"):
        from .policies import RuleBasedPolicy, load_trained

        if a.model:
            policy, cfg = load_trained(a.model)
        else:
            cfg = EnvConfig()
            policy = RuleBasedPolicy(cfg)

    if a.cmd == "eval":
        from .evaluate import evaluate

        print("DR あり :", json.dumps(evaluate(policy, cfg, a.episodes), indent=1))
        print("DR なし :", json.dumps(evaluate(policy, cfg, a.episodes, dr=False), indent=1))
    elif a.cmd == "run":
        from .real_robot import MqttBridge, default_goals_dot, run_on_robot

        goals = json.loads(a.goals) if a.goals else default_goals_dot(cfg)
        with MqttBridge(**_mqtt_kwargs(a)) as bridge:
            log = run_on_robot(bridge, policy, cfg, goals, duration_s=a.duration)
        if a.log:
            log.save(a.log)
    elif a.cmd == "virtual":
        from .virtual_robot import VirtualRobot

        robot = VirtualRobot(EnvConfig(mat=a.mat), n_obstacles=a.obstacles, seed=a.seed, **_mqtt_kwargs(a)).start()
        print("仮想ロボット起動中 (Ctrl+C で終了)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        robot.stop()
        print(f"衝突 {robot.collisions} 回")


if __name__ == "__main__":
    main()
