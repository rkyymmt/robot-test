"""Colab 上で実機 (ATOM Mate for toio) を動かす: MQTT ブリッジと推論ループ。

Colab のホスト型ランタイムはクラウド上にあり toio と直接 BLE 通信できないため、
ATOM (ESP32) が Wi-Fi で MQTT ブローカーに接続し、BLE で toio を操作する中継役になる。

  toio ⇔(BLE)⇔ ATOM ⇔(Wi-Fi/MQTT)⇔ ブローカー ⇔(MQTT)⇔ Colab (方策の推論)
"""
from __future__ import annotations

import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import EnvConfig
from .core import ObservationBuilder, action_to_wheel_units
from .protocol import (
    TOPIC_CMD, TOPIC_STATE, TOPIC_STATUS, RobotState, decode_state, encode_arm, encode_motor, topic,
)  # fmt: skip


class MqttBridge:
    """ATOM のファームウェアと MQTT でやり取りするクライアント。"""

    def __init__(
        self,
        host: str,
        port: int = 1883,
        topic_prefix: str = "toio-sim2real/demo",
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: bool = False,
        transport: str = "tcp",  # Colab から 1883 番ポートに出られない場合は "websockets"
        ws_path: str = "/mqtt",
    ):
        import paho.mqtt.client as mqtt

        self.host, self.port, self.prefix = host, port, topic_prefix
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"colab-{secrets.token_hex(4)}", transport=transport
        )
        if transport == "websockets":
            self._client.ws_set_options(path=ws_path)
        if username:
            self._client.username_pw_set(username, password)
        if tls:
            self._client.tls_set()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._first_state = threading.Event()
        self._state: Optional[RobotState] = None
        self._seq = 0
        self._last_ack = -1
        self.robot_status = "unknown"
        self.latency_ms: deque[int] = deque(maxlen=500)
        self.state_times: deque[float] = deque(maxlen=100)
        self.history: deque[RobotState] = deque(maxlen=4000)  # 20Hz で約 200 秒分

    # ---- 接続 ----
    def connect(self, timeout: float = 10.0) -> "MqttBridge":
        self._client.connect(self.host, self.port, keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(timeout):
            raise TimeoutError(f"MQTT ブローカー {self.host}:{self.port} に接続できません")
        return self

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.stop_motors()
        self.close()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            print(f"MQTT 接続失敗: {reason_code}")
            return
        # retained の status を最初の state より先に受け取れるよう、status から購読する
        client.subscribe(topic(self.prefix, TOPIC_STATUS), qos=0)
        client.subscribe(topic(self.prefix, TOPIC_STATE), qos=0)
        self._connected.set()

    def _on_message(self, client, userdata, msg):
        now = time.monotonic()
        if msg.topic.endswith("/" + TOPIC_STATUS):
            self.robot_status = msg.payload.decode(errors="replace")
            return
        try:
            st = decode_state(msg.payload, now)
        except (ValueError, KeyError) as e:
            print(f"不正な state メッセージ: {e}")
            return
        with self._lock:
            self._state = st
            self.history.append(st)
            self.state_times.append(now)
            if st.ack != self._last_ack and st.latency_ms >= 0:
                self._last_ack = st.ack
                self.latency_ms.append(st.latency_ms)
        self._first_state.set()

    # ---- 受信 ----
    def latest(self) -> Optional[RobotState]:
        with self._lock:
            return self._state

    def wait_for_state(self, timeout: float = 10.0) -> RobotState:
        if not self._first_state.wait(timeout):
            raise TimeoutError(
                f"ロボットから状態が届きません (status={self.robot_status})。"
                "ATOM の電源・Wi-Fi 設定・TOPIC_PREFIX を確認してください"
            )
        return self.latest()

    def states_since(self, t: float) -> list[RobotState]:
        """time.monotonic() が t 以降に受信した状態をすべて返す。"""
        with self._lock:
            return [s for s in self.history if s.recv_time >= t]

    def state_rate_hz(self) -> float:
        with self._lock:
            ts = list(self.state_times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    # ---- 送信 ----
    def send_motor(self, left: int, right: int, timeout_ms: int = 500, state_ms: Optional[int] = None) -> int:
        self._seq += 1
        self._client.publish(topic(self.prefix, TOPIC_CMD), encode_motor(self._seq, left, right, timeout_ms, state_ms))
        return self._seq

    def stop_motors(self, repeat: int = 3) -> None:
        for _ in range(repeat):
            self.send_motor(0, 0)
            time.sleep(0.03)

    def set_armed(self, on: bool) -> None:
        """ARM/DISARM。ARM はファームウェアで ALLOW_REMOTE_ARM=1 のときだけ有効 (既定は本体ボタンのみ)。"""
        self._client.publish(topic(self.prefix, TOPIC_CMD), encode_arm(on))


# ---------------------------------------------------------------- 制御
def default_goals_dot(cfg: EnvConfig, inset_mm: float = 90.0) -> list[tuple[int, int]]:
    """マットの四隅から inset_mm 内側の 4 点を対角線順に巡回するゴール (dot)。"""
    mat = cfg.mat_spec
    W, H = mat.width_mm, mat.height_mm
    pts = [(inset_mm, inset_mm), (W - inset_mm, H - inset_mm), (W - inset_mm, inset_mm), (inset_mm, H - inset_mm)]
    return [tuple(int(round(v)) for v in mat.mm_to_dot(x, y)) for x, y in pts]


class GoalSequence:
    """巡回するゴール (mm)。"""

    def __init__(self, goals_mm: Sequence[tuple[float, float]], loop: bool = True):
        assert goals_mm, "ゴールを 1 つ以上指定してください"
        self.goals = [tuple(map(float, g)) for g in goals_mm]
        self.loop = loop
        self.index = 0
        self.reached = 0

    @property
    def current(self) -> tuple[float, float]:
        return self.goals[self.index % len(self.goals)]

    @property
    def finished(self) -> bool:
        return not self.loop and self.index >= len(self.goals)

    def advance(self) -> None:
        self.reached += 1
        self.index += 1


class PolicyController:
    """実機の状態 → 観測 → 方策 → 速度指示値。env.ToioAvoidEnv と同じ部品・同じ順序で処理する。"""

    def __init__(self, policy, cfg: EnvConfig, goals: GoalSequence, deterministic: bool = True):
        self.policy, self.cfg, self.goals = policy, cfg, goals
        self.deterministic = deterministic
        self.builder = ObservationBuilder(cfg)
        self.prev_action = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.builder.reset()
        self.prev_action = np.zeros(2, dtype=np.float32)

    def step(self, st: RobotState) -> tuple[int, int, np.ndarray]:
        mat = self.cfg.mat_spec
        pose = None
        if st.pos_valid:
            x, y = mat.dot_to_mm(st.x_dot, st.y_dot)
            pose = (x, y, math.radians(st.angle_deg))
            gx, gy = self.goals.current
            if math.hypot(gx - x, gy - y) < self.cfg.goal_tolerance_mm:
                self.goals.advance()
        obs = self.builder.update(st.tof_mm, pose, self.goals.current, self.prev_action)
        action, _ = self.policy.predict(obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(2).clip(-1.0, 1.0)
        left, right = action_to_wheel_units(action, self.cfg)
        self.prev_action = action
        return left, right, action


@dataclass
class RunLog:
    rows: list[tuple] = field(default_factory=list)
    COLUMNS = (
        "t", "x_dot", "y_dot", "angle", "pos_valid", "tof", "left", "right", "act_fwd", "act_turn",
        "goal_x_dot", "goal_y_dot", "goals_reached", "latency_ms", "shield", "clash", "armed", "stopped",
    )  # fmt: skip

    def to_dict(self) -> dict[str, np.ndarray]:
        arr = np.array(self.rows, dtype=np.float64).reshape(-1, len(self.COLUMNS))
        return {c: arr[:, i] for i, c in enumerate(self.COLUMNS)}

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.to_dict())

    @staticmethod
    def load(path: str | Path) -> dict[str, np.ndarray]:
        return dict(np.load(path))


def run_on_robot(
    bridge: MqttBridge,
    policy,
    cfg: EnvConfig,
    goals_dot: Sequence[tuple[float, float]],
    duration_s: float = 60.0,
    loop_goals: bool = True,
    print_every_s: float = 2.0,
    stale_timeout_s: float = 0.5,
    pose_lost_timeout_s: float = 0.5,
) -> RunLog:
    """方策で実機を動かす。Colab のセル停止ボタン (KeyboardInterrupt) でも安全に止まる。

    goals_dot: マット座標 (dot) のゴール列。到達すると次へ進む。
    """
    mat = cfg.mat_spec
    goals = GoalSequence([mat.dot_to_mm(*g) for g in goals_dot], loop=loop_goals)
    ctrl = PolicyController(policy, cfg, goals)
    log = RunLog()
    timeout_ms = int(cfg.cmd_timeout_s * 1000)

    st = bridge.wait_for_state()
    if not st.armed:
        print("⚠ ロボットが DISARM 状態です。ATOM のボタンを押すと動き始めます (toio の LED が緑)")
    print(f"開始: {duration_s:.0f} 秒間, ゴール {len(goals.goals)} 点, 制御周期 {cfg.control_dt_s * 1000:.0f} ms")

    t0 = time.monotonic()
    next_tick, next_print = t0, t0
    last_pose_t, prev_clash, clash_count = t0, False, 0
    try:
        while (now := time.monotonic()) - t0 < duration_s and not goals.finished:
            st = bridge.latest()
            stopped = ""
            if now - st.recv_time > stale_timeout_s:
                stopped = "state stale"
            elif st.tof_error:
                stopped = "ToF error"
            elif not st.armed:
                stopped = "DISARM"  # 実行されない行動を観測履歴に混ぜない
            elif st.pos_valid:
                last_pose_t = now
            elif now - last_pose_t > pose_lost_timeout_s:
                stopped = "pose lost"

            if stopped:
                left, right, action = 0, 0, np.zeros(2, dtype=np.float32)
                ctrl.reset()  # 再開時に古い履歴を使わない
                bridge.send_motor(0, 0, timeout_ms)
            else:
                left, right, action = ctrl.step(st)
                bridge.send_motor(left, right, timeout_ms, state_ms=st.ms)

            clash_count += st.clash and not prev_clash
            prev_clash = st.clash
            gx, gy = mat.mm_to_dot(*goals.current)
            log.rows.append((
                now, st.x_dot, st.y_dot, st.angle_deg, st.pos_valid, -1 if st.tof_mm is None else st.tof_mm,
                left, right, action[0], action[1], gx, gy, goals.reached, st.latency_ms, st.shield, st.clash,
                st.armed, bool(stopped),
            ))  # fmt: skip

            if now >= next_print:
                next_print += print_every_s
                tof = f"{st.tof_mm:4d}" if st.tof_mm is not None else "  --"
                lat = int(np.median(bridge.latency_ms)) if bridge.latency_ms else -1
                print(
                    f"t={now - t0:5.1f}s pos=({st.x_dot:3d},{st.y_dot:3d},{st.angle_deg:3d}°) "
                    f"ToF={tof}mm cmd=({left:+3d},{right:+3d}) goal#{goals.index % len(goals.goals) + 1} "
                    f"reached={goals.reached} lat={lat}ms"
                    f"{' SHIELD' if st.shield else ''}{' [' + stopped + ']' if stopped else ''}"
                )

            next_tick += cfg.control_dt_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # 処理が遅れたら周期をリセット
    except KeyboardInterrupt:
        print("中断しました")
    finally:
        bridge.stop_motors()
    print(f"終了: ゴール到達 {goals.reached} 回, 衝突検知 (toio の clash) {clash_count} 回")
    return log
