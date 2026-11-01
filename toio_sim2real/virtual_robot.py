"""実機の代わりに MQTT で動く仮想ロボット。

ファームウェア (atom_toio_bridge.ino) と同じトピック・メッセージ形式を話すので、
ハードウェアがなくても Colab の推論ノートブックを最後まで試せる。
中身は学習に使ったシミュレータ (ToioSim) で、通信遅延は実際のネットワークのものがそのまま乗る。
"""
from __future__ import annotations

import json
import math
import secrets
import threading
import time
from dataclasses import replace
from typing import Optional

import numpy as np

from .config import EnvConfig
from .protocol import TOPIC_CMD, TOPIC_STATE, TOPIC_STATUS, encode_state, topic
from .sim import ToioSim, generate_layout, nominal_params, sample_params

STATE_PERIOD_S = 0.05  # ファームウェアの STATE_PERIOD_MS と同じ


class VirtualRobot:
    def __init__(
        self,
        cfg: EnvConfig,
        host: str,
        port: int = 1883,
        topic_prefix: str = "toio-sim2real/demo",
        username: Optional[str] = None,
        password: Optional[str] = None,
        tls: bool = False,
        transport: str = "tcp",
        ws_path: str = "/mqtt",
        seed: int = 0,
        n_obstacles: int = 3,
        randomize: bool = True,
        armed: bool = True,
    ):
        import paho.mqtt.client as mqtt

        self.cfg, self.prefix, self.armed = cfg, topic_prefix, armed
        rng = np.random.default_rng(seed)
        params = sample_params(rng, cfg) if randomize else nominal_params(cfg)
        # 通信遅延・パケットロスは本物のネットワークで発生するのでシミュレータ側では 0 にする
        params = replace(params, latency_s=0.0, latency_jitter_s=0.0, cmd_drop_prob=0.0)
        self.obstacles, pose = generate_layout(rng, cfg, n_obstacles, params.radius_mm)
        self.sim = ToioSim(cfg)
        self.sim.reset(rng, params, self.obstacles, pose)
        self.trajectory: list[tuple[float, float, float]] = []
        self.collisions = 0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = time.monotonic()
        self._ack, self._latency_ms, self._clash = -1, -1, False

        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"vrobot-{secrets.token_hex(4)}", transport=transport)
        if transport == "websockets":
            c.ws_set_options(path=ws_path)
        if username:
            c.username_pw_set(username, password)
        if tls:
            c.tls_set()
        c.will_set(topic(topic_prefix, TOPIC_STATUS), "offline", retain=True)
        c.on_connect = lambda cl, *a: cl.subscribe(topic(topic_prefix, TOPIC_CMD))
        c.on_message = self._on_message
        self._client = c
        self._host, self._port = host, port

    def _ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000)

    def _on_message(self, client, userdata, msg):
        try:
            d = json.loads(msg.payload)
        except ValueError:
            return
        with self._lock:
            if "arm" in d:
                self.armed = bool(d["arm"])
            if "seq" in d:
                self._ack = int(d["seq"])
                if "sms" in d:
                    self._latency_ms = self._ms() - int(d["sms"])
                if self.armed:
                    self.sim.submit_command(int(d.get("l", 0)), int(d.get("r", 0)), delay_s=0.0)

    # ------------------------------------------------------------------
    def start(self) -> "VirtualRobot":
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()
        self._t0 = time.monotonic()  # sim.t = 0 と実時間の起点をそろえる
        self._client.publish(topic(self.prefix, TOPIC_STATUS), "online", retain=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._client.publish(topic(self.prefix, TOPIC_STATUS), "offline", retain=True).wait_for_publish(2)
        self._client.loop_stop()
        self._client.disconnect()

    def _run(self) -> None:
        sim, mat = self.sim, self.cfg.mat_spec
        W, H = mat.width_mm, mat.height_mm
        seq, shield, next_pub = 0, False, time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                if not self.armed:
                    sim.pending.clear()
                    sim.active_cmd = (0, 0)
                # シミュレータ時間を実時間に追従させる
                while sim.t < time.monotonic() - self._t0:
                    x0, y0 = sim.x, sim.y
                    shield = sim.advance(sim.dt) > 0
                    if sim.clearance_mm <= 0:  # 障害物にめり込まないよう押し戻す (toio の clash として通知)
                        sim.x, sim.y = x0, y0
                        sim.update_clearance()
                        self.collisions += not self._clash
                        self._clash = True
                    else:
                        self._clash = False
                    self.trajectory.append((sim.t, sim.x, sim.y))

                if time.monotonic() >= next_pub:
                    next_pub += STATE_PERIOD_S
                    m = sim.measure(delay_s=0.0)
                    on_mat = 0 <= sim.x <= W and 0 <= sim.y <= H  # マット外では Position ID が読めない
                    pose = m.pose if on_mat else None
                    x_dot, y_dot = mat.mm_to_dot(*pose[:2]) if pose else (0, 0)
                    angle = math.degrees(pose[2]) if pose else 0
                    seq += 1
                    payload = encode_state(
                        seq, self._ms(), m.tof_mm, x_dot, y_dot, angle, pose is not None, self._clash, 100,
                        self.armed, self._ack, self._latency_ms, shield, *sim.applied_cmd,
                    )  # fmt: skip
                    self._client.publish(topic(self.prefix, TOPIC_STATE), payload)
            time.sleep(sim.dt / 2)
