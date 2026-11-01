"""Colab ⇔ ATOM 間の MQTT メッセージ定義。

firmware/atom_toio_bridge/atom_toio_bridge.ino と virtual_robot.py が同じ形式を使う。
変更するときは 3 か所をそろえること。

トピック (prefix はファームウェアの TOPIC_PREFIX と同じ値):
  {prefix}/state   ATOM → Colab  20Hz  ロボットの状態
  {prefix}/cmd     Colab → ATOM  10Hz  モーター指令 / ARM 切り替え
  {prefix}/status  ATOM → Colab  retained "online" / "offline" (LWT)

state: {"seq":12,"ms":34567,"tof":215,"x":250,"y":250,"a":90,"pv":1,"clash":0,"bat":80,
        "arm":1,"ack":55,"lat":180,"sh":0,"l":30,"r":28,"ble":1}
  tof  ToF 距離 [mm]。-1 は測定範囲外 (障害物なし)、-2 はセンサー異常
  x,y,a  Position ID (dot, 度)。pv=0 なら読み取れていない (マット外・持ち上げ)
  ack  最後に受け取った cmd の seq
  lat  その cmd の元になった state をサンプリングしてから cmd が届くまでの時間 [ms] (ATOM の時計で計測)
  sh   安全停止 (前進成分カット) が働いているか
  l,r  実際に toio へ送った速度指示値

cmd:   {"seq":56,"l":30,"r":28,"to":500,"sms":34500}
  to   この時間 [ms] 次の指令が来なければ停止
  sms  この指令の計算に使った state の ms (遅延計測用)
       {"arm":1}  ARM / DISARM (ファームウェアで ALLOW_REMOTE_ARM が有効な場合のみ ARM 可)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

TOPIC_STATE = "state"
TOPIC_CMD = "cmd"
TOPIC_STATUS = "status"


def topic(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}"


@dataclass
class RobotState:
    recv_time: float  # Colab 側で受信した時刻 (time.monotonic)
    seq: int
    ms: int
    tof_mm: Optional[int]  # None = 測定範囲外またはセンサー異常
    tof_error: bool
    x_dot: int
    y_dot: int
    angle_deg: int
    pos_valid: bool
    clash: bool
    battery: int
    armed: bool
    ack: int
    latency_ms: int
    shield: bool
    left: int
    right: int
    toio_connected: bool


def decode_state(payload: bytes, recv_time: float) -> RobotState:
    d = json.loads(payload)
    tof = int(d.get("tof", -2))
    return RobotState(
        recv_time=recv_time,
        seq=int(d.get("seq", 0)),
        ms=int(d.get("ms", 0)),
        tof_mm=tof if tof >= 0 else None,
        tof_error=tof == -2,
        x_dot=int(d.get("x", 0)),
        y_dot=int(d.get("y", 0)),
        angle_deg=int(d.get("a", 0)),
        pos_valid=bool(d.get("pv", 0)),
        clash=bool(d.get("clash", 0)),
        battery=int(d.get("bat", -1)),
        armed=bool(d.get("arm", 0)),
        ack=int(d.get("ack", -1)),
        latency_ms=int(d.get("lat", -1)),
        shield=bool(d.get("sh", 0)),
        left=int(d.get("l", 0)),
        right=int(d.get("r", 0)),
        toio_connected=bool(d.get("ble", 0)),
    )


def encode_state(
    seq: int, ms: int, tof_mm: Optional[float], x_dot: float, y_dot: float, angle_deg: float, pos_valid: bool,
    clash: bool, battery: int, armed: bool, ack: int, latency_ms: int, shield: bool, left: int, right: int,
) -> bytes:  # fmt: skip
    return json.dumps(
        {
            "seq": seq, "ms": ms, "tof": -1 if tof_mm is None else int(round(tof_mm)),
            "x": int(round(x_dot)), "y": int(round(y_dot)), "a": int(round(angle_deg)) % 360,
            "pv": int(pos_valid), "clash": int(clash), "bat": battery, "arm": int(armed),
            "ack": ack, "lat": latency_ms, "sh": int(shield), "l": left, "r": right, "ble": 1,
        },  # fmt: skip
        separators=(",", ":"),
    ).encode()


def encode_motor(seq: int, left: int, right: int, timeout_ms: int, state_ms: Optional[int] = None) -> bytes:
    d = {"seq": seq, "l": int(left), "r": int(right), "to": int(timeout_ms)}
    if state_ms is not None:
        d["sms"] = int(state_ms)
    return json.dumps(d, separators=(",", ":")).encode()


def encode_arm(on: bool) -> bytes:
    return json.dumps({"arm": int(on)}).encode()
