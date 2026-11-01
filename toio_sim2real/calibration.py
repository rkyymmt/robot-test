"""実機キャリブレーション: 実機の特性を測り、シミュレータの DR 範囲に収まっているか確認する。

sim2real の手順:
  1. 実機で遅延・速度・旋回・ToF を測る (このモジュール)
  2. 測定値が EnvConfig.dr の範囲外なら suggest_config() で範囲を広げた設定を作る
  3. その設定で追加学習 (train(..., resume_from=...)) してから再度実機で試す
"""
from __future__ import annotations

import copy
import math
import time
from typing import Optional

import numpy as np

from .config import EnvConfig
from .core import wrap_angle
from .protocol import RobotState
from .real_robot import MqttBridge

# ATOM での計測遅延に加えて、BLE 書き込み・ToF のサンプリング間隔ぶんを足して DR と比べる
EXTRA_LATENCY_S = 0.05


def _collect(bridge: MqttBridge, seconds: float, cmd: tuple[int, int] = (0, 0), period: float = 0.1) -> list[RobotState]:
    """seconds 秒間、cmd を 10Hz で送り続け、その間に受信した状態をすべて返す。"""
    t_start = time.monotonic()
    while time.monotonic() - t_start < seconds:
        st = bridge.latest()
        bridge.send_motor(cmd[0], cmd[1], state_ms=st.ms if st else None)
        time.sleep(period)
    return bridge.states_since(t_start)


def _settled_pose(bridge: MqttBridge, cfg: EnvConfig, seconds: float = 0.5) -> tuple[float, float, float]:
    sts = [s for s in _collect(bridge, seconds) if s.pos_valid]
    if not sts:
        raise RuntimeError("Position ID が読めません。toio をマットの上に置いてください")
    xy = np.array([cfg.mat_spec.dot_to_mm(s.x_dot, s.y_dot) for s in sts])
    ang = np.radians([s.angle_deg for s in sts])
    return float(xy[:, 0].mean()), float(xy[:, 1].mean()), math.atan2(np.sin(ang).mean(), np.cos(ang).mean())


def link_check(bridge: MqttBridge, seconds: float = 5.0) -> dict:
    """状態の受信レートと制御ループ遅延 (モーターは停止のまま)。"""
    bridge.latency_ms.clear()
    sts = _collect(bridge, seconds)
    lat = np.array(bridge.latency_ms, dtype=float)
    lat = lat[lat >= 0]
    st = sts[-1] if sts else bridge.latest()
    return {
        "state_hz": bridge.state_rate_hz(),
        "latency_ms_median": float(np.median(lat)) if len(lat) else float("nan"),
        "latency_ms_p90": float(np.percentile(lat, 90)) if len(lat) else float("nan"),
        "latency_ms_max": float(lat.max()) if len(lat) else float("nan"),
        "battery": st.battery if st else -1,
        "armed": st.armed if st else False,
        "toio_connected": st.toio_connected if st else False,
    }


def tof_check(bridge: MqttBridge, seconds: float = 3.0) -> dict:
    """ToF の統計。障害物なしで測れば床や周囲の誤検出、前方に物を置けば向きの確認になる。"""
    sts = _collect(bridge, seconds)
    vals = np.array([s.tof_mm for s in sts if s.tof_mm is not None], dtype=float)
    return {
        "samples": len(sts),
        "valid_ratio": len(vals) / max(len(sts), 1),
        "min_mm": float(vals.min()) if len(vals) else float("nan"),
        "median_mm": float(np.median(vals)) if len(vals) else float("nan"),
        "max_mm": float(vals.max()) if len(vals) else float("nan"),
    }


def _drive(bridge: MqttBridge, cfg: EnvConfig, left: int, right: int, seconds: float) -> dict:
    """指令 (left, right) で走らせ、定常走行中の位置・角度の時系列を返す。

    時刻は ATOM の時計 (state の ms) を使うので、通信遅延やジッタの影響を受けない。
    """
    st = bridge.wait_for_state()
    if not st.armed:
        raise RuntimeError("DISARM 状態です。ATOM のボタンを押して ARM してから実行してください")
    p0 = _settled_pose(bridge, cfg)
    sts = _collect(bridge, seconds, cmd=(left, right))
    bridge.stop_motors()
    time.sleep(0.6)
    p1 = _settled_pose(bridge, cfg)

    moving = [s for s in sts if (s.left, s.right) == (left, right) and s.pos_valid]
    if len(moving) < 8:
        raise RuntimeError("走行中のデータが足りません (seconds を長くするか、通信状態を確認)")
    t = np.array([s.ms for s in moving], dtype=float) / 1000.0
    t -= t[0]
    keep = t >= 0.15  # 加速中のサンプルを除く
    xy = np.array([cfg.mat_spec.dot_to_mm(s.x_dot, s.y_dot) for s in moving])[keep]
    ang = np.unwrap(np.radians([s.angle_deg for s in moving]))[keep]
    return {"p0": p0, "p1": p1, "t": t[keep], "xy": xy, "ang": ang}


def straight_test(bridge: MqttBridge, cfg: EnvConfig, unit: int = 30, seconds: float = 1.5) -> dict:
    """直進させて速度係数 (motor_gain)・左右差 (wheel_asymmetry)・座標系の向きを確認する。"""
    r = _drive(bridge, cfg, unit, unit, seconds)
    dist = np.linalg.norm(r["xy"] - r["xy"][0], axis=1)
    speed = float(np.polyfit(r["t"], dist, 1)[0])
    w = float(np.polyfit(r["t"], r["ang"], 1)[0])
    q = w * cfg.robot.wheel_base_mm / (2.0 * max(speed, 1e-6))  # (vl - vr) / (vl + vr)
    (x0, y0, th0), (x1, y1, _) = r["p0"], r["p1"]
    return {
        "distance_mm": math.hypot(x1 - x0, y1 - y0),
        "speed_mm_s": speed,
        "motor_gain": speed / (unit * cfg.robot.mm_per_s_per_unit),
        "wheel_asymmetry": (1 + q) / (1 - q) if abs(q) < 0.9 else float("nan"),
        # 前進方向と toio の角度の差。0° 付近でなければ座標系の想定が違う
        "heading_error_deg": math.degrees(wrap_angle(math.atan2(y1 - y0, x1 - x0) - th0)),
    }


def turn_test(bridge: MqttBridge, cfg: EnvConfig, unit: int = 20, seconds: float = 1.0) -> dict:
    """その場で右旋回 (左輪前進・右輪後退) させ、旋回の速さと向きを確認する。"""
    r = _drive(bridge, cfg, unit, -unit, seconds)
    w = float(np.polyfit(r["t"], r["ang"], 1)[0])
    w_nominal = 2.0 * unit * cfg.robot.mm_per_s_per_unit / cfg.robot.wheel_base_mm
    return {
        "turn_rate_deg_s": math.degrees(w),
        "turn_gain": w / w_nominal,  # 1.0 が公称。負なら旋回方向の想定が逆
    }


def compare_with_dr(
    cfg: EnvConfig,
    link: Optional[dict] = None,
    straight: Optional[dict] = None,
    turn: Optional[dict] = None,
    tof_empty: Optional[dict] = None,
) -> list[tuple[bool, str]]:
    """測定結果が DR 範囲に入っているかを (OK?, メッセージ) のリストで返す。"""
    dr, out = cfg.dr, []

    def check(name: str, value: float, rng: tuple[float, float], fmt: str = ".2f"):
        ok = rng[0] <= value <= rng[1]
        out.append((ok, f"{name}: 実測 {value:{fmt}} / DR 範囲 [{rng[0]:{fmt}}, {rng[1]:{fmt}}]"))

    if link:
        check("制御ループ遅延 p90 [s]", link["latency_ms_p90"] / 1000 + EXTRA_LATENCY_S, dr.latency_s)
        if link["state_hz"] < 15:
            out.append((False, f"状態の受信レートが低い ({link['state_hz']:.1f} Hz)。Wi-Fi / ブローカーを確認"))
    if straight:
        check("速度係数 motor_gain", straight["motor_gain"], dr.motor_gain)
        check("左右差 wheel_asymmetry", straight["wheel_asymmetry"], dr.wheel_asymmetry, ".3f")
        ok = abs(straight["heading_error_deg"]) < 20
        out.append((ok, f"前進方向と toio 角度の差: {straight['heading_error_deg']:+.1f}° (±20° 以内なら OK)"))
    if turn:
        check("旋回ゲイン turn_gain", turn["turn_gain"], dr.motor_gain)
    if tof_empty:
        med = tof_empty["median_mm"]
        ok = not (tof_empty["valid_ratio"] > 0.3 and med < cfg.tof_max_range_mm + 50)
        msg = f"障害物なしの ToF: 有効率 {tof_empty['valid_ratio']:.0%}, 中央値 {med:.0f} mm"
        if not ok:
            msg += f" → 床や周囲を検出している可能性。tof_max_range_mm を {max(med - 80, 100):.0f} 程度に下げて再学習"
        out.append((ok, msg))
    return out


def suggest_config(
    cfg: EnvConfig,
    link: Optional[dict] = None,
    straight: Optional[dict] = None,
    turn: Optional[dict] = None,
    tof_empty: Optional[dict] = None,
    margin: float = 0.1,
) -> EnvConfig:
    """測定値を含むように DR 範囲を広げた EnvConfig を返す (追加学習用)。"""
    new = copy.deepcopy(cfg)
    dr = new.dr

    def widen(rng: tuple[float, float], value: float) -> tuple[float, float]:
        if not math.isfinite(value):
            return rng
        lo, hi = rng
        span = (hi - lo) * margin
        return (min(lo, value - span), max(hi, value + span))

    if link:
        dr.latency_s = widen(dr.latency_s, link["latency_ms_p90"] / 1000 + EXTRA_LATENCY_S)
    if straight:
        dr.motor_gain = widen(dr.motor_gain, straight["motor_gain"])
        dr.wheel_asymmetry = widen(dr.wheel_asymmetry, straight["wheel_asymmetry"])
    if turn and turn["turn_gain"] > 0:
        dr.motor_gain = widen(dr.motor_gain, turn["turn_gain"])
    if tof_empty and tof_empty["valid_ratio"] > 0.3 and tof_empty["median_mm"] < cfg.tof_max_range_mm + 50:
        new.tof_max_range_mm = float(max(tof_empty["median_mm"] - 80, 100))
    return new
