"""matplotlib による可視化 (Colab 上でそのまま表示できる)。

座標はマットと同じく y 下向きなので、y 軸を反転して表示する。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, patches

from .config import EnvConfig
from .sim import Obstacles


def draw_mat(ax, cfg: EnvConfig, obstacles: Optional[Obstacles] = None) -> None:
    W, H = cfg.mat_spec.width_mm, cfg.mat_spec.height_mm
    ax.add_patch(patches.Rectangle((0, 0), W, H, fc="#f4f1e8", ec="#555", lw=1.5))
    m = cfg.edge_margin_mm
    ax.add_patch(patches.Rectangle((m, m), W - 2 * m, H - 2 * m, fc="none", ec="#bbb", lw=0.8, ls="--"))
    if obstacles is not None:
        for cx, cy, r in obstacles.circles:
            ax.add_patch(patches.Circle((cx, cy), r, fc="#6b6b6b", ec="#333"))
        for box in obstacles.boxes:
            ax.add_patch(patches.Polygon(box, closed=True, fc="#6b6b6b", ec="#333"))
    pad = 20
    ax.set_xlim(-pad, W + pad)
    ax.set_ylim(H + pad, -pad)  # y 下向き
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")


def plot_episode(result, cfg: EnvConfig, ax=None, title: Optional[str] = None):
    """evaluate.run_episode(record=True) の結果を 1 枚の図にする。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * cfg.mat_spec.height_mm / cfg.mat_spec.width_mm))
    draw_mat(ax, cfg, result.obstacles)
    tr = result.trace
    xs, ys = [s["x"] for s in tr], [s["y"] for s in tr]
    ax.plot(xs, ys, "-", color="#1f77b4", lw=1.5)
    ax.plot(xs[0], ys[0], "o", color="#1f77b4", ms=8, label="start")
    goals = list(dict.fromkeys(s["goal"] for s in tr))  # 出現順に重複除去
    for i, g in enumerate(goals):
        reached = i < len(goals) - 1
        ax.plot(*g, "*", ms=14, color="#2ca02c" if reached else "#d62728")
        ax.annotate(str(i + 1), g, textcoords="offset points", xytext=(6, -10), fontsize=8)
    if result.collided:
        ax.plot(xs[-1], ys[-1], "x", color="red", ms=14, mew=3, label="collision")
    status = "collision" if result.collided else "out of mat" if result.out_of_bounds else "ok"
    ax.set_title(title or f"goals={result.goals}  steps={result.steps}  ({status})")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def animate_episode(result, cfg: EnvConfig, max_frames: int = 400, every: int = 1):
    """エピソードをアニメーションにする。Colab では IPython.display.HTML(anim.to_jshtml()) で表示。"""
    tr = result.trace[: max_frames * every : every]
    fig, ax = plt.subplots(figsize=(5.5, 5.5 * cfg.mat_spec.height_mm / cfg.mat_spec.width_mm))
    draw_mat(ax, cfg, result.obstacles)
    path, = ax.plot([], [], "-", color="#1f77b4", lw=1)
    body = patches.Circle((0, 0), tr[0]["radius"], fc="#9ecae1", ec="#08519c", zorder=3)
    ax.add_patch(body)
    head, = ax.plot([], [], "-", color="#08519c", lw=2, zorder=4)
    cone = patches.Wedge((0, 0), 1, 0, 1, fc="#fdae6b", alpha=0.5, zorder=2)
    ax.add_patch(cone)
    goal, = ax.plot([], [], "*", ms=16, color="#d62728", zorder=5)
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace")
    off = cfg.robot.tof_offset_mm

    def update(i):
        s = tr[i]
        path.set_data([p["x"] for p in tr[: i + 1]], [p["y"] for p in tr[: i + 1]])
        body.center = (s["x"], s["y"])
        hx, hy = s["x"] + s["radius"] * math.cos(s["th"]), s["y"] + s["radius"] * math.sin(s["th"])
        head.set_data([s["x"], hx], [s["y"], hy])
        tof = s["tof"] if s["tof"] is not None else cfg.tof_max_range_mm
        center = math.degrees(s["th"]) + s["tof_angle_deg"]
        cone.set_center((s["x"] + off * math.cos(s["th"]), s["y"] + off * math.sin(s["th"])))
        cone.set_radius(min(tof, cfg.tof_max_range_mm))
        cone.set_theta1(center - s["tof_fov_deg"] / 2)
        cone.set_theta2(center + s["tof_fov_deg"] / 2)
        goal.set_data([s["goal"][0]], [s["goal"][1]])
        tof_txt = f"{s['tof']:.0f}mm" if s["tof"] is not None else "--"
        text.set_text(f"t={s['t']:5.1f}s  ToF={tof_txt:>6}  L={s['cmd'][0]:+4d} R={s['cmd'][1]:+4d}")
        return path, body, head, cone, goal, text

    anim = animation.FuncAnimation(fig, update, frames=len(tr), interval=cfg.control_dt_s * 1000 * every, blit=True)
    plt.close(fig)
    return anim


def plot_latency_sweep(rows: Sequence[dict], labels: Sequence[str] = (), dr_range: Optional[tuple] = None):
    """evaluate.latency_sweep の結果 (複数方策可) を比較する。rows は [rows_policy1, rows_policy2, ...]。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for i, rs in enumerate(rows):
        lat = [r["latency_s"] for r in rs]
        label = labels[i] if i < len(labels) else f"policy{i}"
        axes[0].plot(lat, [r["goals_per_min"] for r in rs], "o-", label=label)
        axes[1].plot(lat, [r["collision_rate"] + r["out_of_bounds_rate"] for r in rs], "o-", label=label)
    for ax, yl in zip(axes, ["goals / min", "failure rate (collision + out of mat)"]):
        if dr_range:
            ax.axvspan(*dr_range, color="#ddd", alpha=0.6, label="DR range")
        ax.set_xlabel("loop latency [s]")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    return fig


def plot_real_run(log: dict, cfg: EnvConfig, obstacles: Optional[Obstacles] = None):
    """real_robot.run_on_robot の記録 (RunLog.to_dict()) を表示する。

    obstacles: 仮想ロボットで試したときは VirtualRobot.obstacles を渡すと障害物も描く。
    """
    mat = cfg.mat_spec
    ok = log["pos_valid"].astype(bool)
    x_mm, y_mm = mat.dot_to_mm(log["x_dot"], log["y_dot"])
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    draw_mat(ax, cfg, obstacles)
    ax.plot(np.where(ok, x_mm, np.nan), np.where(ok, y_mm, np.nan), "-", lw=1.5)
    gx, gy = mat.dot_to_mm(log["goal_x_dot"], log["goal_y_dot"])
    ax.plot(gx, gy, "*", ms=14, color="#d62728", ls="none")
    clash = np.flatnonzero(np.diff(log["clash"].astype(int), prepend=0) > 0)
    ax.plot(x_mm[clash], y_mm[clash], "x", color="red", ms=12, mew=3, ls="none", label="clash")
    ax.set_title(f"trajectory (goals reached: {int(log['goals_reached'][-1])})")

    t = log["t"] - log["t"][0]
    ax1 = fig.add_subplot(3, 2, 2)
    tof = np.where(log["tof"] < 0, np.nan, log["tof"])
    ax1.plot(t, np.minimum(tof, cfg.tof_max_range_mm), lw=1)
    ax1.axhline(cfg.safety_stop_mm, color="red", lw=0.8, ls="--")
    ax1.set_ylabel("ToF [mm]")
    ax2 = fig.add_subplot(3, 2, 4, sharex=ax1)
    ax2.plot(t, log["left"], lw=1, label="L")
    ax2.plot(t, log["right"], lw=1, label="R")
    ax2.set_ylabel("cmd")
    ax2.legend(fontsize=7, loc="upper right")
    ax3 = fig.add_subplot(3, 2, 6, sharex=ax1)
    ax3.plot(t, log["latency_ms"], lw=1)
    ax3.axhline(cfg.dr.latency_s[1] * 1000, color="gray", lw=0.8, ls="--")
    ax3.set_ylabel("latency [ms]")
    ax3.set_xlabel("time [s]")
    fig.tight_layout()
    return fig
