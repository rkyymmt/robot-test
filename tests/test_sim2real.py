"""シミュレータと実機推論の共通部分のテスト。  python -m pytest tests -q"""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from toio_sim2real.config import MM_PER_DOT, EnvConfig
from toio_sim2real.core import F, ObservationBuilder, action_to_wheel_units, apply_safety_shield
from toio_sim2real.env import ToioAvoidEnv
from toio_sim2real.policies import RuleBasedPolicy
from toio_sim2real.protocol import decode_state, encode_motor, encode_state
from toio_sim2real.real_robot import GoalSequence, PolicyController
from toio_sim2real.sim import Obstacles, clearance, raycast


def nominal_cfg() -> EnvConfig:
    cfg = EnvConfig()
    cfg.dr.enabled = False
    return cfg


def test_gym_api():
    check_env(ToioAvoidEnv(), skip_render_check=True)


def test_raycast_and_clearance():
    obs = Obstacles([[100, 0, 10]], Obstacles.make_box(0, 200, 40, 20, 0.0)[None])
    d = raycast(np.zeros(2), np.array([0.0, math.pi / 2, math.pi]), obs, 1000.0)
    np.testing.assert_allclose(d, [90.0, 190.0, 1000.0])
    assert clearance(np.array([0.0, 150.0]), obs) == pytest.approx(40.0)
    assert clearance(np.array([0.0, 200.0]), obs) < 0  # 箱の内側


def test_straight_speed_matches_toio_spec():
    """公称パラメータで前進最大を 2 秒 → 速度指示値 45 × 2.81 mm/s ≒ 126 mm/s。"""
    env = ToioAvoidEnv(nominal_cfg())
    env.reset(seed=0, options={"layout": (Obstacles.empty(), (100.0, 280.0, 0.0))})
    for _ in range(20):
        env.step([1.0, 0.0])
    assert env.sim.x - 100.0 == pytest.approx(45 * 2.81 * 2.0, rel=0.1)
    assert env.sim.y == pytest.approx(280.0, abs=1.0)


def test_turn_direction_is_clockwise_on_mat():
    """turn > 0 は左輪が速い → マット上 (y 下向き) で heading が増える = 右旋回。"""
    cfg = nominal_cfg()
    left, right = action_to_wheel_units([0.0, 1.0], cfg)
    assert left > right
    env = ToioAvoidEnv(cfg)
    env.reset(seed=0, options={"layout": (Obstacles.empty(), (280.0, 280.0, 0.0))})
    env.step([-0.5, 0.5])
    assert env.sim.th > 0


def test_safety_shield_matches_firmware_rounding():
    assert apply_safety_shield(40, 20, 10.0, 30.0) == (10, -10, True)
    assert apply_safety_shield(-5, 20, 10.0, 30.0) == (-12, 12, True)  # C++ と同じく 0 方向へ丸め
    assert apply_safety_shield(40, 40, None, 30.0) == (40, 40, False)  # ToF 無効なら何もしない
    assert apply_safety_shield(-20, -20, 5.0, 30.0) == (-20, -20, False)  # 後退は許可


def test_shield_stops_robot_in_front_of_wall():
    env = ToioAvoidEnv(nominal_cfg())
    wall = Obstacles(np.zeros((0, 3)), Obstacles.make_box(300, 280, 20, 300, 0.0)[None])
    env.reset(seed=0, options={"layout": (wall, (200.0, 280.0, 0.0))})
    for _ in range(30):
        _, _, terminated, _, info = env.step([1.0, 0.0])
        assert not info["collided"]
    tof = env.sim.true_tof(env.sim.x, env.sim.y, env.sim.th)
    assert tof < nominal_cfg().safety_stop_mm + 10


def test_mat_conversion_roundtrip():
    mat = EnvConfig().mat_spec
    assert mat.width_mm == pytest.approx(410 * MM_PER_DOT)
    assert mat.mm_to_dot(*mat.dot_to_mm(123, 321)) == pytest.approx((123, 321))


def test_config_json_roundtrip(tmp_path):
    cfg = EnvConfig(mat="simple")
    cfg.dr.latency_s = (0.1, 0.9)
    cfg.save(tmp_path / "c.json")
    loaded = EnvConfig.load(tmp_path / "c.json")
    assert loaded == cfg
    assert isinstance(loaded.dr.latency_s, tuple)


def test_real_controller_builds_same_observation_as_builder():
    """実機側 (MQTT の state) と sim 側 (ObservationBuilder 直接) で同じ観測になること。"""
    cfg = nominal_cfg()
    mat = cfg.mat_spec
    goal_dot = (300, 200)
    st = decode_state(encode_state(1, 1000, 123, 150, 250, 90, True, False, 80, True, 0, 100, False, 0, 0), 0.0)

    seen = []

    class Spy:
        def predict(self, obs, deterministic=True):
            seen.append(obs)
            return np.zeros(2, dtype=np.float32), None

    ctrl = PolicyController(Spy(), cfg, GoalSequence([mat.dot_to_mm(*goal_dot)]))
    ctrl.step(st)

    b = ObservationBuilder(cfg)
    x, y = mat.dot_to_mm(150, 250)
    expected = b.update(123, (x, y, math.radians(90)), mat.dot_to_mm(*goal_dot), [0.0, 0.0])
    np.testing.assert_allclose(seen[0], expected)
    # toio が下 (90°) を向いていて、ゴールは右上 → ゴール方位は左 (負)
    assert seen[0][F["goal_sin"]] < 0


def test_protocol_roundtrip():
    st = decode_state(encode_state(5, 999, None, 100, 200, 359, False, True, 50, False, 7, 210, True, -10, 10), 1.5)
    assert st.tof_mm is None and st.angle_deg == 359 and not st.pos_valid and st.clash
    assert (st.ack, st.latency_ms, st.left, st.right) == (7, 210, -10, 10)
    assert encode_motor(3, 10, -10, 500, 1234) == b'{"seq":3,"l":10,"r":-10,"to":500,"sms":1234}'


def test_rule_based_policy_reaches_goals_without_obstacles():
    cfg = nominal_cfg()
    env = ToioAvoidEnv(cfg)
    obs, _ = env.reset(seed=3, options={"n_obstacles": 0})
    pol = RuleBasedPolicy(cfg)
    for _ in range(300):
        obs, _, term, trunc, info = env.step(pol.predict(obs)[0])
        if term or trunc:
            break
    assert info["goals_reached"] >= 2


def test_obstacle_memory_sectors():
    """正面で検出した障害物は、ロボットが 90° 左を向いた後も「右側」に記憶されている。"""
    cfg = nominal_cfg()
    b = ObservationBuilder(cfg)
    goal = (400.0, 300.0)
    obs = b.update(100.0, (200.0, 300.0, 0.0), goal, [0, 0])
    mem = obs[-cfg.memory_sectors:]
    front = cfg.memory_sectors // 2
    assert mem[front] == pytest.approx((100.0 + cfg.robot.tof_offset_mm) / cfg.memory_range_mm)
    assert np.all(np.delete(mem, front) == 1.0)
    # その場で左 (反時計回り, heading 減少) に 90° 向きを変え、ToF は何も見ていない
    obs = b.update(None, (200.0, 300.0, -math.pi / 2), goal, [0, 0])
    mem = obs[-cfg.memory_sectors:]
    right = front + cfg.memory_sectors // 4
    assert mem[right] < 1.0 and mem[front] == 1.0
    # memory_steps 経過すると忘れる
    for _ in range(cfg.memory_steps):
        obs = b.update(None, (200.0, 300.0, -math.pi / 2), goal, [0, 0])
    assert np.all(obs[-cfg.memory_sectors:] == 1.0)
