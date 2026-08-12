# -*- coding: utf-8 -*-
"""挥手判定调优仿真测试：合成 landmark 序列驱动 _calc_arm_horizontal_motion，
覆盖 正常挥手触发 / 慢速挥手 / 挠头·摆手·举手·单程指向 干扰排除 / 互斥场景。
"""
import sys
import time
import types
from collections import deque

import numpy as np

from motion_capture import MotionCapture

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def make_dummy():
    d = object.__new__(MotionCapture)
    d._wrist_xy_history = {'L': deque(maxlen=20), 'R': deque(maxlen=20)}
    d._elbow_xy_history = {'L': deque(maxlen=20), 'R': deque(maxlen=20)}
    d._wave_direction = 0.0
    d._debug = False
    return d


def mk_lm(shoulder=(0.5, 0.3), elbow=None, wrist=None,
          index=None, pinky=None, thumb=None, vis=0.9,
          evis=None, wvis=None, fvis=None):
    """构造 (33,4) landmark；只设置右手关键点，左手不可见。
    evis/wvis/fvis 可独立指定肘/腕/指可见性（默认跟随 vis），
    用于模拟"手贴近头部时 MediaPipe 对腕部评分低"的实机场景。"""
    evis = vis if evis is None else evis
    wvis = vis if wvis is None else wvis
    fvis = vis if fvis is None else fvis
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[11] = [0.5, 0.3, 0.0, vis]           # LEFT_SHOULDER（用于互斥/参考）
    lm[12] = [shoulder[0], shoulder[1], 0.0, vis]  # RIGHT_SHOULDER
    if elbow is not None:
        lm[14] = [elbow[0], elbow[1], 0.0, evis]   # RIGHT_ELBOW
    if wrist is not None:
        lm[16] = [wrist[0], wrist[1], 0.0, wvis]   # RIGHT_WRIST
    if index is not None:
        lm[20] = [index[0], index[1], 0.0, fvis]   # RIGHT_INDEX
    if pinky is not None:
        lm[18] = [pinky[0], pinky[1], 0.0, fvis]   # RIGHT_PINKY
    if thumb is not None:
        lm[22] = [thumb[0], thumb[1], 0.0, fvis]   # RIGHT_THUMB
    return lm


def hand_frame(elbow_x, elbow_y, spread_open=0.37):
    """右手：肘部在 (elbow_x, elbow_y)，肩 (0.5,0.3)，手掌张开 spread 稳定。"""
    e = (elbow_x, elbow_y)
    # 手指相对肘部固定偏移 → spread 恒定
    return mk_lm(elbow=e, wrist=(elbow_x + 0.02, elbow_y + 0.06),
                 index=(elbow_x + 0.12, elbow_y + 0.08),
                 pinky=(elbow_x + 0.09, elbow_y + 0.08),
                 thumb=(elbow_x - 0.08, elbow_y + 0.08))


def run_seq(dummy, seq):
    out = []
    for lm in seq:
        v = MotionCapture._calc_arm_horizontal_motion(dummy, lm, 0.04)
        out.append((v, dummy._wave_direction))
    return out


def peak(out):
    return max(v for v, _ in out) if out else 0.0


# ---------- W 系列：真实挥手必须触发 ----------
# W1 快速挥手：肘部 0.75→0.30→0.70 往返（横向跨度 0.45，肘部在肩下方）
pts = [(0.75 - i * 0.05, 0.42) for i in range(10)] + \
      [(0.30 + i * 0.05, 0.42) for i in range(8)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W1 快速挥手应触发(intensity>=0.2)", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# W2 反向挥手（左→右）
pts = [(0.30 + i * 0.05, 0.42) for i in range(10)] + \
      [(0.75 - i * 0.05, 0.42) for i in range(8)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W2 反向挥手应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# W3 慢速挥手 1Hz：单次去-回（0.65→0.38→0.62，跨度 0.27），覆盖窗口
pts = [(0.65 - i * 0.027, 0.42) for i in range(10)] + \
      [(0.38 + i * 0.027, 0.42) for i in range(9)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W3 慢速挥手(1Hz单次往返)应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# W4 挥手后停住（往返挥完停在终点，历史窗口仍含挥动段）
pts = [(0.75 - i * 0.05, 0.42) for i in range(8)] + \
      [(0.40 + i * 0.05, 0.42) for i in range(6)] + \
      [(0.65, 0.42)] * 6
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W4 挥完停住仍应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# W5 含噪声真实挥手（±0.02 噪声叠加）
import random as _rnd
_rnd.seed(7)
pts = [(0.72 - i * 0.04 + _rnd.uniform(-0.02, 0.02), 0.42 + _rnd.uniform(-0.01, 0.01))
       for i in range(11)] + \
      [(0.35 + i * 0.04 + _rnd.uniform(-0.02, 0.02), 0.42) for i in range(9)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W5 含噪声挥手应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# ---------- R 系列：干扰手势必须排除 ----------
# R1 纯竖向举手（x 不动，肘部从肩下举到肩上）→ 高度门 + x跨度不足
pts = [(0.55, 0.42 - i * 0.03) for i in range(10)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("R1 纯竖向举手不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# R2 举手后悬停（肘部在肩上方，横向小幅摆动）→ 高度门排除
pts = [(0.55 + (0.01 if i % 2 else -0.01), 0.15 + (0.01 if i % 2 else -0.01))
       for i in range(10)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("R2 举起悬停(肘过肩)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# R3 挠头：肘部抬到肩上方小幅弧线摆动 + spread 剧烈变化
def scratch_frame(elbow_x, elbow_y, spread):
    e = (elbow_x, elbow_y)
    off = spread / 3.0
    return mk_lm(elbow=e, wrist=(elbow_x + 0.02, elbow_y + 0.04),
                 index=(elbow_x + off, elbow_y + 0.06),
                 pinky=(elbow_x + off * 0.8, elbow_y + 0.06),
                 thumb=(elbow_x - off, elbow_y + 0.06))

seq = []
for i in range(12):
    seq.append(scratch_frame(0.55 + 0.02 * np.sin(i * 0.8),
                             0.12 + 0.02 * np.cos(i * 0.8),
                             0.10 + 0.15 * (i % 2)))  # spread 0.10/0.25 交替
r = run_seq(make_dummy(), seq)
check("R3 挠头(肘过肩+spread抖动)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# R4 上下摆手/拍手：肘部在肩下，纵向主导（y 大幅摆动，x 小幅）
pts = [(0.55 + (0.01 if i % 2 else -0.01), 0.45 - i * 0.02) for i in range(12)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("R4 上下摆手(纵向主导)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# R5 单程指向/摊手：x 只朝一个方向移动（0 翻转）
pts = [(0.75 - i * 0.03, 0.42) for i in range(12)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("R5 单程指向(无往返)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# R6 手指动作（弹指/挠手）：肘部几乎不动，spread 剧烈变化
def finger_frame(spread):
    return scratch_frame(0.60, 0.42, spread)

seq = [finger_frame(0.08 + 0.20 * (i % 2)) for i in range(12)]
r = run_seq(make_dummy(), seq)
check("R6 手指动作(spread抖动)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# ---------- 互斥场景 ----------
# M1 挥手期间举手应=0：先挥，再举起肘部过肩
pts = [(0.75 - i * 0.05, 0.42) for i in range(8)] + \
      [(0.40 - i * 0.015, 0.42 - i * 0.03) for i in range(6)]  # 边举边回
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("M1 挥手后转举手 最后应停触发", r[-1][0] < 0.2, f"last={r[-1][0]:.2f}")

# M2 静止举手（悬停）不应触发挥手
pts = [(0.55, 0.15)] * 12
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("M2 静止举手不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# ---------- 锤桌场景（实机误触回归：肘部绕肩弧线，x/y 跨度都大，翻转 x=1/y=1 平手） ----------
# H1 锤桌：肘部绕肩大幅转动（角度跨度 ~5 rad，跨x/跨y 都大）
# → 角度跨度上限（2.5 rad）排除（实机锤桌角度跨度 2.8~4.9 rad）
import math as _m
SH = (0.5, 0.3)
pts = []
for i in range(20):
    th = -2.5 + i * (5.0 / 19)   # 从 -2.5 到 +2.5 rad，跨 5 rad
    pts.append((SH[0] + 0.25 * _m.cos(th), SH[1] + 0.25 * _m.sin(th)))
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("H1 锤桌(角度跨度>2.5rad)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# H4 平手弧线挥手：翻转 x=1/y=1 平手，但角度跨度正常（~1.3 rad）
# → 平手放行，应触发（真实弧线挥手去-回弧线常出现 x/y 翻转持平）
pts = [(0.62 - i * 0.035, 0.40 + i * 0.03) for i in range(6)] + \
      [(0.45 + i * 0.035, 0.58 - i * 0.03) for i in range(6)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("H4 平手弧线挥手(翻转x=1/y=1)应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# H2 锤桌横纵跨度接近（x=0.38/y=0.27，y/x=0.71）→ 翻转平手排除
pts = [(0.55 - i * 0.015, 0.45 - i * 0.012) for i in range(6)] + \
      [(0.47 + i * 0.015, 0.38 + i * 0.012) for i in range(6)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("H2 锤桌大弧线(y/x≈0.7)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# H3 连续快速锤桌（多次纵向往复，y 翻转 > x 翻转）→ 排除
pts = []
for i in range(16):
    phase = i % 4
    y = 0.45 - (0.06 if phase < 2 else 0.0)  # 纵向往复
    x = 0.52 + (0.01 if i % 2 else -0.01)
    pts.append((x, y))
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("H3 连续锤桌(纵向主导)不应触发", peak(r) < 0.2, f"peak={peak(r):.2f}")

# ---------- 自定义预设匹配验证（gesture_presets.PresetEngine） ----------
import gesture_presets as gp

def make_engine_with_custom():
    eng = gp.PresetEngine()
    # 构造自定义预设：duration=0.3，baseline 静止，action 挥手模板
    preset = gp.GesturePreset(
        name="自定义挥掌",
        description="test",
        conditions={
            "duration": 0.3,
            "baseline": {"arm_horizontal_motion": 0.05, "motion_energy": 0.05},
            "action": {"arm_horizontal_motion": 0.55, "motion_energy": 0.35},
            "action_std": {"arm_horizontal_motion": 0.05, "motion_energy": 0.05},
            "trajectory": [[0.0, 0.0], [0.5, 0.1], [1.0, 0.0], [0.5, -0.1]],
        },
        action="custom_play",
        priority=20,
    )
    preset._is_custom = True
    eng.add_preset(preset)
    return eng


def run_match(eng, features_list, dt=0.05):
    """逐帧喂入特征，返回触发动作序列（timestamp 递增，使 hold_timer 正常累积）。"""
    outs = []
    ts = 0.0
    for feats in features_list:
        f = dict(feats)
        f["timestamp"] = ts
        outs.append(eng.match(f))
        ts += dt
    return outs


# C1 做挥手动作（wave 信号激活）→ 自定义应让位（route=wave 优先，不串台）
eng = make_engine_with_custom()
feats = [{"arm_horizontal_motion": 0.6, "motion_energy": 0.4,
          "wave_direction": -0.8, "palm_hover": 0.0,
          "posture": "unknown", "hand_position": "unknown",
          "arm_trajectory": [[0.0, 0.0], [0.4, 0.1], [0.9, 0.0], [0.45, -0.1]]}
         for _ in range(8)]  # 0.4s > duration 0.3
outs = run_match(eng, feats)
check("C1 挥手时自定义让位(不串台)", "custom_play" not in outs, f"outs={outs}")

# C2 静止（特征≈baseline）→ 不应触发（d_baseline ≤ 1σ）
eng = make_engine_with_custom()
feats = [{"arm_horizontal_motion": 0.05, "motion_energy": 0.05,
          "wave_direction": 0.0, "palm_hover": 0.0,
          "posture": "unknown", "hand_position": "unknown",
          "arm_trajectory": []} for _ in range(8)]
outs = run_match(eng, feats)
check("C2 静止不应触发自定义", "custom_play" not in outs, f"outs={outs}")

# C3 短促动作（0.15s < duration 0.3）→ 不应触发（旧 duration=1.0 永远达不到的场景已放宽）
eng = make_engine_with_custom()
feats = [{"arm_horizontal_motion": 0.6, "motion_energy": 0.4,
          "wave_direction": 0.7, "palm_hover": 0.0,
          "posture": "unknown", "hand_position": "unknown",
          "arm_trajectory": [[0.0, 0.0], [0.4, 0.1], [0.9, 0.0], [0.45, -0.1]]}
         for _ in range(3)]  # 0.15s
outs = run_match(eng, feats)
check("C3 短促动作(0.15s<0.3s)不应触发", "custom_play" not in outs, f"outs={outs}")

# C4 删除后重录链路：remove_preset 后 add_preset 同名应成功（引擎同步清理）
eng = make_engine_with_custom()
eng.remove_preset("自定义挥掌")
try:
    eng.add_preset(gp.GesturePreset(
        name="自定义挥掌", description="test",
        conditions={"duration": 0.3,
                    "baseline": {"arm_horizontal_motion": 0.05, "motion_energy": 0.05},
                    "action": {"arm_horizontal_motion": 0.55, "motion_energy": 0.35},
                    "action_std": {"arm_horizontal_motion": 0.05, "motion_energy": 0.05}},
        action="custom_play", priority=20))
    eng.presets[-1]._is_custom = True
    ok = len(eng.presets) == 1
except ValueError:
    ok = False
check("C4 删除后同名重录应成功注入引擎", ok, f"presets={len(eng.presets)}")

# ---------- 静态/动态有机结合验证（动态度 D 加权融合） ----------
import custom_preset as cp

# S1 静态录制：举掌悬停（构型稳定、轨迹幅度小、能量平稳）→ 应产出低 D + pose 模板 + 长 duration
rec = cp.PresetRecorder(duration_sec=3.0)
rec.start()  # 进入倒计时；手动推进状态到录制
rec.state = rec.STATE_RECORDING
rec.recording_start = time.time()
STATIC_FEAT = {
    "motion_energy": 0.02, "arm_motion_energy": 0.03, "body_speed": 0.01,
    "arm_amplitude": 0.05, "body_rhythm": 0.01, "arm_horizontal_motion": 0.0,
    "palm_hover": 0.1, "wave_direction": 0.0,
    # 五段构型：头部/胸腔/大臂/小臂/手部
    "pose_head_dx": 0.0, "pose_head_dy": -0.25,
    "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,
    "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
    "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
    "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
    "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
    "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
    "pose_hand_dx_R": 0.26, "pose_hand_dy_R": -0.15,
    "pose_spread_L": 0.33, "pose_spread_R": 0.35,
    "posture": "unknown", "hand_position": "unknown", "is_active": False,
    "arm_trajectory": [[0.5, 0.4], [0.51, 0.41], [0.5, 0.4]],  # 幅度极小
}
for _ in range(30):
    rec.feed(dict(STATIC_FEAT))  # 30 帧
conds_s = rec.generate_conditions()
check("S1 静态录制 → dynamism 低(<0.3)",
      conds_s.get("dynamism", 1.0) < 0.3,
      f"D={conds_s.get('dynamism')}")
check("S1 静态录制 → 有 pose 模板",
      isinstance(conds_s.get("pose"), dict) and len(conds_s["pose"]) > 0,
      f"pose_keys={list(conds_s.get('pose', {}).keys())[:3]}")
check("S1 静态录制 → duration 偏长(>0.7)",
      conds_s.get("duration", 0.0) > 0.7, f"duration={conds_s.get('duration')}")

# S2 动态录制：挥手（轨迹幅度大、能量波动）→ 应产出高 D + 短 duration
rec2 = cp.PresetRecorder(duration_sec=3.0)
rec2.start()
rec2.state = rec2.STATE_RECORDING
rec2.recording_start = time.time()
import random as _rnd2
_rnd2.seed(3)
for i in range(30):
    # 真实挥手：肘部横扫幅度大（x 跨 0.3），轨迹覆盖大范围
    x = 0.30 + (i % 6) * 0.06          # 0.30~0.60 横向往复
    traj = [[0.30, 0.42], [0.45, 0.44], [0.60, 0.42], [0.45, 0.40]]
    fe = {
        "motion_energy": 0.25 + _rnd2.uniform(-0.08, 0.08),
        "arm_motion_energy": 0.4, "body_speed": 0.05,
        "arm_amplitude": 0.3, "body_rhythm": 0.05,
        "arm_horizontal_motion": 0.5, "palm_hover": 0.0,
        "wave_direction": -0.8,
        "pose_head_dx": 0.0, "pose_head_dy": -0.24,
        "pose_torso_dx": 0.0, "pose_torso_dy": 0.36,
        "pose_upper_dx_L": -0.14, "pose_upper_dy_L": -0.09,
        "pose_upper_dx_R": 0.16, "pose_upper_dy_R": -0.08,
        "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.06,
        "pose_fore_dx_R": 0.10, "pose_fore_dy_R": -0.04,
        "pose_hand_dx_L": -0.20, "pose_hand_dy_L": -0.10,
        "pose_hand_dx_R": 0.30, "pose_hand_dy_R": 0.0,
        "pose_spread_L": 0.3, "pose_spread_R": 0.32,
        "posture": "unknown", "hand_position": "unknown", "is_active": True,
        "arm_trajectory": traj,
    }
    rec2.feed(fe)
conds_d = rec2.generate_conditions()
check("S2 动态录制 → dynamism 高(>0.7)",
      conds_d.get("dynamism", 0.0) > 0.7,
      f"D={conds_d.get('dynamism')}")
check("S2 动态录制 → duration 短(<0.5)",
      conds_d.get("duration", 1.0) < 0.5, f"duration={conds_d.get('duration')}")

# S3 静态预设：保持构型 1.2s → 触发 custom_play
eng3 = gp.PresetEngine()
sp = gp.GesturePreset(name="静态举掌", description="test",
                      conditions=dict(conds_s), action="custom_play", priority=20)
sp._is_custom = True
eng3.add_preset(sp)
hold_feats = []
for _ in range(24):  # 1.2s
    f = dict(STATIC_FEAT)
    f["arm_trajectory"] = [[0.5, 0.4], [0.51, 0.41], [0.5, 0.4]]
    hold_feats.append(f)
outs3 = run_match(eng3, hold_feats)
check("S3 静态保持 1.2s 应触发 custom_play", "custom_play" in outs3, f"outs={outs3}")

# S4 静态预设：保持 0.4s（不足 duration）→ 不应触发
eng4 = gp.PresetEngine()
sp4 = gp.GesturePreset(name="静态举掌", description="test",
                       conditions=dict(conds_s), action="custom_play", priority=20)
sp4._is_custom = True
eng4.add_preset(sp4)
outs4 = run_match(eng4, hold_feats[:8])  # 0.4s
check("S4 静态保持 0.4s(不足)不应触发", "custom_play" not in outs4, f"outs={outs4}")

# S5 静态预设：挥手动作（构型/轨迹都不吻合）→ 不应触发
eng5 = gp.PresetEngine()
sp5 = gp.GesturePreset(name="静态举掌", description="test",
                       conditions=dict(conds_s), action="custom_play", priority=20)
sp5._is_custom = True
eng5.add_preset(sp5)
wave_feats = []
for i in range(16):
    x = 0.35 + (i % 6) * 0.04
    wave_feats.append({
        "motion_energy": 0.3, "arm_motion_energy": 0.4, "body_speed": 0.05,
        "arm_amplitude": 0.3, "body_rhythm": 0.05,
        "arm_horizontal_motion": 0.5, "palm_hover": 0.0, "wave_direction": -0.8,
        "pose_head_dx": 0.1, "pose_head_dy": -0.10,   # 构型明显不同（头部偏移）
        "pose_torso_dx": 0.2, "pose_torso_dy": 0.1,
        "pose_upper_dx_L": -0.30, "pose_upper_dy_L": 0.15,
        "pose_upper_dx_R": 0.35, "pose_upper_dy_R": 0.1,
        "pose_fore_dx_L": -0.25, "pose_fore_dy_L": 0.2,
        "pose_fore_dx_R": 0.30, "pose_fore_dy_R": 0.15,
        "pose_hand_dx_L": -0.35, "pose_hand_dy_L": 0.2,
        "pose_hand_dx_R": 0.45, "pose_hand_dy_R": 0.1,
        "pose_spread_L": 0.2, "pose_spread_R": 0.25,
        "posture": "unknown", "hand_position": "unknown", "is_active": True,
        "arm_trajectory": [[x, 0.4], [x + 0.05, 0.41], [x + 0.1, 0.4], [x + 0.05, 0.39]],
    })
outs5 = run_match(eng5, wave_feats)
check("S5 静态预设不应被挥手误触", "custom_play" not in outs5, f"outs={outs5}")

# S6 动态预设：无 wave 信号、且姿势与录制模板一致的动作 0.5s → 触发
#（五部分判断法下，回放姿势必须与录制捕获的部位一致才会命中；
#  姿势不一致应被拒绝——这正是"全部满足"要消除的误触行为）
eng6 = gp.PresetEngine()
dp = gp.GesturePreset(name="动态挥手", description="test",
                      conditions=dict(conds_d), action="custom_play", priority=20)
dp._is_custom = True
eng6.add_preset(dp)
no_wave_feats = []
for i in range(10):  # 0.5s
    f = dict(wave_feats[i])
    f["wave_direction"] = 0.0   # 去掉 wave 信号，验证自定义独立触发
    # 姿势对齐录制模板（conds_d 的 pose 均值），满足五段全部命中
    f.update({
        "pose_head_dx": 0.0, "pose_head_dy": -0.24,
        "pose_torso_dx": 0.0, "pose_torso_dy": 0.36,
        "pose_upper_dx_L": -0.14, "pose_upper_dy_L": -0.09,
        "pose_upper_dx_R": 0.16, "pose_upper_dy_R": -0.08,
        "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.06,
        "pose_fore_dx_R": 0.10, "pose_fore_dy_R": -0.04,
        "pose_hand_dx_L": -0.20, "pose_hand_dy_L": -0.10,
        "pose_hand_dx_R": 0.30, "pose_hand_dy_R": 0.0,
        "pose_spread_L": 0.3, "pose_spread_R": 0.32,
    })
    no_wave_feats.append(f)
outs6 = run_match(eng6, no_wave_feats)
check("S6 动态预设(无wave信号,姿势对齐)0.5s 应触发", "custom_play" in outs6, f"outs={outs6}")

# ---------- 本次专项验证：内置挥手切歌/上一首难触发修复 ----------
# W6 边界跨度挥手：横向跨度约 0.065（旧阈值 0.08 不触发，新阈值 0.06 触发）
pts = [(0.70 - i * 0.01625, 0.42) for i in range(8)] + \
      [(0.57 + i * 0.01625, 0.42) for i in range(6)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W6 边界跨度挥手(跨x≈0.065)应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# W7 肘部略高于肩（相对肩 y≈-0.10，介于旧 -0.08 与新 -0.12 之间）→ 应触发
pts = [(0.72 - i * 0.05, 0.20) for i in range(8)] + \
      [(0.37 + i * 0.05, 0.20) for i in range(6)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("W7 肘略高于肩(-0.10)挥手应触发", peak(r) >= 0.2, f"peak={peak(r):.2f}")

# N1 内置预设链路：加载 default_presets.json，span≈0.065（intensity 0.16 ≥ min 0.15）
# → 挥手切歌（player_next）与上一首（player_prev）应触发
import json as _json
engN = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engN.add_preset(pr)

def run_next_feats(dir_sign):
    feats = []
    for i in range(6):  # duration 0.05 → 3 帧即够
        feats.append({"arm_horizontal_motion": 0.16, "motion_energy": 0.0,
                      "body_speed": 0.0, "arm_amplitude": 0.0, "body_rhythm": 0.0,
                      "wave_direction": dir_sign, "palm_hover": 0.0,
                      "posture": "unknown", "hand_position": "unknown",
                      "arm_trajectory": []})
    return run_match(engN, feats)

outsN = run_next_feats(0.8)
check("N1 内置挥手切歌(player_next)应触发", "player_next" in outsN, f"outs={outsN}")
outsN2 = run_next_feats(-0.8)
check("N2 内置挥手上一首(player_prev)应触发", "player_prev" in outsN2, f"outs={outsN2}")
# N3 静止（intensity 0.04 < min 0.15）→ 不应触发内置切歌
outsN3 = run_next_feats(0.8) if False else None
engN2 = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engN2.add_preset(pr)
feats0 = [{"arm_horizontal_motion": 0.04, "motion_energy": 0.0, "body_speed": 0.0,
           "arm_amplitude": 0.0, "body_rhythm": 0.0, "wave_direction": 0.0,
           "palm_hover": 0.0, "posture": "unknown", "hand_position": "unknown",
           "arm_trajectory": []} for _ in range(6)]
outsN3 = run_match(engN2, feats0)
check("N3 静止不应触发内置切歌", "player_next" not in outsN3 and "player_prev" not in outsN3,
      f"outs={outsN3}")

# ---------- 本次修复专项验证：连贯切歌 + 自定义不串台 ----------
# X1 连续两次挥手（间隔 0.5s > 冷却 0.45s）→ 应触发两次 player_next（连贯切歌）
def wave_frame_burst(t, arm_motion=0.3, direction=0.8):
    return {"timestamp": t, "arm_horizontal_motion": arm_motion,
            "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
            "body_rhythm": 0.0, "wave_direction": direction, "palm_hover": 0.0,
            "posture": "unknown", "hand_position": "unknown", "arm_trajectory": []}

engX = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engX.add_preset(pr)

# 第一挥：t=0~0.15s；间隙 0.5s；第二挥：t=0.65~0.8s
seq = []
for i in range(4):
    seq.append(wave_frame_burst(i * 0.05))
seq.append(wave_frame_burst(0.65))
seq.append(wave_frame_burst(0.70))
seq.append(wave_frame_burst(0.75))
seq.append(wave_frame_burst(0.80))
outsX = run_match(engX, seq)
next_count = outsX.count("player_next")
check("X1 连续两次挥手应触发两次 player_next", next_count >= 2,
      f"next_count={next_count} outs={outsX}")

# X2 自定义不串台：做挥手动作（wave_direction 激活）时，自定义预设不应抢占
engX2 = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engX2.add_preset(pr)
# 注入一个"动作相似"的自定义预设（arm_horizontal_motion 接近挥手的动作模板）
cx = gp.GesturePreset(name="相似自定义", description="test",
                      conditions={"duration": 0.3,
                                  "baseline": {"arm_horizontal_motion": 0.05},
                                  "action": {"arm_horizontal_motion": 0.5},
                                  "action_std": {"arm_horizontal_motion": 0.05}},
                      action="custom_play", priority=20)
cx._is_custom = True
engX2.add_preset(cx)
seq2 = [wave_frame_burst(i * 0.05, arm_motion=0.5) for i in range(8)]
outsX2 = run_match(engX2, seq2)
check("X2 挥手时自定义不应抢占(route=wave 优先)", "custom_play" not in outsX2,
      f"outs={outsX2}")
check("X2 挥手仍应触发 player_next", "player_next" in outsX2, f"outs={outsX2}")

# X3 自定义动作（无 wave 信号）仍应正常触发
engX3 = gp.PresetEngine()
cx3 = gp.GesturePreset(name="拍手自定义", description="test",
                       conditions={"duration": 0.3,
                                   "baseline": {"arm_horizontal_motion": 0.05},
                                   "action": {"arm_horizontal_motion": 0.5},
                                   "action_std": {"arm_horizontal_motion": 0.05}},
                       action="custom_play", priority=20)
cx3._is_custom = True
engX3.add_preset(cx3)
seq3 = [{"timestamp": i * 0.05, "arm_horizontal_motion": 0.5, "motion_energy": 0.3,
         "body_speed": 0.0, "arm_amplitude": 0.0, "body_rhythm": 0.0,
         "wave_direction": 0.0, "palm_hover": 0.0, "posture": "unknown",
         "hand_position": "unknown", "arm_trajectory": []} for i in range(8)]
outsX3 = run_match(engX3, seq3)
check("X3 无挥手信号的自定义动作仍应触发", "custom_play" in outsX3, f"outs={outsX3}")

# ---------- 本次方向修复专项验证：首段方向判定 + 收手不反向 ----------
# D1 首段方向：向右挥出（x 增大）后收手回程（x 减小），
# 方向应保持 +1（首段挥出方向），收手回程不把方向带反
d = make_dummy()
pts_r = [(0.40 + i * 0.05, 0.42) for i in range(8)]   # 向右挥出
pts_back = [(0.75 - i * 0.05, 0.42) for i in range(6)] # 收手向左回程
seq_r = [hand_frame(x, y) for x, y in (pts_r + pts_back)]
r = run_seq(d, seq_r)
peak_r = max(v for v, _ in r)
last_dir = r[-1][1]
check("D1 右挥+收手: 方向应保持 +1(不回程带反)", last_dir == 1.0,
      f"last_dir={last_dir}")
check("D1 右挥+收手: 仍应触发(intensity达标)", peak_r >= 0.2, f"peak={peak_r:.2f}")

# D2 向左挥出后收手向右回程 → 方向应保持 -1
d = make_dummy()
pts_l = [(0.75 - i * 0.05, 0.42) for i in range(8)]   # 向左挥出
pts_back2 = [(0.40 + i * 0.05, 0.42) for i in range(6)] # 收手向右回程
r = run_seq(d, [hand_frame(x, y) for x, y in (pts_l + pts_back2)])
check("D2 左挥+收手: 方向应保持 -1(不回程带反)", r[-1][1] == -1.0,
      f"last_dir={r[-1][1]}")

# D3 引擎层：右挥触发 player_next，随后向左回程不应触发 player_prev
engD = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engD.add_preset(pr)
seq_d = []
ts = 0.0
for i in range(8):   # 右挥
    seq_d.append({"timestamp": ts, "arm_horizontal_motion": 0.5,
                  "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                  "body_rhythm": 0.0, "wave_direction": 0.8, "palm_hover": 0.0,
                  "posture": "unknown", "hand_position": "unknown",
                  "arm_trajectory": []})
    ts += 0.05
for i in range(6):   # 收手回程（方向应保持 0.8，不触发 prev）
    seq_d.append({"timestamp": ts, "arm_horizontal_motion": 0.3,
                  "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                  "body_rhythm": 0.0, "wave_direction": 0.8, "palm_hover": 0.0,
                  "posture": "unknown", "hand_position": "unknown",
                  "arm_trajectory": []})
    ts += 0.05
outsD = run_match(engD, seq_d)
check("D3 右挥应触发 player_next", "player_next" in outsD, f"outs={outsD}")
check("D3 收手回程不应触发 player_prev", "player_prev" not in outsD, f"outs={outsD}")

# D4 不同方向正确：右挥后间隔 >0.9s 再左挥 → 先 next 后 prev
engD2 = gp.PresetEngine()
with open("presets/default_presets.json", encoding="utf-8") as _f:
    for p in _json.load(_f)["presets"]:
        pr = gp.GesturePreset(name=p["name"], description=p.get("description", ""),
                              conditions=p["conditions"], action=p["action"],
                              priority=p.get("priority", 10))
        engD2.add_preset(pr)
seq_d2 = []
ts = 0.0
for i in range(8):   # 右挥 → next
    seq_d2.append({"timestamp": ts, "arm_horizontal_motion": 0.5,
                   "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                   "body_rhythm": 0.0, "wave_direction": 0.8, "palm_hover": 0.0,
                   "posture": "unknown", "hand_position": "unknown",
                   "arm_trajectory": []})
    ts += 0.05
ts += 1.0            # 间隔 1.0s（> 反向冷却 0.9s）
for i in range(8):   # 左挥 → prev
    seq_d2.append({"timestamp": ts, "arm_horizontal_motion": 0.5,
                   "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                   "body_rhythm": 0.0, "wave_direction": -0.8, "palm_hover": 0.0,
                   "posture": "unknown", "hand_position": "unknown",
                   "arm_trajectory": []})
    ts += 0.05
outsD2 = run_match(engD2, seq_d2)
check("D4 右挥触发 player_next", "player_next" in outsD2, f"outs={outsD2}")
check("D4 间隔后左挥触发 player_prev", "player_prev" in outsD2, f"outs={outsD2}")

# ---------- 本次修复专项验证：举手暂停不串自定义 + 向前指仍触发 ----------
# E1 交警掌心向前（palm_hover ≥ 0.5，构型与"向前指"相似）→ 自定义不应触发
#（_match_template 的 palm_hover 互斥 + match 路由 hover 优先双保险）
engE = gp.PresetEngine()
cxE = gp.GesturePreset(name="抬手向前指", description="test",
                       conditions={"duration": 0.3,
                                   "baseline": {"pose_hand_dx_R": 0.02},
                                   "pose": {"pose_head_dx": 0.0, "pose_head_dy": -0.24,
                                            "pose_torso_dx": 0.0, "pose_torso_dy": 0.34,
                                            "pose_upper_dx_R": 0.18, "pose_upper_dy_R": -0.22,
                                            "pose_fore_dx_R": 0.20, "pose_fore_dy_R": 0.05,
                                            "pose_hand_dx_R": 0.38, "pose_hand_dy_R": -0.17,
                                            "pose_spread_R": 0.05},
                                   "pose_std": {"pose_head_dx": 0.05, "pose_head_dy": 0.05,
                                                "pose_torso_dx": 0.05, "pose_torso_dy": 0.05,
                                                "pose_upper_dx_R": 0.05, "pose_upper_dy_R": 0.05,
                                                "pose_fore_dx_R": 0.05, "pose_fore_dy_R": 0.05,
                                                "pose_hand_dx_R": 0.05, "pose_hand_dy_R": 0.05,
                                                "pose_spread_R": 0.05},
                                   "dynamism": 0.0},
                       action="custom_play", priority=20)
cxE._is_custom = True
engE.add_preset(cxE)
# 交警掌心向前帧：palm_hover=1.0，构型与向前指相似（五段接近）
police_feats = [{"timestamp": i * 0.05, "palm_hover": 1.0, "wave_direction": 0.0,
                 "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                 "body_rhythm": 0.0, "arm_horizontal_motion": 0.0,
                 "pose_head_dx": 0.01, "pose_head_dy": -0.23,
                 "pose_torso_dx": 0.01, "pose_torso_dy": 0.33,
                 "pose_upper_dx_R": 0.19, "pose_upper_dy_R": -0.21,
                 "pose_fore_dx_R": 0.19, "pose_fore_dy_R": 0.06,
                 "pose_hand_dx_R": 0.39, "pose_hand_dy_R": -0.16,
                 "pose_spread_R": 0.06,
                 "posture": "unknown", "hand_position": "unknown",
                 "arm_trajectory": []} for i in range(8)]
outsE = run_match(engE, police_feats)
check("E1 交警掌心向前不应触发自定义", "custom_play" not in outsE, f"outs={outsE}")

# E2 抬手向前指（palm_hover < 0.5，构型匹配模板）→ 自定义应触发
engE2 = gp.PresetEngine()
cxE2 = gp.GesturePreset(name="抬手向前指", description="test",
                        conditions={"duration": 0.3,
                                    "baseline": {"pose_hand_dx_R": 0.02},
                                    "pose": {"pose_head_dx": 0.0, "pose_head_dy": -0.24,
                                             "pose_torso_dx": 0.0, "pose_torso_dy": 0.34,
                                             "pose_upper_dx_R": 0.18, "pose_upper_dy_R": -0.22,
                                             "pose_fore_dx_R": 0.20, "pose_fore_dy_R": 0.05,
                                             "pose_hand_dx_R": 0.38, "pose_hand_dy_R": -0.17,
                                             "pose_spread_R": 0.05},
                                    "pose_std": {"pose_head_dx": 0.05, "pose_head_dy": 0.05,
                                                 "pose_torso_dx": 0.05, "pose_torso_dy": 0.05,
                                                 "pose_upper_dx_R": 0.05, "pose_upper_dy_R": 0.05,
                                                 "pose_fore_dx_R": 0.05, "pose_fore_dy_R": 0.05,
                                                 "pose_hand_dx_R": 0.05, "pose_hand_dy_R": 0.05,
                                                 "pose_spread_R": 0.05},
                                    "dynamism": 0.0},
                        action="custom_play", priority=20)
cxE2._is_custom = True
engE2.add_preset(cxE2)
point_feats = [{"timestamp": i * 0.05, "palm_hover": 0.0, "wave_direction": 0.0,
                "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
                "body_rhythm": 0.0, "arm_horizontal_motion": 0.0,
                "pose_head_dx": 0.0, "pose_head_dy": -0.25,
                "pose_torso_dx": 0.0, "pose_torso_dy": 0.33,
                "pose_upper_dx_R": 0.19, "pose_upper_dy_R": -0.21,
                "pose_fore_dx_R": 0.19, "pose_fore_dy_R": 0.06,
                "pose_hand_dx_R": 0.37, "pose_hand_dy_R": -0.16,
                "pose_spread_R": 0.06,
                "posture": "unknown", "hand_position": "unknown",
                "arm_trajectory": []} for i in range(8)]
outsE2 = run_match(engE2, point_feats)
check("E2 抬手向前指(palm_hover低)应触发自定义", "custom_play" in outsE2, f"outs={outsE2}")

# E3 hover 路由保持：palm_hover 曾激活（≥0.5），随后几帧瞬时失败（<0.5），
# 保持窗口（0.5s）内路由仍为 hover → 自定义不触发（防抖动帧串台）
engE3 = gp.PresetEngine()
cxE3 = gp.GesturePreset(name="抬手向前指", description="test",
                        conditions={"duration": 0.3,
                                    "baseline": {"pose_hand_dx_R": 0.02},
                                    "pose": {"pose_head_dx": 0.0, "pose_head_dy": -0.24,
                                             "pose_torso_dx": 0.0, "pose_torso_dy": 0.34,
                                             "pose_upper_dx_R": 0.18, "pose_upper_dy_R": -0.22,
                                             "pose_fore_dx_R": 0.20, "pose_fore_dy_R": 0.05,
                                             "pose_hand_dx_R": 0.38, "pose_hand_dy_R": -0.17,
                                             "pose_spread_R": 0.05},
                                    "pose_std": {"pose_head_dx": 0.05, "pose_head_dy": 0.05,
                                                 "pose_torso_dx": 0.05, "pose_torso_dy": 0.05,
                                                 "pose_upper_dx_R": 0.05, "pose_upper_dy_R": 0.05,
                                                 "pose_fore_dx_R": 0.05, "pose_fore_dy_R": 0.05,
                                                 "pose_hand_dx_R": 0.05, "pose_hand_dy_R": 0.05,
                                                 "pose_spread_R": 0.05},
                                    "dynamism": 0.0},
                        action="custom_play", priority=20)
cxE3._is_custom = True
engE3.add_preset(cxE3)
hover_hold_feats = []
for i in range(14):
    # 前 6 帧 palm_hover 激活（1.0），后 8 帧抖动失败（0.3），但构型始终相似
    ph = 1.0 if i < 6 else 0.3
    hover_hold_feats.append({"timestamp": i * 0.05, "palm_hover": ph,
                             "wave_direction": 0.0, "motion_energy": 0.0,
                             "body_speed": 0.0, "arm_amplitude": 0.0,
                             "body_rhythm": 0.0, "arm_horizontal_motion": 0.0,
                             "pose_head_dx": 0.01, "pose_head_dy": -0.23,
                             "pose_torso_dx": 0.01, "pose_torso_dy": 0.33,
                             "pose_upper_dx_R": 0.19, "pose_upper_dy_R": -0.21,
                             "pose_fore_dx_R": 0.19, "pose_fore_dy_R": 0.06,
                             "pose_hand_dx_R": 0.39, "pose_hand_dy_R": -0.16,
                             "pose_spread_R": 0.06,
                             "posture": "unknown", "hand_position": "unknown",
                             "arm_trajectory": []})
outsE3 = run_match(engE3, hover_hold_feats)
check("E3 hover保持期内抖动帧不串自定义", "custom_play" not in outsE3, f"outs={outsE3}")

# ---------- 本次五段升级专项验证：五段全命中触发 / 单段偏离不触发 ----------
# 构造一个完整的五段静态预设（头部/胸腔/大臂/小臂/手部全部采样）
FULL_POSE = {
    "pose_head_dx": 0.0, "pose_head_dy": -0.25,
    "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,
    "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
    "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
    "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
    "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
    "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
    "pose_hand_dx_R": 0.26, "pose_hand_dy_R": -0.15,
    "pose_spread_L": 0.33, "pose_spread_R": 0.35,
}
FULL_STD = {k: 0.05 for k in FULL_POSE}

def make_five_seg_engine():
    eng = gp.PresetEngine()
    p = gp.GesturePreset(name="五段姿势", description="test",
                         conditions={"duration": 0.3,
                                     "action": {},   # 必须有 action dict 才走 _match_template
                                     "baseline": {"pose_hand_dx_R": 0.02},
                                     "pose": dict(FULL_POSE),
                                     "pose_std": dict(FULL_STD),
                                     "dynamism": 0.0},
                         action="custom_play", priority=20)
    p._is_custom = True
    eng.add_preset(p)
    return eng

def five_seg_frame(palm=0.0, **overrides):
    f = {"timestamp": 0.0, "palm_hover": palm, "wave_direction": 0.0,
         "motion_energy": 0.0, "body_speed": 0.0, "arm_amplitude": 0.0,
         "body_rhythm": 0.0, "arm_horizontal_motion": 0.0,
         "posture": "unknown", "hand_position": "unknown", "arm_trajectory": []}
    f.update(FULL_POSE)
    f.update(overrides)
    return f

# F1 五段全部命中（与模板一致）→ 应触发
engF1 = make_five_seg_engine()
featsF1 = []
for i in range(8):
    f = five_seg_frame()
    f["timestamp"] = i * 0.05
    featsF1.append(f)
outsF1 = run_match(engF1, featsF1)
check("F1 五段全命中应触发 custom_play", "custom_play" in outsF1, f"outs={outsF1}")

# F2 头部偏离（>2σ）：head_dy 偏移 0.15 → 头部段不通过 → 不应触发
engF2 = make_five_seg_engine()
featsF2 = []
for i in range(8):
    f = five_seg_frame(pose_head_dy=-0.10)   # 与模板 -0.25 差 0.15 = 3σ
    f["timestamp"] = i * 0.05
    featsF2.append(f)
outsF2 = run_match(engF2, featsF2)
check("F2 头部偏离3σ不应触发", "custom_play" not in outsF2, f"outs={outsF2}")

# F3 胸腔偏离：torso_dy 偏移 0.15 → 胸腔段不通过 → 不应触发
engF3 = make_five_seg_engine()
featsF3 = []
for i in range(8):
    f = five_seg_frame(pose_torso_dy=0.50)   # 与模板 0.35 差 0.15 = 3σ
    f["timestamp"] = i * 0.05
    featsF3.append(f)
outsF3 = run_match(engF3, featsF3)
check("F3 胸腔偏离3σ不应触发", "custom_play" not in outsF3, f"outs={outsF3}")

# F4 大臂偏离：upper_dx_R 与 upper_dy_R 同时偏移 3σ → 大臂段平均 >2σ → 不应触发
engF4 = make_five_seg_engine()
featsF4 = []
for i in range(8):
    f = five_seg_frame(pose_upper_dx_R=0.30, pose_upper_dy_R=0.05)  # 模板 0.15/-0.10，各差 0.15=3σ
    f["timestamp"] = i * 0.05
    featsF4.append(f)
outsF4 = run_match(engF4, featsF4)
check("F4 大臂偏离3σ不应触发", "custom_play" not in outsF4, f"outs={outsF4}")

# F5 手部偏离：hand_dx_R/hand_dy_R/spread_R 同时偏移 3σ → 手部段平均 >2σ → 不应触发
engF5 = make_five_seg_engine()
featsF5 = []
for i in range(8):
    f = five_seg_frame(pose_hand_dx_R=0.41, pose_hand_dy_R=0.0, pose_spread_R=0.50)
    f["timestamp"] = i * 0.05
    featsF5.append(f)
outsF5 = run_match(engF5, featsF5)
check("F5 手部偏离3σ不应触发", "custom_play" not in outsF5, f"outs={outsF5}")

# ---------- 部位筛选专项验证：只动臂部→头部胸腔不约束；纳入段全命中才触发 ----------
# G1 录制"只动臂部"的**非挥手类**动作（大臂/小臂/手部变化明显，头部/胸腔几乎不动，
# wave_direction=0 不激活挥手路由）→ 应只纳入 大臂/小臂/手部 三段，不纳入头部/胸腔。
# 先喂倒计时静止基线（模拟真实录制：倒计时采 baseline），再切录制喂动作帧。
recG = cp.PresetRecorder(duration_sec=3.0)
recG.start()  # 进入倒计时（STATE_COUNTDOWN → 采 baseline_samples）
for _ in range(10):   # 静止基线：手在低位、运动量≈0
    recG.feed({"motion_energy": 0.02, "arm_motion_energy": 0.02, "body_speed": 0.01,
               "arm_amplitude": 0.03, "body_rhythm": 0.01, "arm_horizontal_motion": 0.01,
               "palm_hover": 0.0, "wave_direction": 0.0,
               "pose_head_dx": 0.0, "pose_head_dy": -0.24,
               "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,
               "pose_upper_dx_L": -0.30, "pose_upper_dy_L": -0.20,
               "pose_upper_dx_R": 0.30, "pose_upper_dy_R": -0.20,
               "pose_fore_dx_L": -0.25, "pose_fore_dy_L": -0.18,
               "pose_fore_dx_R": 0.26, "pose_fore_dy_R": -0.15,
               "pose_hand_dx_L": -0.40, "pose_hand_dy_L": -0.28,
               "pose_hand_dx_R": 0.41, "pose_hand_dy_R": -0.25,
               "pose_spread_L": 0.33, "pose_spread_R": 0.35,
               "posture": "unknown", "hand_position": "unknown", "is_active": False,
               "arm_trajectory": [[0.5, 0.4], [0.51, 0.41], [0.5, 0.4]]})
recG.state = recG.STATE_RECORDING
recG.recording_start = time.time()
for i in range(30):
    dx = 0.06 * (i % 6)          # 大臂/小臂/手部横向往复 0~0.06（span≥0.02 被纳入）
    g = {
        "motion_energy": 0.18 + (i % 2) * 0.04, "arm_motion_energy": 0.25,
        "body_speed": 0.02, "arm_amplitude": 0.2, "body_rhythm": 0.02,
        "arm_horizontal_motion": 0.15, "palm_hover": 0.0, "wave_direction": 0.0,
        "pose_head_dx": 0.0, "pose_head_dy": -0.24,        # 头部固定（span=0 不纳入）
        "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,       # 胸腔固定（span=0 不纳入）
        "pose_upper_dx_L": -0.15 - dx, "pose_upper_dy_L": -0.10,
        "pose_upper_dx_R": 0.15 + dx, "pose_upper_dy_R": -0.10,   # 大臂动
        "pose_fore_dx_L": -0.10 - dx, "pose_fore_dy_L": -0.08,
        "pose_fore_dx_R": 0.11 + dx, "pose_fore_dy_R": -0.05,     # 小臂动
        "pose_hand_dx_L": -0.25 - dx, "pose_hand_dy_L": -0.18,
        "pose_hand_dx_R": 0.26 + dx, "pose_hand_dy_R": -0.15,     # 手部动
        "pose_spread_L": 0.33, "pose_spread_R": 0.35,
        "posture": "unknown", "hand_position": "unknown", "is_active": True,
        "arm_trajectory": [[0.3, 0.4], [0.33, 0.41], [0.3, 0.4], [0.27, 0.39]],  # 小幅轨迹
    }
    recG.feed(g)
condsG = recG.generate_conditions()
poseG = condsG.get("pose", {})
check("G1 只动臂部: pose 不含头部键", "pose_head_dx" not in poseG,
      f"keys={sorted(poseG.keys())[:6]}")
check("G1 只动臂部: pose 不含胸腔键", "pose_torso_dx" not in poseG,
      f"keys={sorted(poseG.keys())[:6]}")
check("G1 只动臂部: pose 含手部键", "pose_hand_dx_R" in poseG,
      f"keys={sorted(poseG.keys())[:6]}")

# G2 用 G1 的模板建预设：头部/胸腔随意变（未纳入）仍应触发（纳入段全部命中）
# 纳入段（大臂/小臂/手部）取录制模板均值：dx 均值 0.15 → upper_dx_R≈0.30 等
engG = gp.PresetEngine()
pg = gp.GesturePreset(name="只动臂部", description="test",
                      conditions=dict(condsG), action="custom_play", priority=20)
pg._is_custom = True
engG.add_preset(pg)
G_POSE_MEAN = {
    "pose_upper_dx_L": -0.30, "pose_upper_dy_L": -0.10,
    "pose_upper_dx_R": 0.30, "pose_upper_dy_R": -0.10,
    "pose_fore_dx_L": -0.25, "pose_fore_dy_L": -0.08,
    "pose_fore_dx_R": 0.26, "pose_fore_dy_R": -0.05,
    "pose_hand_dx_L": -0.40, "pose_hand_dy_L": -0.18,
    "pose_hand_dx_R": 0.41, "pose_hand_dy_R": -0.15,
    "pose_spread_L": 0.33, "pose_spread_R": 0.35,
}
featsG = []
for i in range(20):   # 1.0s > duration（G1 有动作 → duration 偏短）
    f = {"timestamp": i * 0.05, "palm_hover": 0.0, "wave_direction": 0.0,
         "motion_energy": 0.20, "arm_motion_energy": 0.25, "body_speed": 0.02,
         "arm_amplitude": 0.2, "body_rhythm": 0.02, "arm_horizontal_motion": 0.15,
         "pose_head_dx": 0.4, "pose_head_dy": 0.3,    # 头部明显偏离也 OK（未纳入）
         "pose_torso_dx": -0.3, "pose_torso_dy": 0.1, # 胸腔明显偏离也 OK（未纳入）
         "posture": "unknown", "hand_position": "unknown",
         "arm_trajectory": []}
    f.update(G_POSE_MEAN)
    featsG.append(f)
outsG = run_match(engG, featsG)
check("G2 头部胸腔随意变(未纳入)仍触发", "custom_play" in outsG, f"outs={outsG}")

# G3 纳入段全命中才触发：手部大幅偏离（未达成）→ 不触发
engG3 = gp.PresetEngine()
pg3 = gp.GesturePreset(name="只动臂部", description="test",
                       conditions=dict(condsG), action="custom_play", priority=20)
pg3._is_custom = True
engG3.add_preset(pg3)
featsG3 = []
for i in range(8):
    featsG3.append({"timestamp": i * 0.05, "palm_hover": 0.0, "wave_direction": 0.0,
                    "motion_energy": 0.3, "body_speed": 0.02, "arm_amplitude": 0.3,
                    "body_rhythm": 0.02, "arm_horizontal_motion": 0.5,
                    "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
                    "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
                    "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
                    "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
                    "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
                    "pose_hand_dx_R": 0.80, "pose_hand_dy_R": 0.60,  # 手部大幅偏离
                    "pose_spread_L": 0.33, "pose_spread_R": 0.35,
                    "posture": "unknown", "hand_position": "unknown",
                    "arm_trajectory": []})
outsG3 = run_match(engG3, featsG3)
check("G3 手部大幅偏离(纳入段未达成)不应触发", "custom_play" not in outsG3, f"outs={outsG3}")

# ---------- 动作类型"必须真的在动"专项验证：静止误触回归 ----------
# H1 手部参与的动作（active_segments 非空）：静止不动（手停在录制姿态附近，
# 运动特征≈基线）→ 不应触发（motion_ok=False → 动作类型被拒绝）
recH = cp.PresetRecorder(duration_sec=3.0)
recH.start()
recH.state = recH.STATE_RECORDING
recH.recording_start = time.time()
for i in range(30):
    dx = 0.06 * (i % 6)   # 手部/臂部明显动（span≥0.02 → active_segments 非空）
    h = {
        "motion_energy": 0.20, "arm_motion_energy": 0.30, "body_speed": 0.02,
        "arm_amplitude": 0.25, "body_rhythm": 0.02, "arm_horizontal_motion": 0.3,
        "palm_hover": 0.0, "wave_direction": 0.0,
        "pose_head_dx": 0.0, "pose_head_dy": -0.24,
        "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,
        "pose_upper_dx_L": -0.15 - dx, "pose_upper_dy_L": -0.10,
        "pose_upper_dx_R": 0.15 + dx, "pose_upper_dy_R": -0.10,
        "pose_fore_dx_L": -0.10 - dx, "pose_fore_dy_L": -0.08,
        "pose_fore_dx_R": 0.11 + dx, "pose_fore_dy_R": -0.05,
        "pose_hand_dx_L": -0.25 - dx, "pose_hand_dy_L": -0.18,
        "pose_hand_dx_R": 0.26 + dx, "pose_hand_dy_R": -0.15,
        "pose_spread_L": 0.33, "pose_spread_R": 0.35,
        "posture": "unknown", "hand_position": "unknown", "is_active": True,
        "arm_trajectory": [[0.3, 0.4], [0.36, 0.42], [0.3, 0.4], [0.24, 0.38]],
    }
    recH.feed(h)
condsH = recH.generate_conditions()
check("H1 录制含明显参与部位 → active_segments 非空",
      len(condsH.get("active_segments", [])) > 0,
      f"active={condsH.get('active_segments')}")

engH = gp.PresetEngine()
ph = gp.GesturePreset(name="手部参与动作", description="test",
                      conditions=dict(condsH), action="custom_play", priority=20)
ph._is_custom = True
engH.add_preset(ph)
# 静止帧：运动特征≈基线（motion_energy 0.02），手停在录制姿态附近（pose 接近）
idle_feats = []
for i in range(20):
    idle_feats.append({"timestamp": i * 0.05, "palm_hover": 0.0, "wave_direction": 0.0,
                       "motion_energy": 0.02, "arm_motion_energy": 0.01,
                       "body_speed": 0.01, "arm_amplitude": 0.02, "body_rhythm": 0.01,
                       "arm_horizontal_motion": 0.02,
                       "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
                       "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
                       "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
                       "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
                       "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
                       "pose_hand_dx_R": 0.26, "pose_hand_dy_R": -0.15,
                       "pose_spread_L": 0.33, "pose_spread_R": 0.35,
                       "posture": "unknown", "hand_position": "unknown",
                       "arm_trajectory": []})
outsH = run_match(engH, idle_feats)
check("H1 静止不动(手停录制姿态附近)不应触发", "custom_play" not in outsH, f"outs={outsH}")

# H2 同一动作：真的在动（运动特征偏离基线，姿态接近录制均值）→ 应触发
engH2 = gp.PresetEngine()
ph2 = gp.GesturePreset(name="手部参与动作", description="test",
                       conditions=dict(condsH), action="custom_play", priority=20)
ph2._is_custom = True
engH2.add_preset(ph2)
move_feats = []
for i in range(20):
    move_feats.append({"timestamp": i * 0.05, "palm_hover": 0.0, "wave_direction": 0.0,
                       "motion_energy": 0.20, "arm_motion_energy": 0.30,
                       "body_speed": 0.02, "arm_amplitude": 0.25, "body_rhythm": 0.02,
                       "arm_horizontal_motion": 0.3,
                       "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
                       "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
                       "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
                       "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
                       "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
                       "pose_hand_dx_R": 0.26, "pose_hand_dy_R": -0.15,
                       "pose_spread_L": 0.33, "pose_spread_R": 0.35,
                       "posture": "unknown", "hand_position": "unknown",
                       "arm_trajectory": []})
outsH2 = run_match(engH2, move_feats)
check("H2 真的在动(偏离基线)应触发", "custom_play" in outsH2, f"outs={outsH2}")

# H3 纯静态姿势（active_segments 空）：静止保持 → 仍应触发（不受"必须动"限制）
recS = cp.PresetRecorder(duration_sec=3.0)
recS.start()
recS.state = recS.STATE_RECORDING
recS.recording_start = time.time()
STATIC_FEAT_H = {
    "motion_energy": 0.02, "arm_motion_energy": 0.02, "body_speed": 0.01,
    "arm_amplitude": 0.03, "body_rhythm": 0.01, "arm_horizontal_motion": 0.01,
    "palm_hover": 0.1, "wave_direction": 0.0,
    "pose_head_dx": 0.0, "pose_head_dy": -0.24,
    "pose_torso_dx": 0.0, "pose_torso_dy": 0.35,
    "pose_upper_dx_L": -0.15, "pose_upper_dy_L": -0.10,
    "pose_upper_dx_R": 0.15, "pose_upper_dy_R": -0.10,
    "pose_fore_dx_L": -0.10, "pose_fore_dy_L": -0.08,
    "pose_fore_dx_R": 0.11, "pose_fore_dy_R": -0.05,
    "pose_hand_dx_L": -0.25, "pose_hand_dy_L": -0.18,
    "pose_hand_dx_R": 0.26, "pose_hand_dy_R": -0.15,
    "pose_spread_L": 0.33, "pose_spread_R": 0.35,
    "posture": "unknown", "hand_position": "unknown", "is_active": False,
    "arm_trajectory": [[0.5, 0.4], [0.51, 0.41], [0.5, 0.4]],
}
for _ in range(30):
    recS.feed(dict(STATIC_FEAT_H))
condsS = recS.generate_conditions()
check("H3 纯静态录制 → active_segments 为空", len(condsS.get("active_segments", [])) == 0,
      f"active={condsS.get('active_segments')}")
engS = gp.PresetEngine()
ps = gp.GesturePreset(name="静态姿势", description="test",
                      conditions=dict(condsS), action="custom_play", priority=20)
ps._is_custom = True
engS.add_preset(ps)
hold_featsS = []
for i in range(24):  # 1.2s 静止保持
    f = dict(STATIC_FEAT_H)
    f["timestamp"] = i * 0.05
    hold_featsS.append(f)
outsS = run_match(engS, hold_featsS)
check("H3 纯静态姿势静止保持仍应触发", "custom_play" in outsS, f"outs={outsS}")

# ---------- 本次修复专项验证：腕部高度门（戴耳机/挠头/调摄像头误触排除） ----------
# 根因：此前挥手判定只查肘部高度门，手臂弯曲抬起时肘部可在肩下方、腕部却
# 抬到耳/头顶高度（wry<-0.10），被误判为挥手切歌。新增腕部高度门后应排除。
# 且实机中手贴近头部时 MediaPipe 对腕部可见性评分普遍很低（0.1~0.5，触发帧
# 常 0.09~0.28），腕部高度门记录阈值须低于此范围（WRIST_VIS_MIN=0.10），
# 否则 wry 全为 None、高度门空转失效（"实机仍误触"二次根因）。以下用例
# 均用低腕部可见性（0.10~0.28）模拟实机，验证高度门在低可见性下仍生效。
# J1 戴耳机：手臂弯曲抬起，肘部在肩下横向摆出再收回，腕部在耳边高度（wry≈-0.15）
def hp_frame(ex, ey, hx, hy, wvis, evis):
    return mk_lm(elbow=(ex, ey), wrist=(hx, hy),
                 index=(hx + 0.04, hy - 0.02), pinky=(hx - 0.04, hy - 0.02),
                 thumb=(hx - 0.05, hy + 0.02),
                 wvis=wvis, evis=evis, fvis=max(0.10, wvis - 0.05))

_rnd.seed(11)
seq = []
for i in range(9):
    t = i / 8
    seq.append(hp_frame(0.46 + 0.16 * t, 0.36 - 0.02 * t, 0.52 + 0.24 * t, 0.15,
                        wvis=_rnd.uniform(0.10, 0.28), evis=_rnd.uniform(0.30, 0.90)))
for i in range(9):
    t = i / 8
    seq.append(hp_frame(0.62 - 0.15 * t, 0.34 + 0.02 * t, 0.76 - 0.22 * t, 0.15,
                        wvis=_rnd.uniform(0.10, 0.28), evis=_rnd.uniform(0.30, 0.90)))
r = run_seq(make_dummy(), seq)
check("J1 戴耳机(腕vis低0.1~0.28)不应误触切歌", peak(r) < 0.15, f"peak={peak(r):.3f}")

# J2 挠头：手在头顶小幅划圈，腕部 wry≈-0.20（肘部可保持在肩附近，腕vis 低）
def sc_frame(ex, ey, hx, hy, wvis, evis):
    return mk_lm(elbow=(ex, ey), wrist=(hx, hy),
                 index=(hx + 0.05, hy - 0.03), pinky=(hx - 0.04, hy - 0.02),
                 thumb=(hx - 0.05, hy + 0.03),
                 wvis=wvis, evis=evis, fvis=max(0.10, wvis - 0.05))

_rnd.seed(13)
seq = []
for i in range(16):
    a = i * 0.7
    seq.append(sc_frame(0.55 + 0.03 * np.sin(a), 0.28 + 0.012 * np.cos(a),
                        0.50 + 0.035 * np.sin(a * 1.2), 0.10 + 0.02 * np.cos(a * 1.2),
                        wvis=_rnd.uniform(0.10, 0.28), evis=_rnd.uniform(0.30, 0.90)))
r = run_seq(make_dummy(), seq)
check("J2 挠头(腕vis低0.1~0.28)不应误触切歌", peak(r) < 0.15, f"peak={peak(r):.3f}")

# J3 手动调摄像头：手伸向屏幕上方摄像头，小幅左右调整，腕部 wry≈-0.19（腕vis 低）
def cam_frame(ex, ey, hx, hy, wvis, evis):
    return mk_lm(elbow=(ex, ey), wrist=(hx, hy),
                 index=(hx + 0.06, hy - 0.02), pinky=(hx - 0.05, hy - 0.02),
                 thumb=(hx - 0.05, hy + 0.02),
                 wvis=wvis, evis=evis, fvis=max(0.10, wvis - 0.05))

_rnd.seed(17)
seq = []
for i in range(14):
    a = i * 0.6
    seq.append(cam_frame(0.50 + 0.025 * np.sin(a), 0.32,
                         0.50 + 0.05 * np.sin(a * 1.1), 0.11,
                         wvis=_rnd.uniform(0.10, 0.28), evis=_rnd.uniform(0.30, 0.90)))
r = run_seq(make_dummy(), seq)
check("J3 手动调摄像头(腕vis低0.1~0.28)不应误触切歌", peak(r) < 0.15, f"peak={peak(r):.3f}")

# J6 腕部可见性混合波动（0.10~0.50，与实机日志一致）仍应排除
def hp_frame2(ex, ey, hx, hy, wvis, evis):
    return mk_lm(elbow=(ex, ey), wrist=(hx, hy),
                 index=(hx + 0.04, hy - 0.02), pinky=(hx - 0.04, hy - 0.02),
                 thumb=(hx - 0.05, hy + 0.02),
                 wvis=wvis, evis=evis, fvis=max(0.10, wvis - 0.10))

_rnd.seed(19)
seq = []
for i in range(9):
    t = i / 8
    seq.append(hp_frame2(0.46 + 0.16 * t, 0.36 - 0.02 * t, 0.52 + 0.24 * t, 0.15,
                         wvis=_rnd.uniform(0.10, 0.50), evis=_rnd.uniform(0.30, 0.90)))
for i in range(9):
    t = i / 8
    seq.append(hp_frame2(0.62 - 0.15 * t, 0.34 + 0.02 * t, 0.76 - 0.22 * t, 0.15,
                         wvis=_rnd.uniform(0.10, 0.50), evis=_rnd.uniform(0.30, 0.90)))
r = run_seq(make_dummy(), seq)
check("J6 戴耳机(腕vis 0.1~0.5波动)不应误触切歌", peak(r) < 0.15, f"peak={peak(r):.3f}")

# ---------- 本次修复专项验证：弧线挥手 y/x 比放宽（1.6→2.0） ----------
# 根因：实机挥手肘部绕肩弧线，纵向分量常达横向 1.2~1.6 倍，旧阈值 1.6 恰好
# 卡在参考上限，弧线稍大即被排除 → 难触发。放宽到 2.0 后应触发。
# J4 斜弧线挥手：跨x=0.12 跨y=0.20，y/x≈1.67（旧阈值 1.6 会排除）
pts = [(0.72 - i * 0.015, 0.40 + i * 0.025) for i in range(8)] + \
      [(0.60 + i * 0.015, 0.60 - i * 0.025) for i in range(8)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("J4 斜弧线挥手(y/x≈1.67)应触发", peak(r) >= 0.15, f"peak={peak(r):.3f}")

# J5 纵向主导 y/x>2.0：跨x=0.08 跨y=0.18 → 仍排除（上下摆手/锤桌类）
pts = [(0.70 - i * 0.01, 0.40 + i * 0.0225) for i in range(8)] + \
      [(0.62 + i * 0.01, 0.58 - i * 0.0225) for i in range(8)]
r = run_seq(make_dummy(), [hand_frame(x, y) for x, y in pts])
check("J5 上下摆手(y/x≈2.25)仍不应触发", peak(r) < 0.15, f"peak={peak(r):.3f}")

print(f"\n===== {len(PASS)} 通过, {len(FAIL)} 失败 =====")
sys.exit(1 if FAIL else 0)
