# -*- coding: utf-8 -*-
"""挥手检测仿真测试：用合成关键点序列驱动 _calc_arm_horizontal_motion，
复现"挥手切不动 / 举手竖向误触"并验证修复。"""
import sys
import types
from collections import deque

import numpy as np

from motion_capture import MotionCapture

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def make_dummy():
    d = types.SimpleNamespace()
    d._wrist_xy_history = deque(maxlen=10)
    d._wave_direction = 0.0
    return d


def landmarks(right=None, left=None, vis=0.9):
    """构造 (33,4) 关键点；right/left = (x, y) 或 None（不可见）。"""
    lm = np.zeros((33, 4), dtype=np.float64)
    if right:
        for i in (16, 18, 20):
            lm[i] = [right[0], right[1], 0.0, vis]
    if left:
        for i in (15, 17, 19):
            lm[i] = [left[0], left[1], 0.0, vis]
    return lm


def run_seq(dummy, seq, dt=0.04):
    """逐帧驱动，返回 [(intensity, wave_direction)]。"""
    out = []
    for lm in seq:
        v = MotionCapture._calc_arm_horizontal_motion(dummy, lm, dt)
        out.append((v, dummy._wave_direction))
    return out


def seq_from_points(points, side="right"):
    return [landmarks(**{side: p}) if p else landmarks() for p in points]


# ---------- 场景 1：快速挥手（0.35s 横扫 40% 屏宽，右手） ----------
pts = [(0.75 - i * 0.05, 0.45) for i in range(10)]  # 0.75 -> 0.30
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
final_dir = r[-1][1]
check("W1 快速挥手应触发(intensity>=0.2)", peak >= 0.2, f"peak={peak:.2f}")
check("W1 方向应为-1(原始x减小)", final_dir == -1.0, f"dir={final_dir}")

# ---------- 场景 2：反方向快速挥手 ----------
pts = [(0.30 + i * 0.05, 0.45) for i in range(10)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("W2 反向挥手应触发", peak >= 0.2, f"peak={peak:.2f}")
check("W2 方向应为+1(原始x增大)", r[-1][1] == 1.0, f"dir={r[-1][1]}")

# ---------- 场景 3：纯竖向举手（x 几乎不动）不应触发 ----------
pts = [(0.60 + (0.005 if i % 2 else -0.005), 0.70 - i * 0.055) for i in range(11)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("R1 纯竖向举手不应触发", peak < 0.2, f"peak={peak:.2f}")

# ---------- 场景 4：斜向移动但横向跨度大 → 严格横向规则下会触发 ----------
# （用户明确要求"只看横向位移"：横向跨度够就触发，不看纵向）
pts = [(0.65 - i * 0.025, 0.70 - i * 0.05) for i in range(11)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("R2 斜向大横向位移应触发(只看横向)", peak >= 0.2, f"peak={peak:.2f}")

# ---------- 场景 5：举起后原地上下晃（y ±0.05 抖动，x 稳定） ----------
pts = [(0.55, 0.15 + (0.05 if i % 2 else -0.05)) for i in range(10)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("R3 举起后上下晃不应触发", peak < 0.2, f"peak={peak:.2f}")

# ---------- 场景 6：慢速自然挥手（0.8s 横扫 35%，低帧率 12fps） ----------
pts = [(0.70 - i * 0.035, 0.45) for i in range(11)]
r = run_seq(make_dummy(), seq_from_points(pts), dt=0.083)
peak = max(v for v, _ in r)
check("W3 慢速挥手应触发", peak >= 0.2, f"peak={peak:.2f}")

# ---------- 场景 7：挥手后停在终点（历史窗口含整段挥动） ----------
pts = [(0.75 - i * 0.05, 0.45) for i in range(9)] + [(0.30, 0.45)] * 4
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("W4 挥完停住仍应触发", peak >= 0.2, f"peak={peak:.2f}")

# ---------- 场景 8：往返甩手（甩出去再收回，也应能切歌） ----------
pts = [(0.40, 0.45)] + [(0.40 + i * 0.04, 0.45) for i in range(6)] \
      + [(0.64 - i * 0.04, 0.45) for i in range(6)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("W5 往返甩手应触发", peak >= 0.2, f"peak={peak:.2f}")

# ---------- 场景 9：举起后稳定，微小抖动（±0.02）不触发 ----------
pts = [(0.55 + (0.02 if i % 2 else -0.02), 0.15) for i in range(10)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("R4 举起后微抖不应触发", peak < 0.2, f"peak={peak:.2f}")


# ---------- 场景 10：举手过程中手指可见性抖动（根因回归测试） ----------
def landmarks_flicker(wrist, fingers, fingers_vis):
    """手腕固定在 wrist，手指在 fingers 位置但可见性闪烁。"""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[16] = [wrist[0], wrist[1], 0.0, 0.9]
    for i in (18, 20):
        lm[i] = [fingers[0], fingers[1], 0.0, fingers_vis]
    return lm


# 手腕从 (0.60,0.65) 竖直举到 (0.55,0.15)；手指在手腕左偏 0.10 处，
# 可见性在 0.35/0.20 间交替（跨过 0.3 阈值）。
seq = []
for i in range(12):
    w = (0.60 - i * 0.004, 0.65 - i * 0.042)
    f = (w[0] - 0.10, w[1])
    seq.append(landmarks_flicker(w, f, 0.35 if i % 2 else 0.20))
r = run_seq(make_dummy(), seq)
peak = max(v for v, _ in r)
check("R5 举手+手指可见性抖动不应误触", peak < 0.2, f"peak={peak:.2f}")


# ---------- 场景 11：掌心上下移动 + 手腕可见性抖动（误触根因回归） ----------
def landmarks_wrist_flicker(wrist, wrist_vis, fingers):
    """手腕可见性闪烁（跨过 0.3 阈值），手指远离手腕 0.12。"""
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[16] = [wrist[0], wrist[1], 0.0, wrist_vis]
    for i in (18, 20):
        lm[i] = [fingers[0], fingers[1], 0.0, 0.9]
    return lm


# 掌心保持朝镜头上下移动：手腕 x 稳定在 0.55±0.005，y 0.20→0.55，
# 手腕可见度在 0.35/0.20 间交替；手指在左偏 0.12 处（若被顶替会产生跳变）。
seq = []
for i in range(14):
    w = (0.55 + (0.005 if i % 2 else -0.005), 0.20 + i * 0.025)
    seq.append(landmarks_wrist_flicker(w, 0.35 if i % 2 else 0.20,
                                       (w[0] - 0.12, w[1])))
r = run_seq(make_dummy(), seq)
peak = max(v for v, _ in r)
check("R6 掌心上下移动+手腕抖动不应误触切歌", peak < 0.2, f"peak={peak:.2f}")

# ---------- 场景 12：上下移动手 + x 周期抖动 ±0.06（原始跨度 0.12） ----------
# 抖动周期为 2 帧，中值滤波应基本滤除 → 不触发
pts = [(0.55 + (0.06 if i % 2 else -0.06), 0.20 + i * 0.03) for i in range(12)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("R7 周期抖动被中值滤波抑制", peak < 0.2, f"peak={peak:.2f}")

# ---------- 场景 13：真实挥手叠加 ±0.02 噪声（跨度 0.40）仍应触发 ----------
import random as _rnd
_rnd.seed(7)
pts = [(0.72 - i * 0.04 + _rnd.uniform(-0.02, 0.02), 0.45) for i in range(11)]
r = run_seq(make_dummy(), seq_from_points(pts))
peak = max(v for v, _ in r)
check("W6 含噪声真实挥手应触发", peak >= 0.2, f"peak={peak:.2f}")
check("W6 方向正确(-1)", r[-1][1] == -1.0, f"dir={r[-1][1]}")

print(f"\n===== {len(PASS)} 通过, {len(FAIL)} 失败 =====")
sys.exit(1 if FAIL else 0)
