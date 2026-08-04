# -*- coding: utf-8 -*-
"""手掌悬停（举手暂停）检测仿真测试。"""
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


def lm_hand(wrist, index, pinky, vis=0.9, wrist_vis=None):
    """右手三点独立位置；wrist_vis 缺省时与 vis 相同。其余点不可见。"""
    if wrist_vis is None:
        wrist_vis = vis
    lm = np.zeros((33, 4), dtype=np.float64)
    lm[16] = [wrist[0], wrist[1], 0.0, wrist_vis]
    lm[20] = [index[0], index[1], 0.0, vis]
    lm[18] = [pinky[0], pinky[1], 0.0, vis]
    return lm


def drive(dummy, frames):
    """模拟真实流水线：先挥手模块（填充腕部历史）再手掌悬停模块。"""
    out = []
    for lm in frames:
        MotionCapture._calc_arm_horizontal_motion(dummy, lm, 0.04)
        out.append(MotionCapture._calc_palm_hover(dummy, lm))
    return out


# P1: 掌心正对镜头（展开 0.06）静止 12 帧 → 第 4 帧起为 1.0
frames = [lm_hand((0.50, 0.30), (0.47, 0.22), (0.53, 0.22)) for _ in range(12)]
r = drive(make_dummy(), frames)
check("P1 前3帧未触发", all(v == 0.0 for v in r[:3]), str(r[:3]))
check("P1 悬停应触发", all(v == 1.0 for v in r[3:]), str(r[3:6]))

# P2: 掌心正对但正在挥手（腕部移动）→ 不触发
frames = [lm_hand((0.30 + i * 0.05, 0.30), (0.27 + i * 0.05, 0.22),
                  (0.33 + i * 0.05, 0.22)) for i in range(10)]
r = drive(make_dummy(), frames)
check("P2 挥动中不触发悬停", all(v == 0.0 for v in r), f"peak={max(r)}")

# P3: 手静止但侧对镜头（展开 0.01 塌陷）→ 不触发
frames = [lm_hand((0.50, 0.30), (0.495, 0.22), (0.505, 0.22)) for _ in range(12)]
r = drive(make_dummy(), frames)
check("P3 侧对镜头不触发", all(v == 0.0 for v in r), f"peak={max(r)}")

# P4: 指根完全不可见（0.15 < 0.25，掌背/遮挡）→ 不触发
frames = [lm_hand((0.50, 0.30), (0.47, 0.22), (0.53, 0.22), vis=0.15)
          for _ in range(12)]
r = drive(make_dummy(), frames)
check("P4 指根不可见不触发", all(v == 0.0 for v in r), f"peak={max(r)}")

# P4b: 手腕清晰(0.9)但指根仅 0.35（掌心正对的典型低分）→ 应触发
frames = [lm_hand((0.50, 0.30), (0.47, 0.22), (0.53, 0.22),
                  vis=0.35, wrist_vis=0.9) for _ in range(12)]
r = drive(make_dummy(), frames)
check("P4b 指根低可见度0.35应触发", all(v == 1.0 for v in r[3:]), str(r[3:6]))

# P4c: 手腕也不清晰（0.3 < 0.4）→ 不触发
frames = [lm_hand((0.50, 0.30), (0.47, 0.22), (0.53, 0.22), vis=0.3)
          for _ in range(12)]
r = drive(make_dummy(), frames)
check("P4c 手腕不清晰不触发", all(v == 0.0 for v in r), f"peak={max(r)}")

# P5: 悬停中途开始挥动 → 立即退出悬停
d = make_dummy()
frames = [lm_hand((0.50, 0.30), (0.47, 0.22), (0.53, 0.22)) for _ in range(6)]
frames += [lm_hand((0.50 + i * 0.06, 0.30), (0.47 + i * 0.06, 0.22),
                   (0.53 + i * 0.06, 0.22)) for i in range(1, 6)]
r = drive(d, frames)
check("P5 悬停段触发", any(v == 1.0 for v in r[:6]), str(r[:6]))
check("P5 挥动段立即退出", all(v == 0.0 for v in r[6:]), str(r[6:]))

print(f"\n===== {len(PASS)} 通过, {len(FAIL)} 失败 =====")
sys.exit(1 if FAIL else 0)
