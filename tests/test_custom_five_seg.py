# -*- coding: utf-8 -*-
"""
_test_custom_five_seg.py - 自定义动作五部分判断法 + 静态/动态混合模式模拟验证

覆盖（gesture-tuning 流程第 5 步）：
  S1 动态动作（D≈0.9，混合偏运动 + 手部重点）：
     a 姿势+手型+运动全部对齐 → 命中
     b 运动量相同但手型不同（张掌→握拳）→ 拒绝（手部重点约束）
     c 未捕获部位（头部/胸腔）乱动 → 不约束（仍命中）
     d 静止近似基线 → 拒绝（动作类必须真的在动）
     e DTW 形态命中但手型不同 → 仍拒绝（手型门槛不被绕过）
     f 运动量相同、手臂位置偏但手型对 → 混合模式命中（动态偏运动判定）
  S2 静态动作（D=0.0，纯构型）：五段全部命中 → 触发；任一段偏离 → 拒绝
  S3 与通用动作互斥：wave_direction / palm_hover 激活时自定义不触发
  S4 引擎级：内置挥手正常触发；挥手期间自定义让位；自定义期间挥手让位
  S5 结构稳定性：reset / remove_preset / 纯运动旧预设不崩溃
  S6 旧格式范围条件预设（custom_153438.json 同构）：
     满足 min/max 范围 → 命中；超出范围（如 motion_energy 过高）→ 拒绝
     （此前误触变频繁的回归：范围约束被新判定器忽略）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gesture_presets import PresetEngine, GesturePreset, CustomActionMatcher

FAIL = []


def check(name, cond, info=""):
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name} {info}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------------------
#  构造数据
# ---------------------------------------------------------------------------
# 动态动作模板：录制捕获 大臂/小臂/手部（头部/胸腔未捕获 → 不约束）
DYNAMIC_POSE = {
    "pose_upper_dx_L": 0.10, "pose_upper_dy_L": 0.15,
    "pose_upper_dx_R": 0.12, "pose_upper_dy_R": 0.14,
    "pose_fore_dx_L": 0.20, "pose_fore_dy_L": 0.25,
    "pose_fore_dx_R": 0.22, "pose_fore_dy_R": 0.24,
    "pose_hand_dx_L": 0.30, "pose_hand_dy_L": 0.35,
    "pose_hand_dx_R": 0.32, "pose_hand_dy_R": 0.34,
    "pose_spread_L": 0.30, "pose_spread_R": 0.32,  # 张掌（真实录制量级）
}
DYNAMIC_POSE_STD = {k: 0.05 for k in DYNAMIC_POSE}

DYN_ACTION = {
    "arm_motion_energy": 0.60, "motion_energy": 0.30,
    "body_speed": 0.05, "arm_amplitude": 0.50,
    "arm_horizontal_motion": 0.20, "body_rhythm": 0.10,
    "palm_hover": 0.0, "wave_direction": 0.0,
}
DYN_BASELINE = {
    "arm_motion_energy": 0.05, "motion_energy": 0.03,
    "body_speed": 0.01, "arm_amplitude": 0.05,
    "arm_horizontal_motion": 0.01, "body_rhythm": 0.02,
    "palm_hover": 0.0, "wave_direction": 0.0,
}
DYN_STD = {k: 0.05 for k in DYN_ACTION}

TRAJ = [[0.10 + 0.05 * i, 0.30 + 0.02 * ((i % 4) - 1.5)] for i in range(20)]

DYNAMIC_COND = {
    "duration": 0.3, "dynamism": 0.9,
    "active_segments": ["大臂", "小臂", "手部"],
    "action": DYN_ACTION, "baseline": DYN_BASELINE, "action_std": DYN_STD,
    "pose": DYNAMIC_POSE, "pose_std": DYNAMIC_POSE_STD,
    "trajectory": TRAJ,
}

# 静态动作：无 active_segments（兜底五段全纳入），无运动模板
STATIC_POSE = dict(DYNAMIC_POSE)
STATIC_POSE.update({
    "pose_head_dx": 0.0, "pose_head_dy": -0.20,
    "pose_torso_dx": 0.0, "pose_torso_dy": 0.0,
})
STATIC_COND = {
    "duration": 1.0, "dynamism": 0.0,
    "pose": STATIC_POSE,
    "pose_std": {k: 0.05 for k in STATIC_POSE},
}


def base_features(**over):
    f = {
        "arm_motion_energy": 0.60, "motion_energy": 0.30,
        "body_speed": 0.05, "arm_amplitude": 0.50,
        "arm_horizontal_motion": 0.20, "body_rhythm": 0.10,
        "palm_hover": 0.0, "wave_direction": 0.0,
        "posture": "unknown", "hand_position": "unknown",
        "is_active": True, "arm_trajectory": TRAJ,
        **DYNAMIC_POSE,
    }
    f.update(over)
    return f


def feed(engine, frames, dt=0.05, hold=0.4):
    """连续帧喂入，返回输出序列（含各帧 match 结果）。"""
    ts = 0.0
    outs = []
    n = 0
    while ts < hold:
        f = dict(frames(n))
        f["timestamp"] = ts
        outs.append(engine.match(f))
        ts += dt
        n += 1
    return outs


# ---------------------------------------------------------------------------
#  S1 动态自定义动作（混合模式：偏运动 + 手部重点）
# ---------------------------------------------------------------------------
matcher = CustomActionMatcher()

# S1a 姿势/手型/运动全部对齐 → 命中
check("S1a 动态动作正确复现 → 命中",
      matcher.match(DYNAMIC_COND, base_features()) is True)

# S1b 运动量相同、手型不同（张掌 0.30 → 握拳 0.02）→ 拒绝（手部重点）
fist = base_features(pose_spread_L=0.02, pose_spread_R=0.03)
check("S1b 运动量相同但手型不同 → 拒绝",
      matcher.match(DYNAMIC_COND, fist) is False)

# S1c 未捕获部位（头部/胸腔）乱动 → 不约束，仍命中
moved_head = base_features(pose_head_dx=0.40, pose_head_dy=0.35,
                           pose_torso_dx=0.20, pose_torso_dy=0.10)
check("S1c 未捕获部位乱动不约束 → 命中",
      matcher.match(DYNAMIC_COND, moved_head) is True)

# S1d 静止（特征≈基线）→ 拒绝（动作类必须真的在动）
static_feat = {k: DYN_BASELINE[k] for k in DYN_BASELINE}
static_feat.update(DYNAMIC_POSE)
static_feat["arm_trajectory"] = []
check("S1d 静止近似基线 → 拒绝",
      matcher.match(DYNAMIC_COND, static_feat) is False)

# S1e DTW 形态命中（轨迹相同）但手型不同 → 仍拒绝（手型门槛不被绕过）
check("S1e DTW 命中但手型不同 → 拒绝",
      matcher.match(DYNAMIC_COND, fist) is False)

# S1f 运动量相同、手臂位置偏（手部段 6σ）但手型对 → 混合模式命中
arm_off = base_features(pose_hand_dx_R=0.32 + 0.30)
check("S1f 手臂位置偏但手型对 → 命中（动态偏运动判定）",
      matcher.match(DYNAMIC_COND, arm_off) is True)

# ---------------------------------------------------------------------------
#  S2 静态自定义动作（D=0，纯构型：五段全部命中）
# ---------------------------------------------------------------------------
stat_feat = {**STATIC_POSE, "arm_motion_energy": 0.0, "motion_energy": 0.0,
             "body_speed": 0.0, "arm_amplitude": 0.0,
             "arm_horizontal_motion": 0.0, "body_rhythm": 0.0,
             "palm_hover": 0.0, "wave_direction": 0.0, "arm_trajectory": []}
check("S2a 静态五段全部命中 → 命中",
      matcher.match(STATIC_COND, stat_feat) is True)
stat_wrong = dict(stat_feat)
stat_wrong["pose_hand_dy_R"] = stat_feat["pose_hand_dy_R"] + 0.15
check("S2b 静态任一段偏离(手部+0.15=3σ) → 拒绝",
      matcher.match(STATIC_COND, stat_wrong) is False)

# ---------------------------------------------------------------------------
#  S3 与通用动作互斥
# ---------------------------------------------------------------------------
check("S3a wave_direction 激活时自定义 → 拒绝",
      matcher.match(DYNAMIC_COND, base_features(wave_direction=0.8)) is False)
check("S3b palm_hover 激活时自定义 → 拒绝",
      matcher.match(DYNAMIC_COND, base_features(palm_hover=0.8)) is False)

# ---------------------------------------------------------------------------
#  S4 引擎级：通用动作正常、自定义路由互斥（含内置预设加载）
# ---------------------------------------------------------------------------
eng = PresetEngine()
presets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "presets", "default_presets.json")
eng.load_presets(presets_path)
custom = GesturePreset(name="测试自定义", description="t",
                       conditions=DYNAMIC_COND, action="custom_play",
                       priority=20)
custom._is_custom = True
eng.add_preset(custom)


def wave_frames(n):
    return base_features(wave_direction=0.8 if n % 4 < 2 else -0.8,
                         arm_trajectory=[(0.1 + 0.1 * i, 0.3) for i in range(10)])


def custom_frames(n):
    return base_features()


outs_wave = feed(eng, wave_frames, hold=0.6)
check("S4a 内置挥手仍触发 player_next/prev",
      any(o in ("player_next", "player_prev") for o in outs_wave),
      f"outs={[o for o in outs_wave if o]}")
check("S4b 挥手期间自定义不触发",
      not any(o == "custom_play" for o in outs_wave))

outs_custom = feed(eng, custom_frames, hold=0.6)
check("S4c 自定义动作触发 custom_play",
      any(o == "custom_play" for o in outs_custom),
      f"outs={[o for o in outs_custom if o]}")
check("S4d 自定义期间挥手不触发",
      not any(o in ("player_next", "player_prev") for o in outs_custom))

# ---------------------------------------------------------------------------
#  S5 结构稳定性
# ---------------------------------------------------------------------------
eng.reset()
eng2 = PresetEngine()
legacy = GesturePreset(name="纯运动旧预设", description="t",
                       conditions={"duration": 0.3, "dynamism": 0.9,
                                   "active_segments": ["手部"],
                                   "action": {"arm_motion_energy": 0.6},
                                   "baseline": {"arm_motion_energy": 0.05},
                                   "action_std": {"arm_motion_energy": 0.05}},
                       action="custom_play", priority=20)
legacy._is_custom = True
eng2.add_preset(legacy)
check("S5a 纯运动旧预设（无 pose）可判定不崩溃",
      eng2.match(base_features()) in (None, "custom_play"))
eng2.remove_preset("纯运动旧预设")
check("S5b remove_preset 后计时器清理",
      "纯运动旧预设" not in eng2.hold_timers
      and "纯运动旧预设" not in eng2._custom_hold_timers)
check("S5c 空引擎 match 返回 None", eng2.match(base_features()) is None)

# ---------------------------------------------------------------------------
#  S6 旧格式范围条件预设（custom_153438.json 同构）回归：
#     范围约束必须生效——超出 min/max → 拒绝（此前误触变频繁的根因）
# ---------------------------------------------------------------------------
eng6 = PresetEngine()
legacy_range = GesturePreset(
    name="旧格式范围预设", description="t",
    conditions={
        "duration": 1.0,
        "motion_energy": {"min": 0.0, "max": 0.05},
        "arm_motion_energy": {"min": 0.0, "max": 1.0},
        "body_rhythm": {"min": 0.143, "max": 0.789},
        "arm_horizontal_motion": {"min": 0.0, "max": 0.552},
    },
    action="custom_play", priority=20)
legacy_range._is_custom = True
eng6.add_preset(legacy_range)

# S6a 满足全部范围 → 命中
in_range = {"motion_energy": 0.02, "arm_motion_energy": 0.4,
            "body_rhythm": 0.5, "arm_horizontal_motion": 0.1,
            "palm_hover": 0.0, "wave_direction": 0.0,
            "posture": "unknown", "hand_position": "unknown",
            "arm_trajectory": []}
outs_in = feed(eng6, lambda n: dict(in_range), hold=1.2)
check("S6a 范围满足 → 命中", "custom_play" in outs_in, f"outs={[o for o in outs_in if o]}")

# S6b motion_energy 超上限（0.3 > 0.05）→ 拒绝
out_of_range = dict(in_range)
out_of_range["motion_energy"] = 0.30
outs_out = feed(eng6, lambda n: dict(out_of_range), hold=1.2)
check("S6b 超出范围(motion_energy=0.3) → 拒绝",
      "custom_play" not in outs_out, f"outs={[o for o in outs_out if o]}")

# S6c body_rhythm 超出范围（0.9 > 0.789）→ 拒绝
out_rhythm = dict(in_range)
out_rhythm["body_rhythm"] = 0.90
outs_rhythm = feed(eng6, lambda n: dict(out_rhythm), hold=1.2)
check("S6c 超出范围(body_rhythm=0.9) → 拒绝",
      "custom_play" not in outs_rhythm, f"outs={[o for o in outs_rhythm if o]}")

print()
if FAIL:
    print(f"==== {len(FAIL)} 项失败: {FAIL} ====")
    sys.exit(1)
print("==== 全部通过 ====")
