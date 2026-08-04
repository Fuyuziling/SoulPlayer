# -*- coding: utf-8 -*-
"""自定义预设链路测试：条件生成/清洗/内置动作屏蔽。"""
import sys

from custom_preset import PresetRecorder, CustomPresetManager, CustomPreset
from gesture_presets import PresetEngine, GesturePreset

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ---------- G1: 录制生成条件——unknown 枚举不写入 ----------
rec = PresetRecorder(duration_sec=3.0)
rec.start()
rec.state = rec.STATE_RECORDING  # 跳过倒计时
rec.recording_start = __import__('time').time()
feat_unknown = {"motion_energy": 0.02, "arm_motion_energy": 0.5,
                "body_speed": 0.0, "arm_amplitude": 0.3, "body_rhythm": 0.5,
                "arm_horizontal_motion": 0.1,
                "posture": "unknown", "hand_position": "unknown",
                "is_active": True}
for _ in range(20):
    rec.feed(dict(feat_unknown))
cond = rec.generate_conditions()
check("G1 unknown 姿态不写入条件", "posture" not in cond and "hand_position" not in cond,
      str([k for k in cond]))
check("G1 不写 is_active 条件", "is_active" not in cond)
check("G1 duration=1.0", cond.get("duration") == 1.0)
check("G1 数值条件保留", "motion_energy" in cond and "arm_horizontal_motion" in cond)

# ---------- G2: 可分类且稳定的枚举会保留 ----------
rec2 = PresetRecorder(duration_sec=3.0)
rec2.start()
rec2.state = rec2.STATE_RECORDING
rec2.recording_start = __import__('time').time()
for i in range(20):
    f = dict(feat_unknown)
    f["posture"] = "sitting" if i < 18 else "unknown"  # 90% sitting
    f["hand_position"] = "down"
    rec2.feed(f)
cond2 = rec2.generate_conditions()
check("G2 稳定姿态保留", cond2.get("posture") == ["sitting"], str(cond2.get("posture")))
check("G2 稳定手位保留", cond2.get("hand_position") == ["down"])

# ---------- G3: 历史预设清洗 ----------
p = CustomPreset(name="t1", description="", action="custom_play", priority=20,
                 tracks=["/a.mp3"],
                 conditions={"duration": 1.5, "posture": ["unknown"],
                             "hand_position": ["unknown"], "is_active": True,
                             "motion_energy": {"min": 0.0, "max": 0.05}})
changed = CustomPresetManager._sanitize_conditions(p)
check("G3 清洗有改动", changed)
check("G3 unknown 枚举已删", "posture" not in p.conditions and "hand_position" not in p.conditions)
check("G3 is_active 已删", "is_active" not in p.conditions)
check("G3 duration 降到 1.0", p.conditions["duration"] == 1.0)
check("G3 数值条件未动", p.conditions["motion_energy"]["max"] == 0.05)

# ---------- E1: 自定义匹配期间内置动作被屏蔽 ----------
engine = PresetEngine()
wave = GesturePreset(name="挥手切歌", description="", action="player_next", priority=10,
                     conditions={"arm_horizontal_motion": {"min": 0.2, "max": 1.0},
                                 "duration": 0.05})
custom = CustomPreset(name="自定义", description="", action="custom_play", priority=20,
                      tracks=["/a.mp3"],
                      conditions={"motion_energy": {"min": 0.0, "max": 1.0},
                                  "duration": 1.0})
custom._is_custom = True
engine.add_preset(wave)
engine.add_preset(custom)

# 特征：挥手强度拉满 + 自定义也在匹配（motion_energy 0.3 在 0~1 内）
t = 1000.0
fired = []
for i in range(15):  # 1.5 秒，步长 0.1
    t += 0.1
    a = engine.match({"timestamp": t, "arm_horizontal_motion": 1.0,
                      "motion_energy": 0.3})
    if a:
        fired.append((round(t - 1000, 1), a))

check("E1 前 1.0 秒内置挥手被屏蔽", all(a != "player_next" for _, a in fired if _ <= 1.0),
      str(fired))
check("E1 自定义到点正常触发", any(a == "custom_play" for _, a in fired), str(fired))

# ---------- E2: 自定义不匹配时内置挥手正常触发 ----------
engine2 = PresetEngine()
engine2.add_preset(GesturePreset(name="挥手切歌", description="", action="player_next",
                                 priority=10,
                                 conditions={"arm_horizontal_motion": {"min": 0.2, "max": 1.0},
                                             "duration": 0.05}))
custom2 = CustomPreset(name="自定义", description="", action="custom_play", priority=20,
                       tracks=["/a.mp3"],
                       conditions={"motion_energy": {"min": 0.0, "max": 0.1},
                                   "duration": 1.0})
custom2._is_custom = True
engine2.add_preset(custom2)
t = 2000.0
fired2 = []
for i in range(5):
    t += 0.1
    a = engine2.match({"timestamp": t, "arm_horizontal_motion": 1.0,
                       "motion_energy": 0.9})  # 自定义不匹配
    if a:
        fired2.append(a)
check("E2 自定义不匹配时挥手正常触发", "player_next" in fired2, str(fired2))

print(f"\n===== {len(PASS)} 通过, {len(FAIL)} 失败 =====")
sys.exit(1 if FAIL else 0)
