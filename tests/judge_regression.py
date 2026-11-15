# -*- coding: utf-8 -*-
"""
判定链条回归测试 —— 不启摄像头，直接驱动真实 PresetEngine。

用途：改动 gesture_presets.py（判定/锁存）或 main.py（冷却闸门）后必跑，
      确认「引擎 raw 触发」与「过闸门后实际执行」两个数字没有恶化。

设计要点：
- MainGate 复用生产代码的 ACTION_COOLDOWN_RULES（从 main.py 导入），
  因此改了冷却配置这里会立刻跟着变，不会出现两处漂移。
- 场景 C 的期望值是有意保留的 2 次：两次触发间隔 1.55s，远超 0.35/0.6s
  冷却窗口，代表"用户真的又挥了一次"，不是缺陷。

用法：
    py tests/judge_regression.py            # 全场景回归
    py tests/judge_regression.py -v         # 打印每次触发的帧号与预设名
退出码：0 = 全通过，1 = 有场景未达标。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# 导入 main 会触发 app_logging.setup()，这里只要纯透传：
# 不落盘、不装控制台 handler，否则测试输出会打成两份
os.environ.setdefault("SOULPLAYER_NO_LOG", "1")

from gesture_presets import PresetEngine
from custom_preset import CustomPresetManager
# 直接复用生产代码的冷却规则表，保证测试锁定的就是线上实际生效的配置
from main import ACTION_COOLDOWN_RULES

VERBOSE = "-v" in sys.argv


class MainGate:
    """main.py 的 _check_action_cooldown 等价实现（虚拟时间版）。

    行为必须与 main.py 中同名方法保持一致，唯一区别是把 time.time()
    换成外部传入的虚拟时间戳。冷却数值**不在此处重复定义**，直接读
    ACTION_COOLDOWN_RULES —— 早先这里是手抄的表，改了生产配置测试却
    不会跟着变（漂移），现已改为导入同一份。
    """

    def __init__(self):
        self.cd = {}

    def pass_gate(self, action: str, now: float) -> bool:
        rule = ACTION_COOLDOWN_RULES.get(action)
        if rule is None:
            return True
        if now - self.cd.get(action, 0.0) < rule.self_cd:
            return False
        for peer in rule.peers:
            if now - self.cd.get(peer, 0.0) < rule.peer_cd:
                return False
        self.cd[action] = now
        for extra in rule.also_block:
            self.cd[extra] = now
        return True


def build_engine():
    e = PresetEngine()
    e.load_presets("presets/default_presets.json")
    cm = CustomPresetManager(storage_dir="custom_presets")
    cm.load()
    cm.inject_into_engine(e)
    return e


def base_features():
    """静坐基线：低能量、无挥手、无举手。"""
    return {
        "motion_energy": 0.05,
        "arm_motion_energy": 0.04,
        "body_rhythm": 0.20,
        "arm_horizontal_motion": 0.02,
        "palm_hover": 0.0,
        "wave_direction": 0.0,
        "arm_amplitude": 0.10,
        "body_speed": 0.03,
        "left_hand_raise": 0.0,
        "right_hand_raise": 0.0,
    }


_tpl_cache = None


def _load_tpl():
    """取首个新格式自定义模板的 action+pose 数值与离散约束作为'完美匹配'输入。"""
    global _tpl_cache
    if _tpl_cache is None:
        cm = CustomPresetManager(storage_dir="custom_presets")
        cm.load()
        p = [x for x in cm.presets
             if isinstance(x.conditions.get("action"), dict)][0]
        c = p.conditions
        d = {}
        for grp in ("action", "pose"):
            for k, v in (c.get(grp) or {}).items():
                if isinstance(v, (int, float)):
                    d[k] = v
        for k in ("posture", "hand_position",
                  "hand_gesture_L", "hand_gesture_R"):
            if k in c:
                v = c[k]
                d[k] = v[0] if isinstance(v, list) else v
        _tpl_cache = d
    return _tpl_cache


# ── 帧变换函数 ──
def s_idle(f, i, t):
    pass


def s_wave_hold(f, i, t):
    if i < 90:
        f.update({"arm_horizontal_motion": 0.30, "arm_motion_energy": 0.35,
                  "motion_energy": 0.30, "wave_direction": 1.0,
                  "arm_amplitude": 0.50, "body_speed": 0.20})


def s_wave_flicker(f, i, t):
    if i < 90:
        d = 0.0 if 45 <= i <= 47 else 1.0
        f.update({"arm_horizontal_motion": 0.30, "arm_motion_energy": 0.35,
                  "motion_energy": 0.30, "wave_direction": d,
                  "arm_amplitude": 0.50, "body_speed": 0.20})


def s_hand_up(f, i, t):
    if i < 90:
        f.update({"palm_hover": 1.0, "left_hand_raise": 1.0,
                  "arm_amplitude": 0.30})


def s_wave_slight(f, i, t):
    """轻微摆手：肘部跨度约屏宽 6.5%（特征 0.163），保持 20 帧后随窗口滑出。
    这代表"手在摄像头前随手动一下"，不应触发切歌。"""
    if i < 20:
        f.update({"arm_horizontal_motion": 0.163, "arm_motion_energy": 0.20,
                  "motion_energy": 0.18, "wave_direction": 1.0,
                  "arm_amplitude": 0.30, "body_speed": 0.12})


def s_wave_real(f, i, t):
    """真实挥手：跨度爬升至屏宽 14%（特征 0.35），应触发且只触发一次。"""
    if i < 20:
        h = min(0.10 + 0.04 * (i + 1), 0.35)
        f.update({"arm_horizontal_motion": h, "arm_motion_energy": 0.35,
                  "motion_energy": 0.30, "wave_direction": 1.0,
                  "arm_amplitude": 0.50, "body_speed": 0.20})


def s_custom_hold(f, i, t):
    if i < 90:
        f.update(_load_tpl())


def s_custom_jitter(f, i, t):
    """条件持续满足，但每 20 帧出现一次单帧不满足（模拟手型识别抖动）。"""
    if i < 90 and i % 20 != 19:
        f.update(_load_tpl())


def s_custom_drop(f, i, t):
    """条件持续满足，但每 30 帧掉一次特征值（模拟采集丢帧）。"""
    if i < 90:
        if i % 30 == 29:
            f.update({"motion_energy": 0.0, "arm_motion_energy": 0.0})
        else:
            f.update(_load_tpl())


# ── 场景表：(名称, 帧变换, 期望实际执行次数, 备注) ──
SCENARIOS = [
    ("A 静坐 4s", s_idle, 0, "不应有任何触发"),
    ("B 挥手持续 3s", s_wave_hold, 1, "保持期间只触发一次"),
    ("C 挥手方向闪断", s_wave_flicker, 2,
     "两次间隔 1.55s > 冷却，属用户二次挥手，允许"),
    ("D 举手悬停 3s", s_hand_up, 1, "保持期间只触发一次"),
    ("E 自定义持续 3s", s_custom_hold, 1, "跨预设错峰应被拦下"),
    ("F 自定义+单帧抖动", s_custom_jitter, 1, "抖动不得放大为重复触发"),
    ("G 自定义+掉特征值", s_custom_drop, 1, "不得重复且不得串到别的预设"),
    ("H 轻微摆手 6.5%", s_wave_slight, 0, "随手动一下不得切歌（灵敏度上限）"),
    ("I 真实挥手 14%", s_wave_real, 1, "正常挥手必须触发一次（灵敏度下限）"),
]


def run(name, frames_fn, expect, note, frames=120, dt=0.033):
    e = build_engine()
    gate = MainGate()
    raw, eff = [], []
    t = 1000.0
    for i in range(frames):
        t += dt
        f = base_features()
        frames_fn(f, i, t)
        f["timestamp"] = t
        a = e.match(f)
        if a:
            pname = getattr(e.active_preset, "name", "?")
            raw.append((i, a, pname))
            if gate.pass_gate(a, t):
                eff.append((i, a, pname))

    ok = len(eff) == expect
    # 串台检测：一次动作期间出现了多个不同预设
    names = {n for _, _, n in eff}
    cross = len(names) > 1
    status = "PASS" if ok and not cross else "FAIL"
    print(f"[{status}] {name}")
    print(f"       raw={len(raw):>2}  实际执行={len(eff):>2}  期望={expect}"
          f"   {note}")
    if cross:
        print(f"       !! 跨预设串台: {sorted(names)}")
    if VERBOSE:
        if raw:
            print("       raw: " + ", ".join(
                f"帧{i}:{a}({n})" for i, a, n in raw[:8])
                + (" ..." if len(raw) > 8 else ""))
        if eff:
            print("       eff: " + ", ".join(
                f"帧{i}:{a}({n})" for i, a, n in eff[:8])
                + (" ..." if len(eff) > 8 else ""))
    return status == "PASS"


def main():
    print("=" * 68)
    print("判定链条回归测试 —— 引擎锁存 vs main 冷却闸门")
    print("(MainGate 直接复用 main.ACTION_COOLDOWN_RULES，不会漂移)")
    print("=" * 68)
    results = [run(*s) for s in SCENARIOS]
    print("=" * 68)
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 通过")
    if passed < total:
        for (name, *_), ok in zip(SCENARIOS, results):
            if not ok:
                print(f"  未达标: {name}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
