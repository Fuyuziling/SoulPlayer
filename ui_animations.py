# -*- coding: utf-8 -*-
"""
ui_animations.py — SoulPlayer 动画系统

提供：
- ValueAnimator: 数值平滑插值（线性、ease-in-out、elastic）
- PulseEffect: 脉冲效果（录制指示器等）
- FeatureBarAnimator: 特征进度条平滑过渡
- StateTransition: 状态切换过渡（颜色、位置、透明度）

所有动画均基于时间驱动，在主循环中每帧调用 update(dt) 即可。
"""

import time
import math
from typing import Optional, Callable, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np


# ═══════════════════════════════════════════════════════
# 缓动函数
# ═══════════════════════════════════════════════════════

class Easing:
    """常用缓动函数集合，所有函数接收 t∈[0,1] 返回插值结果∈[0,1]"""

    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in(t: float) -> float:
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        return 1 - (1 - t) * (1 - t)

    @staticmethod
    def ease_in_out(t: float) -> float:
        return t * t * (3 - 2 * t)

    @staticmethod
    def ease_out_back(t: float) -> float:
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * math.pow(t - 1, 3) + c1 * math.pow(t - 1, 2)

    @staticmethod
    def elastic(t: float) -> float:
        c4 = (2 * math.pi) / 3
        if t == 0:
            return 0
        if t == 1:
            return 1
        return math.pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

    @staticmethod
    def bounce(t: float) -> float:
        n1 = 7.5625
        d1 = 2.75
        if t < 1 / d1:
            return n1 * t * t
        elif t < 2 / d1:
            t -= 1.5 / d1
            return n1 * t * t + 0.75
        elif t < 2.5 / d1:
            t -= 2.25 / d1
            return n1 * t * t + 0.9375
        else:
            t -= 2.625 / d1
            return n1 * t * t + 0.984375


# ═══════════════════════════════════════════════════════
# 数值动画器
# ═══════════════════════════════════════════════════════

@dataclass
class ValueAnimator:
    """
    数值平滑动画器。

    用法：
        animator = ValueAnimator(target=0.5, duration=0.3)
        # 每帧：
        animator.update(dt)
        current_value = animator.value
    """
    target: float = 0.0
    duration: float = 0.3  # 秒
    easing: Callable[[float], float] = Easing.ease_out

    _value: float = field(default=0.0, repr=False)
    _start_value: float = field(default=0.0, repr=False)
    _start_time: Optional[float] = field(default=None, repr=False)
    _elapsed: float = field(default=0.0, repr=False)
    _running: bool = field(default=False, repr=False)

    @property
    def value(self) -> float:
        return self._value

    def set_target(self, new_target: float, duration: Optional[float] = None) -> None:
        """设置新的目标值，启动过渡动画"""
        if new_target == self.target and self._running:
            return
        self.target = new_target
        self._start_value = self._value
        self._elapsed = 0.0
        self._running = True
        if duration is not None:
            self.duration = duration

    def snap_to(self, value: float) -> None:
        """直接跳转到目标值（无动画）"""
        self._value = value
        self.target = value
        self._running = False

    def update(self, dt: float) -> None:
        """每帧调用，dt 为时间差（秒）"""
        if not self._running:
            return

        self._elapsed += dt
        if self._elapsed >= self.duration:
            self._value = self.target
            self._running = False
            return

        t = self._elapsed / self.duration
        eased = self.easing(t)
        self._value = self._start_value + (self.target - self._start_value) * eased

    def is_running(self) -> bool:
        return self._running


# ═══════════════════════════════════════════════════════
# 特征条动画集合（管理多个特征值的平滑过渡）
# ═══════════════════════════════════════════════════════

class FeatureBarAnimator:
    """
    管理多个特征进度条的动画过渡。

    自动为每个特征创建独立的 ValueAnimator，
    外部只需调用 set_feature() 更新目标值即可。
    """

    FEATURE_KEYS = ["motion_energy", "body_speed", "arm_amplitude", "body_rhythm"]

    def __init__(self, duration: float = 0.2):
        self.duration = duration
        self._animators: Dict[str, ValueAnimator] = {
            k: ValueAnimator(target=0.0, duration=duration, easing=Easing.ease_out)
            for k in self.FEATURE_KEYS
        }
        # 快照值（用于显示）
        self._values: Dict[str, float] = {k: 0.0 for k in self.FEATURE_KEYS}

    def set_features(self, features: Dict[str, Any]) -> None:
        """传入新的特征字典，自动启动过渡动画"""
        for key in self.FEATURE_KEYS:
            raw = features.get(key, 0.0)
            try:
                val = float(np.clip(raw, 0.0, 1.0))
            except (TypeError, ValueError):
                val = 0.0
            self._animators[key].set_target(val, self.duration)

    def update(self, dt: float) -> None:
        """每帧更新所有动画器"""
        for key, anim in self._animators.items():
            anim.update(dt)
            self._values[key] = anim.value

    def get(self, key: str) -> float:
        """获取当前动画值"""
        return self._values.get(key, 0.0)

    def get_all(self) -> Dict[str, float]:
        """获取所有动画值的副本"""
        return self._values.copy()


# ═══════════════════════════════════════════════════════
# 脉冲效果
# ═══════════════════════════════════════════════════════

class PulseEffect:
    """
    周期性脉冲效果。

    用于录制指示器、激活预设高亮等场景。
    """

    def __init__(self, period: float = 1.0, min_scale: float = 0.8, max_scale: float = 1.2):
        self.period = period
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._phase = 0.0

    def update(self, dt: float) -> None:
        """推进相位"""
        self._phase += dt / self.period
        self._phase %= 1.0

    def get_scale(self) -> float:
        """获取当前缩放值（基于正弦波）"""
        s = math.sin(self._phase * 2 * math.pi)
        return self.min_scale + (self.max_scale - self.min_scale) * (0.5 + 0.5 * s)

    def get_alpha(self) -> int:
        """获取当前透明度（0-255）"""
        s = math.sin(self._phase * 2 * math.pi)
        return int(100 + 80 * (0.5 + 0.5 * s))

    def get_phase(self) -> float:
        """获取当前相位 0~1"""
        return self._phase


# ═══════════════════════════════════════════════════════
# 录制指示器动画
# ═══════════════════════════════════════════════════════

class RecordingIndicator:
    """
    录制状态指示器动画。

    包含：
    - 脉冲红点（周期性放大缩小）
    - 进度环（录制时旋转）
    - 计时文字
    """

    def __init__(self, duration: float = 3.0):
        self.duration = duration
        self._elapsed = 0.0
        self._active = False
        self._pulse = PulseEffect(period=0.8, min_scale=0.9, max_scale=1.3)

    def start(self) -> None:
        self._active = True
        self._elapsed = 0.0

    def stop(self) -> None:
        self._active = False

    def update(self, dt: float) -> None:
        if self._active:
            self._elapsed += dt
            self._pulse.update(dt)

    def is_active(self) -> bool:
        return self._active

    def get_progress(self) -> float:
        """返回录制进度 0~1"""
        return min(self._elapsed / self.duration, 1.0)

    def get_pulse_scale(self) -> float:
        return self._pulse.get_scale()

    def get_pulse_alpha(self) -> int:
        return self._pulse.get_alpha()

    def get_time_text(self) -> str:
        """返回已录制时间文字"""
        return f"{self._elapsed:.1f}s"


# ═══════════════════════════════════════════════════════
# 状态过渡管理器
# ═══════════════════════════════════════════════════════

class StateTransition:
    """
    管理 UI 状态切换的过渡动画。

    支持：颜色过渡、位置过渡、透明度过渡。
    """

    def __init__(self, duration: float = 0.2):
        self.duration = duration
        self._color_anim = ValueAnimator(duration=duration, easing=Easing.ease_in_out)
        self._pos_anim = ValueAnimator(duration=duration, easing=Easing.ease_out)
        self._alpha_anim = ValueAnimator(duration=duration, easing=Easing.ease_in_out)

    def transition_to(self, color_t: float = 0.0, pos_t: float = 0.0, alpha_t: float = 0.0) -> None:
        """同时启动多个过渡"""
        self._color_anim.set_target(color_t)
        self._pos_anim.set_target(pos_t)
        self._alpha_anim.set_target(alpha_t)

    def update(self, dt: float) -> None:
        self._color_anim.update(dt)
        self._pos_anim.update(dt)
        self._alpha_anim.update(dt)

    def get_color_progress(self) -> float:
        return self._color_anim.value

    def get_position_progress(self) -> float:
        return self._pos_anim.value

    def get_alpha_progress(self) -> float:
        return self._alpha_anim.value


# ═══════════════════════════════════════════════════════
# 动作触发闪光效果
# ═══════════════════════════════════════════════════════

class ActionFlash:
    """
    动作触发时的短暂闪光效果。

    在触发位置显示一个快速扩散+淡出的圆环。
    """

    def __init__(self, x: int, y: int, max_radius: int = 60, duration: float = 0.5):
        self.x = x
        self.y = y
        self.max_radius = max_radius
        self.duration = duration
        self._elapsed = 0.0
        self._active = False

    def trigger(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """触发一次闪光"""
        self._active = True
        self._elapsed = 0.0
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

    def update(self, dt: float) -> None:
        if not self._active:
            return
        self._elapsed += dt
        if self._elapsed >= self.duration:
            self._active = False

    def is_active(self) -> bool:
        return self._active

    def get_radius(self) -> float:
        """当前扩散半径"""
        t = self._elapsed / self.duration if self.duration > 0 else 1
        return self.max_radius * Easing.ease_out(t)

    def get_alpha(self) -> int:
        """当前透明度（从 255 快速降到 0）"""
        t = self._elapsed / self.duration if self.duration > 0 else 1
        return int(255 * (1 - Easing.ease_in(t)))


# ═══════════════════════════════════════════════════════
# 全局动画管理器（方便 SoulPlayerApp 统一管理）
# ═══════════════════════════════════════════════════════

class AnimationManager:
    """
    SoulPlayer 的动画管理器，整合所有动画对象。

    在 SoulPlayerApp._update() 中每帧调用 update(dt) 即可。
    """

    def __init__(self, recording_duration: float = 3.0):
        self.feature_bars = FeatureBarAnimator(duration=0.2)
        self.recording = RecordingIndicator(duration=recording_duration)
        self.preset_pulse = PulseEffect(period=1.2, min_scale=1.0, max_scale=1.08)
        self.action_flash: Optional[ActionFlash] = None
        # 弹窗入场动画：0→1（ease_out_back 轻微回弹）；默认置于完成态
        self.modal_in = ValueAnimator(duration=0.25, easing=Easing.ease_out_back)
        self.modal_in.snap_to(1.0)

        # 时间跟踪
        self._last_time = time.time()

    def update(self) -> None:
        """计算 dt 并更新所有动画"""
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time

        # 限制 dt 防止挂起后跳变
        dt = min(dt, 0.1)

        self.feature_bars.update(dt)
        self.recording.update(dt)
        self.preset_pulse.update(dt)
        self.modal_in.update(dt)

        if self.action_flash:
            self.action_flash.update(dt)
            if not self.action_flash.is_active():
                self.action_flash = None

    def trigger_action_flash(self, x: int, y: int) -> None:
        """在指定位置触发动作闪光"""
        self.action_flash = ActionFlash(x, y, max_radius=80, duration=0.6)
        self.action_flash.trigger()

    # ── 弹窗入场动画 ──
    def open_modal(self) -> None:
        """播放弹窗入场动画（淡入 + 上滑 + 轻微回弹）"""
        self.modal_in.set_target(1.0)

    def close_modal(self) -> None:
        """弹窗关闭动画（值回落到 0）"""
        self.modal_in.set_target(0.0)

    def get_feature_animated(self, key: str) -> float:
        """获取平滑后的特征值"""
        return self.feature_bars.get(key)

    def get_recording_phase(self) -> float:
        """获取录制脉冲相位"""
        return self.recording.get_pulse_scale() if self.recording.is_active() else 0.0

    def get_preset_pulse_scale(self) -> float:
        """获取预设脉冲缩放"""
        return self.preset_pulse.get_scale()

    def get_preset_pulse_alpha(self) -> int:
        return self.preset_pulse.get_alpha()

    # ── 委托方法 ──
    def start_recording(self) -> None:
        self.recording.start()

    def stop_recording(self) -> None:
        self.recording.stop()

    def set_feature_targets(self, features: Dict[str, Any]) -> None:
        self.feature_bars.set_features(features)
