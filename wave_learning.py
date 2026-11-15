"""
wave_learning.py - 挥手动作自适应学习系统

实现两个核心学习机制：
- P1（自适应阈值）：根据每只手每个方向的历史触发成功率动态调整触发阈值
- P2（幅度归一化）：记录用户历史最大动作幅度，以相对比例判定是否为有效挥手

设计目标：
1. 解决单手模式下某个方向触发率低的问题（如左手横向运动幅度弱）
2. 个性化适配不同用户、不同使用习惯
3. 快速收敛（5-50次挥手内基本稳定）
4. 轻量无依赖（纯numpy实现）

学习周期：
- 冷启动期（0-5次）：仅记录，不调整
- 初步学习（5-15次）：加速学习
- 基本稳定（50-100次）：阈值收敛
- 完全稳定（100+次）：持续微调
"""

import json
import os
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# 导入白名单规则验证
from wave_learning_rules import validate_and_filter_trigger


# ---------------------------------------------------------------------------
#  数据类：学习统计
# ---------------------------------------------------------------------------

@dataclass
class HandDirectionStats:
    """单只手单方向的统计信息"""
    hand: str          # 'L' 或 'R'
    direction: float   # +1.0 或 -1.0

    # P1 自适应阈值统计
    success: int = 0       # 成功触发次数
    failure: int = 0        # 失败次数（检测到挥手但未触发）
    total_attempts: int = 0 # 总尝试次数

    # P2 幅度归一化统计
    max_x_span: float = 0.0      # 历史最大x跨度
    max_y_span: float = 0.0      # 历史最大y跨度
    max_ang_span: float = 0.0    # 历史最大角度跨度
    amplitude_samples: int = 0   # 幅度采样次数

    # 学习状态
    last_update_time: float = 0.0  # 最后更新时间戳
    is_learning: bool = True       # 是否处于学习期

    @property
    def success_rate(self) -> float:
        """成功率（0~1）"""
        total = self.success + self.failure
        return self.success / total if total > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        """失败率（0~1）"""
        total = self.success + self.failure
        return self.failure / total if total > 0 else 0.0

    @property
    def is_stable(self) -> bool:
        """是否已稳定（样本≥50且最近10次无失败）"""
        return (self.success + self.failure >= 50
                and self.failure_rate < 0.1)


# ---------------------------------------------------------------------------
#  P1：自适应阈值管理器
# ---------------------------------------------------------------------------

class WaveAdaptiveThreshold:
    """P1 自适应阈值管理器

    根据每只手每个方向的历史触发成功率动态调整触发阈值：
    - 失败率高 → 降低阈值（提升触发率）
    - 成功率高 → 提高阈值（降低误触率）

    学习策略：
    - 冷启动（0-5次）：不调整，仅记录
    - 加速学习（5-15次）：学习率翻倍，快速适配
    - 正常学习（15+次）：标准学习率，稳定调整
    - 稳定后：微调，防止剧烈波动
    """

    def __init__(self,
                 base_x_threshold: float = 0.06,
                 min_x_threshold: float = 0.03,
                 learning_rate_fast: float = 0.02,
                 learning_rate_normal: float = 0.01,
                 stable_threshold: int = 50):
        """
        Args:
            base_x_threshold: 默认x跨度阈值
            min_x_threshold: 最低x跨度阈值（保护机制）
            learning_rate_fast: 加速学习期学习率
            learning_rate_normal: 正常学习期学习率
            stable_threshold: 稳定判定样本数
        """
        self.base_x_threshold = base_x_threshold
        self.min_x_threshold = min_x_threshold
        self.learning_rate_fast = learning_rate_fast
        self.learning_rate_normal = learning_rate_normal
        self.stable_threshold = stable_threshold

        # 统计数据：{(hand, direction): HandDirectionStats}
        self.stats: Dict[Tuple[str, float], HandDirectionStats] = {}

        # 当前阈值缓存
        self._current_thresholds: Dict[Tuple[str, float], float] = {}

    def get_or_create_stats(self, hand: str, direction: float) -> HandDirectionStats:
        """获取或创建统计数据"""
        key = (hand, direction)
        if key not in self.stats:
            self.stats[key] = HandDirectionStats(
                hand=hand,
                direction=direction,
                last_update_time=time.time()
            )
        return self.stats[key]

    def record_attempt(self, hand: str, direction: float, triggered: bool):
        """记录一次尝试（成功/失败）

        Args:
            hand: 手别 'L' 或 'R'
            direction: 方向 +1.0 或 -1.0
            triggered: 是否成功触发动作
        """
        stats = self.get_or_create_stats(hand, direction)
        stats.total_attempts += 1
        stats.last_update_time = time.time()

        if triggered:
            stats.success += 1
        else:
            stats.failure += 1

        # 更新阈值
        self._update_threshold(stats)

    def update_amplitude(self, hand: str, direction: float,
                        x_span: float, y_span: float, ang_span: float):
        """更新动作幅度统计（用于P2）

        Args:
            hand: 手别
            direction: 方向
            x_span: x方向跨度
            y_span: y方向跨度
            ang_span: 角度跨度
        """
        stats = self.get_or_create_stats(hand, direction)
        stats.max_x_span = max(stats.max_x_span, x_span)
        stats.max_y_span = max(stats.max_y_span, y_span)
        stats.max_ang_span = max(stats.max_ang_span, ang_span)
        stats.amplitude_samples += 1

    def get_threshold(self, hand: str, direction: float) -> Tuple[float, str]:
        """获取当前触发阈值

        Args:
            hand: 手别
            direction: 方向

        Returns:
            (threshold, reason) - 阈值和调整原因
        """
        key = (hand, direction)

        # 无统计数据，使用默认值
        if key not in self.stats:
            return self.base_x_threshold, "default"

        stats = self.stats[key]
        total = stats.success + stats.failure

        # 冷启动期：不调整
        if total < 5:
            return self.base_x_threshold, "cold_start"

        # 有缓存且稳定，直接返回
        if key in self._current_thresholds and stats.is_stable:
            return self._current_thresholds[key], "stable"

        # 计算当前阈值
        threshold = self._current_thresholds.get(key, self.base_x_threshold)
        return threshold, "learning"

    def _update_threshold(self, stats: HandDirectionStats):
        """内部方法：根据统计数据更新阈值"""
        key = (stats.hand, stats.direction)
        total = stats.success + stats.failure

        # 冷启动期不调整
        if total < 5:
            return

        # 获取当前阈值
        current = self._current_thresholds.get(key, self.base_x_threshold)

        # 选择学习率
        if total < 15:
            rate = self.learning_rate_fast
        elif total < self.stable_threshold:
            rate = self.learning_rate_normal
        else:
            # 稳定后：仅当失败率>20%或成功率>90%时才调整
            if not (0.2 < stats.failure_rate < 0.1 or stats.success_rate > 0.9):
                return
            rate = self.learning_rate_normal * 0.5  # 微调模式

        # 根据失败率调整阈值
        if stats.failure_rate > 0.5:
            # 失败率>50%：降低阈值（提升触发率）
            new_threshold = max(current - rate, self.min_x_threshold)
        elif stats.success_rate > 0.9 and total > 20:
            # 成功率>90%且样本>20：提高阈值（降低误触）
            new_threshold = min(current + rate, self.base_x_threshold * 1.2)
        else:
            # 保持现状
            return

        self._current_thresholds[key] = new_threshold

    def get_stats_summary(self) -> Dict:
        """获取统计摘要（用于UI显示/调试）"""
        summary = {}
        for key, stats in self.stats.items():
            hand, direction = key
            threshold, reason = self.get_threshold(hand, direction)
            summary[f"{hand}_{'next' if direction > 0 else 'prev'}"] = {
                "hand": hand,
                "direction": "next" if direction > 0 else "prev",
                "success": stats.success,
                "failure": stats.failure,
                "success_rate": f"{stats.success_rate:.1%}",
                "current_threshold": round(threshold, 3),
                "max_x_span": round(stats.max_x_span, 3),
                "is_stable": stats.is_stable,
                "status": reason
            }
        return summary

    def save(self, filepath: str):
        """保存学习数据到文件"""
        data = {
            "thresholds": {f"{k[0]}_{k[1]}": v for k, v in self._current_thresholds.items()},
            "stats": {f"{k[0]}_{k[1]}": asdict(v) for k, v in self.stats.items()}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """从文件加载学习数据"""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载阈值
        if "thresholds" in data:
            for key_str, threshold in data["thresholds"].items():
                hand, direction_str = key_str.split('_')
                direction = float(direction_str)
                self._current_thresholds[(hand, direction)] = threshold

        # 加载统计
        if "stats" in data:
            for key_str, stats_dict in data["stats"].items():
                hand, direction_str = key_str.split('_')
                direction = float(direction_str)
                self.stats[(hand, direction)] = HandDirectionStats(**stats_dict)


# ---------------------------------------------------------------------------
#  P2：幅度归一化器
# ---------------------------------------------------------------------------

class AmplitudeNormalizer:
    """P2 幅度归一化器

    记录用户历史最大动作幅度，以相对比例判定是否为有效挥手：
    - 不依赖绝对阈值（避免左手/右手差异）
    - 当前幅度达到历史最大幅度的50%即视为有效挥手
    - 内置最小阈值保护（防止冷启动期门槛过低）

    优势：
    - 完美适配单只手的双向运动差异
    - 自动适配不同用户的动作习惯
    - 冷启动友好（有保护阈值）
    """

    def __init__(self,
                 min_x_threshold: float = 0.04,
                 min_ratio: float = 0.5,
                 max_ratio: float = 0.7,
                 window_size: int = 20):
        """
        Args:
            min_x_threshold: 最小x跨度阈值（保护机制）
            min_ratio: 最小触发比例（历史最大幅度的百分比）
            max_ratio: 最大触发比例（防止门槛过低）
            window_size: 滑动窗口大小（用于计算滑动最大幅度）
        """
        self.min_x_threshold = min_x_threshold
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.window_size = window_size

        # 全局最大幅度
        self.global_max: Dict[Tuple[str, float], Dict[str, float]] = {}

        # 滑动窗口最大幅度（最近N次）
        self.window_max: Dict[Tuple[str, float], Dict[str, float]] = {}

        # 窗口历史
        self.history: Dict[Tuple[str, float], list] = {}

    def record_attempt(self, hand: str, direction: float,
                      x_span: float, y_span: float, ang_span: float):
        """记录一次挥手动作的幅度

        Args:
            hand: 手别
            direction: 方向
            x_span: x方向跨度
            y_span: y方向跨度
            ang_span: 角度跨度
        """
        key = (hand, direction)

        # 更新全局最大幅度
        if key not in self.global_max:
            self.global_max[key] = {"x_span": 0.0, "y_span": 0.0, "ang_span": 0.0}

        self.global_max[key]["x_span"] = max(self.global_max[key]["x_span"], x_span)
        self.global_max[key]["y_span"] = max(self.global_max[key]["y_span"], y_span)
        self.global_max[key]["ang_span"] = max(self.global_max[key]["ang_span"], ang_span)

        # 更新滑动窗口
        if key not in self.history:
            self.history[key] = []

        self.history[key].append({
            "x_span": x_span,
            "y_span": y_span,
            "ang_span": ang_span,
            "timestamp": time.time()
        })

        # 保持窗口大小
        if len(self.history[key]) > self.window_size:
            self.history[key].pop(0)

        # 计算滑动窗口最大幅度
        if self.history[key]:
            self.window_max[key] = {
                "x_span": max(h["x_span"] for h in self.history[key]),
                "y_span": max(h["y_span"] for h in self.history[key]),
                "ang_span": max(h["ang_span"] for h in self.history[key])
            }

    def is_wave(self, hand: str, direction: float,
                x_span: float, y_span: float, ang_span: float) -> Tuple[bool, str]:
        """判断是否为有效挥手（基于幅度归一化）

        Args:
            hand: 手别
            direction: 方向
            x_span: 当前x跨度
            y_span: 当前y跨度
            ang_span: 当前角度跨度

        Returns:
            (is_wave, reason) - 是否为有效挥手及原因
        """
        key = (hand, direction)

        # 无历史数据，使用保护阈值
        if key not in self.global_max or self.global_max[key]["x_span"] < 0.01:
            return x_span >= self.min_x_threshold, "no_history"

        # 获取滑动窗口最大幅度（优先）/ 全局最大幅度
        max_x = self.window_max.get(key, {}).get("x_span", self.global_max[key]["x_span"])
        max_ang = self.window_max.get(key, {}).get("ang_span", self.global_max[key]["ang_span"])

        # 计算触发阈值（取min_ratio和max_ratio之间的值）
        threshold_ratio = max(self.min_ratio, min(self.max_ratio, self.min_ratio))
        threshold = max_x * threshold_ratio

        # 多维度判断（x跨度为主，角度跨度为辅）
        x_pass = x_span >= threshold
        ang_pass = ang_span >= max_ang * 0.4  # 角度达到历史最大40%

        if x_pass and ang_pass:
            return True, f"amplitude_ok(x={x_span:.3f}>={threshold:.3f})"
        elif x_pass:
            return True, f"x_ok_ang_low(x={x_span:.3f})"
        elif ang_span >= max_ang * 0.7:
            return True, f"ang_ok_x_low(ang={ang_span:.3f})"
        else:
            return False, f"amplitude_low(x={x_span:.3f}<{threshold:.3f})"

    def get_threshold(self, hand: str, direction: float) -> Tuple[float, float]:
        """获取当前归一化触发阈值

        Returns:
            (x_threshold, ang_threshold)
        """
        key = (hand, direction)

        if key not in self.global_max or self.global_max[key]["x_span"] < 0.01:
            return self.min_x_threshold, 0.20

        max_x = self.window_max.get(key, {}).get("x_span", self.global_max[key]["x_span"])
        max_ang = self.window_max.get(key, {}).get("ang_span", self.global_max[key]["ang_span"])

        x_threshold = max(max_x * self.min_ratio, self.min_x_threshold)
        ang_threshold = max_ang * 0.4

        return x_threshold, ang_threshold

    def get_stats_summary(self) -> Dict:
        """获取统计摘要"""
        summary = {}
        for key, max_vals in self.global_max.items():
            hand, direction = key
            x_th, ang_th = self.get_threshold(hand, direction)
            samples = len(self.history.get(key, []))

            summary[f"{hand}_{'next' if direction > 0 else 'prev'}"] = {
                "hand": hand,
                "direction": "next" if direction > 0 else "prev",
                "max_x_span": round(max_vals["x_span"], 3),
                "max_ang_span": round(max_vals["ang_span"], 3),
                "samples": samples,
                "current_x_threshold": round(x_th, 3),
                "current_ang_threshold": round(ang_th, 3)
            }
        return summary

    def save(self, filepath: str):
        """保存学习数据"""
        data = {
            "global_max": {f"{k[0]}_{k[1]}": v for k, v in self.global_max.items()},
            "window_max": {f"{k[0]}_{k[1]}": v for k, v in self.window_max.items()},
            "history_size": {f"{k[0]}_{k[1]}": len(v) for k, v in self.history.items()}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str):
        """加载学习数据"""
        if not os.path.exists(filepath):
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "global_max" in data:
            for key_str, max_vals in data["global_max"].items():
                hand, direction_str = key_str.split('_')
                direction = 1.0 if direction_str == "next" else -1.0
                self.global_max[(hand, direction)] = max_vals


# ---------------------------------------------------------------------------
#  统一学习管理器（P1+P2组合）
# ---------------------------------------------------------------------------

class WaveLearningManager:
    """挥手动作统一学习管理器

    整合P1（自适应阈值）和P2（幅度归一化）：
    - P1负责基础阈值调整（应对触发率偏低/过高）
    - P2负责个性化幅度适配（应对单手不对称性）
    - 两者结合：最终阈值 = max(P1阈值, P2归一化阈值)
    """

    def __init__(self,
                 data_dir: str = "learning_data",
                 enable_p1: bool = True,
                 enable_p2: bool = True):
        """
        Args:
            data_dir: 学习数据保存目录
            enable_p1: 启用P1自适应阈值
            enable_p2: 启用P2幅度归一化
        """
        self.data_dir = data_dir
        self.enable_p1 = enable_p1
        self.enable_p2 = enable_p2

        # P1 和 P2 实例
        if enable_p1:
            self.p1 = WaveAdaptiveThreshold()
        if enable_p2:
            self.p2 = AmplitudeNormalizer()

        # 数据文件路径
        self.data_file = os.path.join(data_dir, "wave_learning.json")

        # 加载已有数据
        self._load()

    def record_wave_attempt(self, hand: str, direction: float,
                           x_span: float, y_span: float, ang_span: float,
                           triggered: bool, confidence: float = 1.0):
        """记录一次挥手尝试

        Args:
            hand: 手别 'L'/'R'
            direction: 方向 +1/-1
            x_span: x方向跨度
            y_span: y方向跨度
            ang_span: 角度跨度
            triggered: 是否成功触发
            confidence: 触发置信度（0~1），低置信度的触发可能被当作"失败"处理
        """
        # 置信度过滤：如果置信度<0.5，即使triggered=True也视为失败
        # 防止低置信度的误触被学习系统当作"正确结果"
        effective_triggered = triggered and confidence >= 0.5

        # P1：记录触发结果（只记录有效触发）
        if self.enable_p1:
            self.p1.record_attempt(hand, direction, effective_triggered)

        # P2：记录幅度（无论是否触发都记录，用于学习用户动作幅度）
        if self.enable_p2:
            self.p2.record_attempt(hand, direction, x_span, y_span, ang_span)

    def record_visual_features(self, hand: str, direction: float,
                                wave_direction: float,
                                arm_motion: float,
                                x_span: float = 0.0,
                                ang_span: float = 0.0,
                                triggered: bool = False):
        """记录视觉特征数据（用于白名单学习和数据库存储）

        Args:
            hand: 手别 'L'/'R'
            direction: 方向 +1/-1
            wave_direction: 实际检测到的方向幅度（归一化）
            arm_motion: 手臂横向运动幅度（归一化）
            x_span: x方向跨度（可选）
            ang_span: 角度跨度（可选，单位：度）
            triggered: 是否成功触发
        """
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 记录视觉特征数据文件
        visual_data_file = os.path.join(self.data_dir, "visual_features.json")
        visual_data = {"records": []}

        # 加载已有数据
        if os.path.exists(visual_data_file):
            try:
                with open(visual_data_file, 'r', encoding='utf-8') as f:
                    visual_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                visual_data = {"records": []}

        # 添加新记录
        record = {
            "timestamp": time.time(),
            "hand": hand,
            "direction": direction,
            "wave_direction": wave_direction,
            "arm_motion": arm_motion,
            "x_span": x_span,
            "ang_span": ang_span,
            "triggered": triggered,
        }

        visual_data["records"].append(record)

        # 只保留最近1000条记录（防止文件过大）
        if len(visual_data["records"]) > 1000:
            visual_data["records"] = visual_data["records"][-1000:]

        # 保存
        try:
            with open(visual_data_file, 'w', encoding='utf-8') as f:
                json.dump(visual_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[WaveLearning] 视觉特征数据保存失败: {e}")

    def get_threshold(self, hand: str, direction: float) -> Tuple[float, float, str]:
        """获取触发阈值

        Returns:
            (x_threshold, ang_threshold, strategy) - x阈值、角度阈值、使用策略
        """
        x_th = 0.06  # 默认值
        ang_th = 0.20
        strategy = "default"

        # P1 阈值
        if self.enable_p1:
            p1_th, p1_reason = self.p1.get_threshold(hand, direction)
            x_th = p1_th
            strategy = f"P1({p1_reason})"

        # P2 阈值（取较严格者）
        if self.enable_p2:
            p2_x_th, p2_ang_th = self.p2.get_threshold(hand, direction)
            x_th = max(x_th, p2_x_th)
            ang_th = max(ang_th, p2_ang_th)
            strategy += "+P2"

        return x_th, ang_th, strategy

    def is_wave(self, hand: str, direction: float,
                x_span: float, y_span: float, ang_span: float) -> Tuple[bool, str]:
        """综合判断是否为有效挥手

        Args:
            hand: 手别
            direction: 方向
            x_span: x跨度
            y_span: y跨度
            ang_span: 角度跨度

        Returns:
            (is_wave, reason) - 是否为挥手及原因
        """
        # P2 幅度归一化判断
        if self.enable_p2:
            is_wave_p2, reason_p2 = self.p2.is_wave(hand, direction, x_span, y_span, ang_span)
            if is_wave_p2:
                return True, f"P2:{reason_p2}"

        # 回退到绝对阈值判断
        x_th, ang_th, _ = self.get_threshold(hand, direction)
        x_pass = x_span >= x_th
        ang_pass = ang_span >= ang_th

        if x_pass and ang_pass:
            return True, f"threshold_ok(x={x_span:.3f}>={x_th:.3f})"
        else:
            return False, f"threshold_fail(x={x_span:.3f}<{x_th:.3f})"

    def get_learning_summary(self) -> Dict:
        """获取学习摘要（用于UI/日志）"""
        summary = {"P1": {}, "P2": {}, "visual": {}}

        if self.enable_p1:
            summary["P1"] = self.p1.get_stats_summary()
        if self.enable_p2:
            summary["P2"] = self.p2.get_stats_summary()

        # 视觉特征统计
        visual_data_file = os.path.join(self.data_dir, "visual_features.json")
        if os.path.exists(visual_data_file):
            try:
                with open(visual_data_file, 'r', encoding='utf-8') as f:
                    visual_data = json.load(f)
                records = visual_data.get("records", [])
                if records:
                    # 统计最近100条
                    recent = records[-100:]
                    summary["visual"] = {
                        "total_records": len(records),
                        "recent_100": {
                            "avg_wave_direction": round(sum(r["wave_direction"] for r in recent) / len(recent), 3),
                            "avg_arm_motion": round(sum(r["arm_motion"] for r in recent) / len(recent), 3),
                            "trigger_rate": round(sum(1 for r in recent if r["triggered"]) / len(recent), 3),
                        }
                    }
            except (json.JSONDecodeError, IOError):
                pass

        # 学习进度
        total_attempts = sum(
            stats.total_attempts
            for stats in self.p1.stats.values()
        ) if self.enable_p1 else 0

        summary["progress"] = {
            "total_attempts": total_attempts,
            "estimated_days": max(1, (50 - total_attempts // 3) // 10) if total_attempts < 50 else 0,
            "stable_combinations": sum(1 for s in self.p1.stats.values() if s.is_stable) if self.enable_p1 else 0
        }

        return summary

    def save(self):
        """保存学习数据"""
        if self.enable_p1:
            self.p1.save(self.data_file)
        elif self.enable_p2:
            self.p2.save(self.data_file)

    def _load(self):
        """加载学习数据"""
        if self.enable_p1:
            self.p1.load(self.data_file)
        elif self.enable_p2:
            self.p2.load(self.data_file)

    def reset(self):
        """重置所有学习数据"""
        if self.enable_p1:
            self.p1 = WaveAdaptiveThreshold()
        if self.enable_p2:
            self.p2 = AmplitudeNormalizer()
        self.save()


# ---------------------------------------------------------------------------
#  工具函数：集成到现有代码
# ---------------------------------------------------------------------------

def create_wave_learning_manager(config: Optional[Dict] = None) -> WaveLearningManager:
    """创建学习管理器（工厂函数）

    Args:
        config: 配置字典，支持以下键：
            - data_dir: 数据目录
            - enable_p1: 是否启用P1
            - enable_p2: 是否启用P2
            - base_x_threshold: 默认x阈值
            - min_x_threshold: 最小x阈值

    Returns:
        WaveLearningManager 实例
    """
    config = config or {}

    return WaveLearningManager(
        data_dir=config.get("data_dir", "learning_data"),
        enable_p1=config.get("enable_p1", True),
        enable_p2=config.get("enable_p2", True)
    )


# ---------------------------------------------------------------------------
#  测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("挥手动作自适应学习系统 - 测试")
    print("=" * 60)

    # 创建学习管理器
    manager = WaveLearningManager(enable_p1=True, enable_p2=True)

    # 模拟场景：左手向左挥（触发率低的方向）
    print("\n场景：左手向左挥手（触发率偏低的方向）")
    print("-" * 60)

    # 模拟20次挥手（前10次失败，后10次成功）
    for i in range(20):
        hand = 'L'
        direction = 1.0  # 向左

        # 左手横向幅度通常较小
        x_span = 0.035 + np.random.normal(0, 0.01)
        y_span = 0.08 + np.random.normal(0, 0.02)
        ang_span = 0.6 + np.random.normal(0, 0.1)

        # 前10次失败，后10次成功
        triggered = (i >= 10)

        # 记录
        manager.record_wave_attempt(hand, direction, x_span, y_span, ang_span, triggered, confidence=0.8)

        # 获取当前阈值
        x_th, ang_th, strategy = manager.get_threshold(hand, direction)
        is_wave, reason = manager.is_wave(hand, direction, x_span, y_span, ang_span)

        if i < 5 or i % 5 == 4:
            print(f"[{i+1:2d}] x={x_span:.3f} th={x_th:.3f} "
                  f"triggered={triggered} wave={is_wave} ({reason})")

    # 打印学习摘要
    print("\n" + "=" * 60)
    print("学习摘要")
    print("=" * 60)
    summary = manager.get_learning_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 测试保存/加载
    print("\n保存学习数据到 learning_data/wave_learning.json...")
    manager.save()
    print("[OK] 保存完成")

    # 创建新管理器并加载
    print("\n创建新管理器并加载数据...")
    manager2 = WaveLearningManager(enable_p1=True, enable_p2=True)
    summary2 = manager2.get_learning_summary()
    print("[OK] 加载完成，已记录 {} 次尝试".format(summary2['progress']['total_attempts']))

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
