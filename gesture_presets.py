"""
gesture_presets.py - 动作预设系统

负责管理动作预设的定义、加载/保存，以及基于动作特征的实时匹配引擎。
是 SoulPlayer 的核心决策模块：将 motion_capture 输出的特征向量映射为具体的播放控制动作。
"""

import json
import os
import time
from typing import Dict, List, Optional, Any


class GesturePreset:
    """单个动作预设定义"""
    
    def __init__(self, name: str, description: str, 
                 conditions: dict, action: str, priority: int = 0):
        """
        初始化动作预设
        
        Args:
            name: 预设名称，用于唯一标识
            description: 预设的文字描述，便于UI展示
            conditions: 触发条件字典，包含各特征的范围/列表/布尔条件
                数值特征: {'min': float, 'max': float}
                枚举特征: [str]（允许的取值列表）
                布尔特征: bool
                duration: float（需要持续满足的秒数）
            action: 触发动作类型，如 'mood_calm', 'player_pause' 等
            priority: 优先级，数值越大优先级越高，冲突时优先触发
        """
        self.name = name
        self.description = description
        self.conditions = conditions
        self.action = action
        self.priority = priority
    
    def to_dict(self) -> dict:
        """将预设转换为字典格式（用于JSON序列化）"""
        return {
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "action": self.action,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GesturePreset":
        """从字典反序列化创建预设实例"""
        # 校验必要字段，防止缺失关键信息导致运行时错误
        required_keys = ["name", "description", "conditions", "action"]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"预设数据缺少必要字段: '{key}', 数据: {data}")
        
        return cls(
            name=data["name"],
            description=data["description"],
            conditions=data["conditions"],
            action=data["action"],
            priority=data.get("priority", 0)
        )
    
    def __repr__(self) -> str:
        return (
            f"GesturePreset("
            f"name='{self.name}', "
            f"action='{self.action}', "
            f"priority={self.priority}"
            f")"
        )


class PresetEngine:
    """
    预设匹配引擎
    
    核心职责：根据实时动作特征，匹配并返回最高优先级的触发动作。
    匹配采用"持续时间阈值"机制，避免动作抖动导致的误触发。
    """
    
    # 支持数值范围检查的特征键（对应 motion_features 中的 float 值）
    _NUMERIC_KEYS = {"motion_energy", "body_speed", "arm_amplitude", "body_rhythm", "arm_motion_energy", "arm_horizontal_motion", "wave_direction", "palm_hover"}
    # 支持列表/枚举匹配的特征键
    _LIST_KEYS = {"posture", "hand_position"}
    # 支持布尔匹配的特征键
    _BOOL_KEYS = {"is_active"}
    
    def __init__(self):
        """初始化预设匹配引擎"""
        self.presets: List[GesturePreset] = []
        self.active_preset: Optional[GesturePreset] = None
        # 记录每个预设的持续满足时间（秒），键为预设名称
        self.hold_timers: Dict[str, float] = {}
        # 记录上次 match 调用的时间戳，用于计算时间差
        self._last_match_time: Optional[float] = None
    
    def load_presets(self, filepath: str):
        """
        从JSON文件加载预设配置
        
        Args:
            filepath: JSON文件路径，文件格式应符合架构文档定义
            
        Raises:
            FileNotFoundError: 文件不存在时抛出
            json.JSONDecodeError: JSON格式错误时抛出
            KeyError: 缺少 'presets' 字段时抛出
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"预设文件不存在: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "presets" not in data:
            raise KeyError("JSON文件中缺少 'presets' 字段")
        
        # 加载新预设前清空旧状态，防止状态污染
        self.presets = []
        self.hold_timers = {}
        self.active_preset = None
        self._last_match_time = None
        
        for i, preset_data in enumerate(data["presets"]):
            try:
                preset = GesturePreset.from_dict(preset_data)
                self.add_preset(preset)
            except (KeyError, ValueError) as e:
                # 单条预设加载失败时记录并继续，保证其他预设可用
                print(f"[PresetEngine] 加载第 {i+1} 条预设失败: {e}")
                continue
    
    def save_presets(self, filepath: str):
        """
        保存当前所有预设到JSON文件
        
        Args:
            filepath: 保存路径，若父目录不存在则自动创建
        """
        data = {
            "presets": [preset.to_dict() for preset in self.presets]
        }
        
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def add_preset(self, preset: GesturePreset):
        """
        添加预设到引擎
        
        Args:
            preset: GesturePreset 实例
            
        Raises:
            ValueError: 已存在同名预设时抛出
        """
        if any(p.name == preset.name for p in self.presets):
            raise ValueError(f"已存在名为 '{preset.name}' 的预设")
        
        self.presets.append(preset)
        self.hold_timers[preset.name] = 0.0  # 初始化计时器
    
    def remove_preset(self, name: str):
        """
        删除指定名称的预设
        
        Args:
            name: 要删除的预设名称
            
        Raises:
            ValueError: 不存在该预设时抛出
        """
        for i, preset in enumerate(self.presets):
            if preset.name == name:
                self.presets.pop(i)
                self.hold_timers.pop(name, None)
                # 若删除的是当前激活的预设，重置状态防止悬空引用
                if self.active_preset and self.active_preset.name == name:
                    self.active_preset = None
                return
        
        raise ValueError(f"不存在名为 '{name}' 的预设")
    
    def _check_conditions(self, preset: GesturePreset, motion_features: dict) -> bool:
        """
        检查单个预设的所有条件是否满足
        
        对 conditions 中的每个特征键，与 motion_features 中的对应值做比较。
        数值特征检查 [min, max] 闭区间范围，枚举特征检查是否在允许列表内，
        布尔特征检查是否严格相等。
        
        Args:
            preset: 要检查的预设
            motion_features: 当前动作特征向量
            
        Returns:
            bool: 所有条件均满足返回 True，任一不满足返回 False
        """
        conditions = preset.conditions
        
        # 检查数值范围条件（motion_energy, body_speed, arm_amplitude, body_rhythm）
        for key in self._NUMERIC_KEYS:
            if key not in conditions:
                continue
            if key not in motion_features:
                return False  # 缺少必要特征，视为不匹配
            
            value = motion_features[key]
            cond = conditions[key]
            
            if "min" in cond and value < cond["min"]:
                return False
            if "max" in cond and value > cond["max"]:
                return False
        
        # 检查枚举/列表条件（posture, hand_position）
        for key in self._LIST_KEYS:
            if key not in conditions:
                continue
            if key not in motion_features:
                return False
            
            value = motion_features[key]
            allowed_values = conditions[key]
            # 统一支持传入单个字符串或列表
            if isinstance(allowed_values, list):
                if value not in allowed_values:
                    return False
            else:
                if value != allowed_values:
                    return False
        
        # 检查布尔条件（is_active）
        for key in self._BOOL_KEYS:
            if key not in conditions:
                continue
            if key not in motion_features:
                return False
            
            if motion_features[key] != conditions[key]:
                return False
        
        return True
    
    def match(self, motion_features: dict) -> Optional[str]:
        """
        根据动作特征匹配预设，返回触发的 action
        
        匹配逻辑：
        1. 计算与上次调用的时间差 delta_time（基于 motion_features 中的 timestamp）
        2. 对每个预设，检查条件是否满足：
           - 满足：累加 hold_timer（持续时间计时器）
           - 不满足：立即清零 hold_timer
        3. 当 hold_timer >= duration 阈值时，该预设进入可触发状态
        4. 在可触发的预设中，按 priority 最高选出最优预设
        5. 更新 active_preset，返回其 action
        6. 没有任何预设可触发时返回 None，并清空 active_preset
        
        这种"持续时间阈值"机制可有效过滤瞬时噪声，避免动作抖动导致的误触发。
        
        Args:
            motion_features: 动作特征字典，建议包含 timestamp 字段（float）
            
        Returns:
            str | None: 触发的动作类型，如 'mood_calm' / 'player_pause' 等；无匹配时返回 None
        """
        # 使用特征中的时间戳作为当前时间基准（与 motion_capture 模块的时间对齐）
        # 若特征中无 timestamp，则回退到系统时间
        current_timestamp = motion_features.get("timestamp", time.time())
        
        # 计算与上次调用的时间差
        if self._last_match_time is None:
            delta_time = 0.0
        else:
            delta_time = current_timestamp - self._last_match_time
            # 防御异常时间差：时钟回拨或长时间未调用时重置，避免计时器异常累积
            if delta_time < 0 or delta_time > 5.0:
                delta_time = 0.0
        
        self._last_match_time = current_timestamp
        
        matched_presets = []  # 已达到持续时间阈值、可触发的预设列表
        custom_holding = False  # 有自定义预设正在匹配中（含未达 duration 的保持期）

        for preset in self.presets:
            preset_name = preset.name
            # 获取持续时间阈值，默认为 0.0（即立即触发）
            duration = preset.conditions.get("duration", 0.0)

            # 检查条件是否满足
            is_matching = self._check_conditions(preset, motion_features)

            if is_matching:
                if getattr(preset, '_is_custom', False):
                    custom_holding = True
                # 条件满足：累加持续时间计时器
                self.hold_timers[preset_name] = (
                    self.hold_timers.get(preset_name, 0.0) + delta_time
                )

                # 检查是否达到持续时间阈值
                if self.hold_timers[preset_name] >= duration:
                    matched_presets.append(preset)
            else:
                # 条件不满足：立即清零计时器，避免条件恢复后直接继承旧时间
                self.hold_timers[preset_name] = 0.0

        # 自定义动作匹配期间（含保持期），屏蔽所有内置动作：
        # 内置预设计时器清零、不参与本轮触发——结构性杜绝
        # "做自定义动作时误触切歌/上一首/暂停"
        if custom_holding:
            matched_presets = [
                p for p in matched_presets if getattr(p, '_is_custom', False)
            ]
            for p in self.presets:
                if not getattr(p, '_is_custom', False):
                    self.hold_timers[p.name] = 0.0
        
        if not matched_presets:
            self.active_preset = None
            return None
        
        # 按优先级选出最高者，优先级相同时取列表中排前者（先定义者优先）
        best_preset = max(matched_presets, key=lambda p: p.priority)
        self.active_preset = best_preset
        
        return best_preset.action
    
    def get_preset_list(self) -> List[dict]:
        """
        获取所有预设的列表信息（用于UI展示或调试面板）
        
        Returns:
            list: 包含每个预设的详细信息字典，包括当前 hold_time 和激活状态
        """
        return [
            {
                "name": p.name,
                "description": p.description,
                "action": p.action,
                "priority": p.priority,
                "conditions": p.conditions,
                "hold_time": round(self.hold_timers.get(p.name, 0.0), 2),
                "is_active": self.active_preset == p
            }
            for p in self.presets
        ]
    
    def reset(self):
        """重置引擎状态，清空所有计时器和激活状态（用于场景切换或调试）"""
        self.hold_timers = {p.name: 0.0 for p in self.presets}
        self.active_preset = None
        self._last_match_time = None
