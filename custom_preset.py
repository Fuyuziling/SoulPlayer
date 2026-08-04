"""
custom_preset.py - 自定义预设管理模块（v2.0）

功能：
- 自定义预设的录制（采样动作特征范围）
- 自定义预设的存储/加载（每个预设独立 JSON 文件，存放在 custom_presets/ 目录）
- 歌曲关联（文件路径列表）
- 与现有 PresetEngine 集成
- 兼容旧版 single-file 格式（自动迁移）

对外暴露：
- CustomPreset: 自定义预设数据类
- CustomPresetManager: 管理器（录制、保存、加载、播放）
"""

import json
import os
import time
import random
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

from gesture_presets import GesturePreset, PresetEngine


class CustomPreset(GesturePreset):
    """
    自定义预设，继承 GesturePreset，增加关联歌曲列表。
    """

    def __init__(self, name: str, description: str,
                 conditions: dict, action: str, priority: int,
                 tracks: List[str], favorite: bool = False, order: int = 9999):
        super().__init__(name, description, conditions, action, priority)
        self.tracks = tracks  # 关联的歌曲文件路径列表
        self.favorite = favorite  # 收藏标记（收藏项置顶显示）
        self.order = order        # 手动排序序号（越小越靠前）

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tracks"] = self.tracks
        d["favorite"] = self.favorite
        d["order"] = self.order
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CustomPreset":
        # 显式调用 GesturePreset 的类方法，避免 super() 在类方法中的参数传递歧义
        preset = GesturePreset.from_dict(data)
        # 重新包装为 CustomPreset
        return cls(
            name=preset.name,
            description=preset.description,
            conditions=preset.conditions,
            action=preset.action,
            priority=preset.priority,
            tracks=data.get("tracks", []),
            favorite=bool(data.get("favorite", False)),
            order=int(data.get("order", 9999)),
        )

    def pick_random_track(self) -> Optional[str]:
        """从关联歌曲中随机选择一首。"""
        valid_tracks = [t for t in self.tracks if os.path.exists(t)]
        if not valid_tracks:
            return None
        return random.choice(valid_tracks)


class PresetRecorder:
    """
    动作录制器：支持倒计时后录制，可手动结束。

    状态机：
        idle -> countdown(3秒) -> recording -> finished
    """

    STATE_IDLE = "idle"
    STATE_COUNTDOWN = "countdown"
    STATE_RECORDING = "recording"
    STATE_FINISHED = "finished"

    COUNTDOWN_SEC = 3.0

    def __init__(self, duration_sec: float = 3.0):
        self.duration_sec = duration_sec
        self.samples: List[Dict[str, float]] = []
        self.state = self.STATE_IDLE
        self.countdown_start: Optional[float] = None
        self.recording_start: Optional[float] = None

    def start(self) -> None:
        """开始倒计时（倒计时结束后自动进入录制）。"""
        self.samples = []
        self.state = self.STATE_COUNTDOWN
        self.countdown_start = time.time()
        self.recording_start = None

    def stop(self) -> None:
        """手动停止录制（仅在 recording 状态有效）。"""
        if self.state == self.STATE_RECORDING:
            self.state = self.STATE_FINISHED

    def cancel(self) -> None:
        """取消当前录制流程（倒计时或录制中均可取消）。"""
        self.state = self.STATE_IDLE
        self.countdown_start = None
        self.recording_start = None
        self.samples = []

    def feed(self, motion_features: dict) -> None:
        """每帧调用，传入当前动作特征。"""
        if self.state == self.STATE_IDLE:
            return

        if self.state == self.STATE_COUNTDOWN:
            elapsed = time.time() - self.countdown_start
            if elapsed >= self.COUNTDOWN_SEC:
                # 倒计时结束，开始录制
                self.state = self.STATE_RECORDING
                self.recording_start = time.time()

        elif self.state == self.STATE_RECORDING:
            # 检查是否超时（自动结束）
            elapsed = time.time() - self.recording_start
            if elapsed > self.duration_sec:
                self.state = self.STATE_FINISHED
                return

            # 提取数值特征
            sample = {}
            for key in ["motion_energy", "arm_motion_energy", "body_speed", "arm_amplitude", "body_rhythm", "arm_horizontal_motion"]:
                if key in motion_features and isinstance(motion_features[key], (int, float)):
                    sample[key] = float(motion_features[key])
            for key in ["posture", "hand_position"]:
                if key in motion_features:
                    sample[key] = motion_features[key]
            if "is_active" in motion_features:
                sample["is_active"] = bool(motion_features["is_active"])

            self.samples.append(sample)

    def is_countdown(self) -> bool:
        return self.state == self.STATE_COUNTDOWN

    def is_recording(self) -> bool:
        return self.state == self.STATE_RECORDING

    def is_finished(self) -> bool:
        return self.state == self.STATE_FINISHED

    def get_countdown_progress(self) -> float:
        """返回倒计时进度 0.0~1.0。"""
        if self.state != self.STATE_COUNTDOWN or self.countdown_start is None:
            return 0.0
        elapsed = time.time() - self.countdown_start
        return min(elapsed / self.COUNTDOWN_SEC, 1.0)

    def get_countdown_remaining(self) -> float:
        """返回倒计时剩余秒数。"""
        if self.state != self.STATE_COUNTDOWN or self.countdown_start is None:
            return 0.0
        elapsed = time.time() - self.countdown_start
        return max(self.COUNTDOWN_SEC - elapsed, 0.0)

    def get_recording_progress(self) -> float:
        """返回录制进度 0.0~1.0。"""
        if self.state != self.STATE_RECORDING or self.recording_start is None:
            return 0.0
        elapsed = time.time() - self.recording_start
        return min(elapsed / self.duration_sec, 1.0)

    def generate_conditions(self) -> dict:
        """
        根据采样数据生成条件字典。
        策略：数值特征取 min/max 扩展 20% 余量，离散取最频繁值。
        """
        if not self.samples:
            return {"duration": 1.0}

        conditions = {"duration": 1.0}

        numeric_keys = ["motion_energy", "arm_motion_energy", "body_speed", "arm_amplitude", "body_rhythm", "arm_horizontal_motion"]
        for key in numeric_keys:
            values = [s[key] for s in self.samples if key in s]
            if values:
                min_val = min(values)
                max_val = max(values)
                margin = (max_val - min_val) * 0.2 + 0.05
                min_val = max(0.0, min_val - margin)
                max_val = min(1.0, max_val + margin)
                conditions[key] = {"min": round(min_val, 3), "max": round(max_val, 3)}

        # 枚举特征只保留"可分类且稳定"的值：
        # "unknown" 是识别失败的噪声而非动作特征，若写入条件会导致
        # 正常坐下/站立时永远匹配不上（自定义动作不触发的根因）
        for key in ["posture", "hand_position"]:
            values = [s[key] for s in self.samples if key in s]
            if values:
                from collections import Counter
                value, count = Counter(values).most_common(1)[0]
                if value != "unknown" and count / len(values) >= 0.6:
                    conditions[key] = [value]

        # 不写 is_active 条件：它由 motion_energy/body_speed 推导，
        # 与数值条件存在推导关系，一起写入容易自相矛盾
        # （例如 motion_energy≤0.05 与 is_active=true 几乎不可能同时成立，
        #  这是历史预设"录了却永远不触发"的根因之一）
        return conditions

    def get_recording_summary(self) -> str:
        """生成录制摘要描述。"""
        if not self.samples:
            return "无采样数据"
        parts = []
        for key in ["motion_energy", "arm_motion_energy", "body_speed", "arm_amplitude"]:
            values = [s[key] for s in self.samples if key in s]
            if values:
                parts.append(f"{key}={min(values):.2f}~{max(values):.2f}")
        return " | ".join(parts) if parts else "未知"


class CustomPresetManager:
    """
    自定义预设管理器（v2.0）：统一封装录制、保存、加载、播放。

    存储策略：
    - 每个预设保存为 custom_presets/ 目录下的独立 .json 文件
    - 预设文件名使用 sanitized 后的预设名称
    - 兼容旧版 single-file 格式（自动迁移）
    - 单个预设文件损坏不影响其他预设
    """

    def __init__(self, storage_dir: Optional[str] = None):
        """
        初始化管理器。

        Args:
            storage_dir: 预设存储目录，默认使用 <脚本目录>/custom_presets/
        """
        if storage_dir is None:
            base = Path(__file__).parent
            storage_dir = str(base / "custom_presets")
        self.storage_dir = Path(storage_dir)
        # 确保目录存在
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.presets: List[CustomPreset] = []
        self.recorder = PresetRecorder(duration_sec=3.0)

        # 兼容旧版：如果存在旧版 single-file，自动迁移
        self._migrate_from_legacy()
        self.load()

    def _sanitize_filename(self, name: str) -> str:
        """将预设名称转换为合法的文件名。"""
        # 保留字母、数字、中文、下划线、连字符
        sanitized = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', name)
        return sanitized[:100]  # 限制长度

    def _legacy_file_path(self) -> Path:
        """旧版 single-file 的存储路径。"""
        return Path(__file__).parent / "custom_presets.json"

    def _migrate_from_legacy(self):
        """
        从旧版 single-file 格式迁移到新的目录格式。
        迁移完成后删除旧文件（或重命名为 .bak）。
        """
        legacy = self._legacy_file_path()
        if not legacy.exists():
            return

        try:
            with open(legacy, "r", encoding="utf-8") as f:
                data = json.load(f)

            presets_data = data.get("custom_presets", [])
            migrated = 0
            for preset_data in presets_data:
                try:
                    preset = CustomPreset.from_dict(preset_data)
                    # 检查是否已存在（避免重复迁移）
                    if not any(p.name == preset.name for p in self.presets):
                        self.presets.append(preset)
                        self._save_single_preset(preset)
                        migrated += 1
                except Exception as e:
                    print(f"[CustomPresetManager] 迁移预设失败: {e}")

            if migrated > 0:
                print(f"[CustomPresetManager] 已从旧版格式迁移 {migrated} 个预设到 {self.storage_dir}")

            # 迁移完成后重命名旧文件（不直接删除，保留备份）
            backup = legacy.with_suffix(".json.bak")
            legacy.rename(backup)
            print(f"[CustomPresetManager] 旧版文件已备份为 {backup.name}")

        except Exception as e:
            print(f"[CustomPresetManager] 迁移旧版数据失败: {e}")

    @staticmethod
    def _sanitize_conditions(preset) -> bool:
        """清洗预设条件中的历史噪声。返回是否有改动。

        - posture/hand_position 为 ["unknown"]：识别失败的产物，删除
        - is_active 条件：与数值特征存在推导关系易自相矛盾，删除
        - duration > 1.0：降到 1.0（过长容易被动作中断永远达不到）
        """
        changed = False
        cond = preset.conditions
        for key in ("posture", "hand_position"):
            if key in cond:
                vals = cond[key]
                if isinstance(vals, list) and all(v == "unknown" for v in vals):
                    del cond[key]
                    changed = True
                elif vals == "unknown":
                    del cond[key]
                    changed = True
        if "is_active" in cond:
            del cond["is_active"]
            changed = True
        if cond.get("duration", 0) > 1.0:
            cond["duration"] = 1.0
            changed = True
        return changed

    def load(self):
        """
        从 storage_dir 加载所有自定义预设。
        每个 .json 文件对应一个预设，单个文件损坏不影响其他。
        """
        self.presets = []
        if not self.storage_dir.exists():
            return

        for json_file in sorted(self.storage_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 兼容格式检测
                if isinstance(data, dict) and "name" in data:
                    # 单预设格式（v2.0）
                    preset = CustomPreset.from_dict(data)
                    self.presets.append(preset)
                elif isinstance(data, dict) and "custom_presets" in data:
                    # 旧版集合格式（理论上已被迁移，但防御性兼容）
                    for preset_data in data["custom_presets"]:
                        preset = CustomPreset.from_dict(preset_data)
                        self.presets.append(preset)
                else:
                    print(f"[CustomPresetManager] 无法识别的格式: {json_file.name}")

            except json.JSONDecodeError as e:
                print(f"[CustomPresetManager] JSON 解析失败（已跳过）: {json_file.name} — {e}")
            except Exception as e:
                print(f"[CustomPresetManager] 加载失败（已跳过）: {json_file.name} — {e}")

        # 按持久化的 order 恢复手动排序
        self.presets.sort(key=lambda p: getattr(p, 'order', 9999))
        # 清洗历史预设中的噪声条件（unknown 枚举是识别失败的产物，
        # 写入条件会导致永远匹配不上；duration 过长易被打断）
        for p in self.presets:
            if self._sanitize_conditions(p):
                try:
                    self._save_single_preset(p)
                    print(f"[CustomPresetManager] 已清洗预设条件: {p.name}")
                except Exception as e:
                    print(f"[CustomPresetManager] 清洗后保存失败: {p.name} — {e}")
        # 统一提升优先级（迁移旧版 priority=5 的预设）：
        # 自定义动作不存在先后关系，匹配到就触发，不被内置动作抢占
        for p in self.presets:
            p.priority = 20

    def _save_single_preset(self, preset: CustomPreset):
        """将单个预设保存到独立文件。"""
        filename = self._sanitize_filename(preset.name) + ".json"
        filepath = self.storage_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(preset.to_dict(), f, ensure_ascii=False, indent=2)

    def save(self):
        """保存所有预设（逐个写入独立文件）。"""
        # 不再使用 single-file 保存，改为逐个保存
        # 注意：如果删除了预设，对应的文件需要手动清理
        for preset in self.presets:
            self._save_single_preset(preset)

    def add_preset(self, name: str, conditions: dict, tracks: List[str]) -> CustomPreset:
        """
        添加新的自定义预设。

        Args:
            name: 预设名称
            conditions: 触发条件
            tracks: 关联歌曲列表

        Returns:
            创建的 CustomPreset 实例
        """
        # 确保名称唯一
        base_name = name
        counter = 1
        while any(p.name == name for p in self.presets):
            name = f"{base_name}_{counter}"
            counter += 1

        preset = CustomPreset(
            name=name,
            description=f"录制于 {time.strftime('%Y-%m-%d %H:%M')}",
            conditions=conditions,
            action="custom_play",
            priority=20,  # 高于内置动作（举手/挥手=10）：匹配到自定义动作就触发
            tracks=tracks
        )
        self.presets.append(preset)
        self._save_single_preset(preset)
        return preset

    def remove_preset(self, name: str):
        """删除指定名称的预设，并删除对应文件。"""
        preset = self.get_preset(name)
        if preset:
            # 删除对应文件
            filename = self._sanitize_filename(preset.name) + ".json"
            filepath = self.storage_dir / filename
            if filepath.exists():
                filepath.unlink()
        self.presets = [p for p in self.presets if p.name != name]

    def get_preset(self, name: str) -> Optional[CustomPreset]:
        """根据名称获取预设。"""
        for p in self.presets:
            if p.name == name:
                return p
        return None

    # ---------- 编辑接口（重命名/收藏/排序） ----------

    def get_display_presets(self) -> List[CustomPreset]:
        """返回显示顺序：收藏项在前，其余按手动排序（各自内部稳定）。"""
        favs = [p for p in self.presets if getattr(p, 'favorite', False)]
        rest = [p for p in self.presets if not getattr(p, 'favorite', False)]
        return favs + rest

    def rename_preset(self, old_name: str, new_name: str) -> bool:
        """
        重命名预设（同步重命名对应 JSON 文件）。

        Returns:
            True 成功；False 失败（新名为空、与现有预设重名或预设不存在）
        """
        new_name = (new_name or "").strip()
        if not new_name:
            return False
        if any(p.name == new_name for p in self.presets):
            return False
        preset = self.get_preset(old_name)
        if not preset:
            return False

        old_file = self.storage_dir / (self._sanitize_filename(old_name) + ".json")
        preset.name = new_name
        if old_file.exists():
            old_file.unlink()
        self._save_single_preset(preset)
        return True

    def toggle_favorite(self, name: str) -> Optional[bool]:
        """
        切换收藏状态并持久化。

        Returns:
            切换后的收藏状态；预设不存在返回 None
        """
        preset = self.get_preset(name)
        if not preset:
            return None
        preset.favorite = not getattr(preset, 'favorite', False)
        self._save_single_preset(preset)
        return preset.favorite

    def move_preset(self, name: str, direction: int) -> bool:
        """
        在显示顺序中移动预设（direction: -1 前移, +1 后移），并持久化顺序。

        Returns:
            True 移动成功；False 已到边界或预设不存在
        """
        display = self.get_display_presets()
        idx = next((i for i, p in enumerate(display) if p.name == name), None)
        if idx is None:
            return False
        j = idx + direction
        if j < 0 or j >= len(display):
            return False

        # 交换两者在主列表中的位置
        i1 = self.presets.index(display[idx])
        i2 = self.presets.index(display[j])
        self.presets[i1], self.presets[i2] = self.presets[i2], self.presets[i1]

        # 重写 order 并逐个持久化
        for k, p in enumerate(self.presets):
            p.order = k
            self._save_single_preset(p)
        return True

    def pick_track(self, preset_name: str) -> Optional[str]:
        """为指定预设随机选择一首歌曲。"""
        preset = self.get_preset(preset_name)
        if preset:
            return preset.pick_random_track()
        return None

    def inject_into_engine(self, engine: PresetEngine):
        """
        将所有自定义预设注入到 PresetEngine 中。

        注意：这会添加预设，若已存在同名预设则跳过（避免重复）。
        """
        for preset in self.presets:
            preset._is_custom = True  # 标记自定义预设，供引擎做内置动作屏蔽
            try:
                engine.add_preset(preset)
            except ValueError:
                # 已存在同名预设，跳过
                pass

    # ---------- 录制接口 ----------

    def start_recording(self):
        """开始录制流程（先倒计时，再自动进入录制）。"""
        self.recorder.start()

    def stop_recording(self):
        """手动停止录制（提前结束）。"""
        self.recorder.stop()

    def cancel_recording(self):
        """取消当前录制流程（倒计时或录制中均可）。"""
        self.recorder.cancel()

    def feed_recording(self, motion_features: dict):
        """每帧传入特征推进录制状态机。"""
        self.recorder.feed(motion_features)

    def is_countdown(self) -> bool:
        """是否正在倒计时。"""
        return self.recorder.is_countdown()

    def is_recording(self) -> bool:
        """是否正在录制（倒计时结束后）。"""
        return self.recorder.is_recording()

    def is_recording_finished(self) -> bool:
        """录制是否已完成（含手动停止和超时）。"""
        return self.recorder.is_finished()

    def get_countdown_progress(self) -> float:
        """倒计时进度 0.0~1.0。"""
        return self.recorder.get_countdown_progress()

    def get_countdown_remaining(self) -> float:
        """倒计时剩余秒数。"""
        return self.recorder.get_countdown_remaining()

    def get_recording_progress(self) -> float:
        """录制进度 0.0~1.0。"""
        return self.recorder.get_recording_progress()

    def finish_recording(self) -> dict:
        """结束录制，返回生成的条件字典。"""
        self.recorder.stop()
        return self.recorder.generate_conditions()

    def get_recording_summary(self) -> str:
        """获取录制摘要。"""
        return self.recorder.get_recording_summary()
