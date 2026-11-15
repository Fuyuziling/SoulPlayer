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

from gesture_presets import (
    GesturePreset, PresetEngine,
    normalize_trajectory, resample_trajectory, dtw_distance,
    POSE_SEGMENTS,  # 五部分判断法的部位分组（与识别端共用，保证录制/识别一致）
)


class CustomPreset(GesturePreset):
    """
    自定义预设，继承 GesturePreset，增加关联歌曲列表。
    """

    def __init__(self, name: str, description: str,
                 conditions: dict, action: str, priority: int,
                 tracks: List[str], favorite: bool = False, order: int = 9999,
                 start_pct: float = 0.0):
        super().__init__(name, description, conditions, action, priority)
        self.tracks = tracks  # 关联的歌曲文件路径列表
        self.favorite = favorite  # 收藏标记（收藏项置顶显示）
        self.order = order        # 手动排序序号（越小越靠前）
        self.start_pct = start_pct  # 播放起始段落百分比 0.0~1.0（0.0 从头播放）
        # 固定标记为自定义预设：PresetEngine.match 据此在匹配期间屏蔽内置动作。
        # 必须在构造器里设置——否则本次会话新录制的预设经 preset_engine.add_preset
        # 直接加入时缺标记，屏蔽机制不生效（"做自定义动作时误触切歌/上一首"根因）。
        self._is_custom = True

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["tracks"] = self.tracks
        d["favorite"] = self.favorite
        d["order"] = self.order
        d["start_pct"] = self.start_pct
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
            start_pct=float(data.get("start_pct", 0.0) or 0.0),
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
        self.samples: List[Dict[str, float]] = []          # 动作期样本
        self.baseline_samples: List[Dict[str, float]] = []  # 中性基线样本（倒计时/准备期）
        self.trajectories: List[list] = []   # 录制期收集的肘部轨迹快照（DTW 模板用）
        self.hand_visibility_samples: List[float] = []  # 手部关键点平均可见度（录制期）
        self.state = self.STATE_IDLE
        self.countdown_start: Optional[float] = None
        self.recording_start: Optional[float] = None

    def start(self) -> None:
        """开始倒计时（倒计时结束后自动进入录制）。"""
        self.samples = []
        self.baseline_samples = []
        self.trajectories = []
        self.hand_visibility_samples = []  # 重置手部可见度统计
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
        self.trajectories = []

    def feed(self, motion_features: dict) -> None:
        """每帧调用，传入当前动作特征。"""
        if self.state == self.STATE_IDLE:
            return

        if self.state == self.STATE_COUNTDOWN:
            # 倒计时期间采样"中性基线"（用户准备/静止状态），
            # 距离匹配用它判定"是否偏离静止"（静止时特征≈基线 → 不触发）
            sample = self._extract_sample(motion_features)
            if sample:
                self.baseline_samples.append(sample)
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

            sample = self._extract_sample(motion_features)
            if sample:
                self.samples.append(sample)
            # 收集肘部轨迹快照（供 DTW 轨迹模板匹配）
            traj = motion_features.get("arm_trajectory")
            if traj and len(traj) >= 2:
                self.trajectories.append(
                    [(float(p[0]), float(p[1])) for p in traj])
            # 统计手部关键点可见度（用于录制质量评估）
            # 手部关键点：腕部(15/16)、食指根(17/18)、小指根(19/20)、拇指(21/22)
            hand_vis_keys = [
                'pose_hand_dx_L', 'pose_hand_dy_L',
                'pose_hand_dx_R', 'pose_hand_dy_R',
                'pose_spread_L', 'pose_spread_R'
            ]
            hand_vis_values = [
                motion_features.get(k)
                for k in hand_vis_keys
                if k in motion_features and isinstance(motion_features[k], (int, float))
            ]
            # 可见度：有手部特征帧数 / 总帧数（归一化到 0~1）
            if hand_vis_values:
                self.hand_visibility_samples.append(1.0)

    def _extract_sample(self, motion_features: dict) -> dict:
        """从一帧特征中提取录制用样本（数值特征 + 离散枚举 + 构型特征）。"""
        sample = {}
        # 含 palm_hover / wave_direction：它们是内置动作（举手暂停/挥手切歌）
        # 的判别特征，录制时采样它们才能在生成条件时与通用动作区分开
        for key in ["motion_energy", "arm_motion_energy", "body_speed",
                    "arm_amplitude", "body_rhythm", "arm_horizontal_motion",
                    "palm_hover", "wave_direction"]:
            if key in motion_features and isinstance(motion_features[key], (int, float)):
                sample[key] = float(motion_features[key])
        # 静态构型特征（五段：头部/胸腔/大臂/小臂/手部）：与运动特征互补——
        # 动态动作依赖运动量判别，静态动作依赖构型判别。
        # 构型特征仅在关节可见时存在（None 被 isinstance 过滤自动跳过）。
        for key in ["pose_head_dx", "pose_head_dy",
                    "pose_torso_dx", "pose_torso_dy",
                    "pose_upper_dx_L", "pose_upper_dy_L",
                    "pose_upper_dx_R", "pose_upper_dy_R",
                    "pose_fore_dx_L", "pose_fore_dy_L",
                    "pose_fore_dx_R", "pose_fore_dy_R",
                    "pose_hand_dx_L", "pose_hand_dy_L",
                    "pose_hand_dx_R", "pose_hand_dy_R",
                    "pose_spread_L", "pose_spread_R"]:
            if key in motion_features and isinstance(motion_features[key], (int, float)):
                sample[key] = float(motion_features[key])
        for key in ["posture", "hand_position", "hand_gesture_L", "hand_gesture_R"]:
            if key in motion_features:
                sample[key] = motion_features[key]
        if "is_active" in motion_features:
            sample["is_active"] = bool(motion_features["is_active"])
        return sample

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
        根据采样数据生成条件字典（方案A：中性基线 + 动作模板）。

        结构：
        {
          "duration": 0.3,
          "baseline": {特征: 均值},      # 倒计时/准备期的静止基线
          "action": {特征: 均值},        # 录制期的动作模板
          "action_std": {特征: 标准差},   # 动作期标准差（距离归一化用）
          "posture"/"hand_position": [值] # 稳定的离散区分（可选）
        }

        触发时采用距离匹配：当前特征距动作模板近（<2σ），
        且明显偏离中性基线（距基线 > 距动作×1.5）才触发——
        静止时特征≈基线，两条都判不满足，天然免疫"静止即触发"。
        duration=0.3s：连续满足 0.3 秒即触发（举手暂停=0.35 同级），
        瞬时动作（挥/拍约 0.3~0.5s）可达到；旧值 1.0s 需动作保持
        整整 1 秒不中断，任何一帧抖动即清零 → "重录后永远无法触发"。
        """
        if not self.samples:
            return {"duration": 0.3}

        numeric_keys = ["motion_energy", "arm_motion_energy", "body_speed",
                        "arm_amplitude", "body_rhythm", "arm_horizontal_motion",
                        "palm_hover", "wave_direction"]

        def _mean(values):
            return sum(values) / len(values) if values else 0.0

        def _std(values):
            if len(values) < 2:
                return 0.0
            m = _mean(values)
            return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5

        baseline = {}
        action = {}
        action_std = {}
        for key in numeric_keys:
            bvals = [s[key] for s in self.baseline_samples if key in s]
            avals = [s[key] for s in self.samples if key in s]
            if bvals:
                baseline[key] = round(_mean(bvals), 3)
            if avals:
                action[key] = round(_mean(avals), 3)
                action_std[key] = round(_std(avals), 3)

        conditions = {"duration": 0.3}
        if baseline:
            conditions["baseline"] = baseline
        if action:
            conditions["action"] = action
        if action_std:
            conditions["action_std"] = action_std

        # ── 静态构型模板（pose）+ 动态度 D：静态/动态判定有机结合 ──
        # 构型特征（肘角/腕偏移/spread）只在静止/慢动时稳定可靠，
        # 运动特征只在动作时才有信息——两者互补而非冗余。
        # 动态度 D（0~1）由录制期的轨迹幅度 + 运动能量方差共同决定：
        #   轨迹幅度大 / 能量波动大 → D 高（动态动作）→ 偏重运动比对；
        #   轨迹幅度小 / 能量平稳     → D 低（静态动作）→ 偏重构型比对。
        # duration 随 D 自适应：静态动作要求保持更久（0.3→1.0s 线性），
        # 天然过滤"动态手势中某帧恰好经过该构型"的假匹配——
        # 挥手只有一瞬间构型吻合，hold_timer 攒不够 1s。
        # ── 静态构型模板（pose）：只纳入"明显参与动作"的部位 ──
        # 五段分组：头部/胸腔/大臂/小臂/手部。录制时解析动作样本，
        # 某部位在录制期内变化幅度达到阈值（明显在动）才纳入该动作的
        # 判别标准；未明显参与的部位（如只动手臂时头部/胸腔几乎不动）
        # 不纳入 → 触发时不约束该部位，避免"没动的地方也要对上"导致
        # 自定义动作难以触发/误触发。触发端 _match_template 对已纳入
        # 的部位全部比对达成才播放（见 gesture_presets.py）。
        # 阈值 0.02 = 2% 屏宽：参考 MediaPipe 静止抖动约 0.001、生理性
        # 震颤极小；动作时部位位移通常 ≥ 2~3% 屏宽，0.02 可区分"动了"
        # 与"静止未参与"。
        # 五段分组 POSE_SEGMENTS 由 gesture_presets 导入（识别端共用）：
        # 头部/胸腔/大臂/小臂/手部。某部位变化幅度达到 SEG_MIN_SPAN
        # 才纳入该动作的判别标准。
        SEG_MIN_SPAN = 0.02   # 部位变化幅度阈值：明显参与动作才算
        seg_spans = {}
        for seg_name, seg_keys in POSE_SEGMENTS:
            span = 0.0
            for key in seg_keys:
                avals = [s[key] for s in self.samples if key in s]
                if avals:
                    span = max(span, max(avals) - min(avals))
            seg_spans[seg_name] = span

        pose = {}
        pose_std = {}
        active_segs = []
        for seg_name, seg_keys in POSE_SEGMENTS:
            if seg_spans[seg_name] >= SEG_MIN_SPAN:
                active_segs.append(seg_name)
                for key in seg_keys:
                    avals = [s[key] for s in self.samples if key in s]
                    if avals:
                        pose[key] = round(_mean(avals), 4)
                        pose_std[key] = round(max(_std(avals), 0.05), 4)

        # 兜底：所有部位都未明显参与（纯静止录制，如悬停姿势）→ 全部纳入，
        # 让"保持该姿势"成为判据（静态动作需要整身对上）。
        # 注意：兜底时 active_segs 保持为空列表——用于区分"动作类型"
        #（有明显参与部位，触发时须真的在动）与"纯静态姿势"（静止保持即可）。
        if not active_segs:
            for seg_name, seg_keys in POSE_SEGMENTS:
                for key in seg_keys:
                    avals = [s[key] for s in self.samples if key in s]
                    if avals:
                        pose[key] = round(_mean(avals), 4)
                        pose_std[key] = round(max(_std(avals), 0.05), 4)

        # 动态度：轨迹跨度（归一化到 0.15）与**手臂运动能量**方差
        #（归一化到 0.1）各占一半权重，clamp 到 [0,1]。
        # 用 arm_motion_energy 而非 motion_energy：后者是躯干+腿部能量
        #（motion_capture 中显式排除手臂 0-16），拍手/比划等手臂动作时
        # 躯干不动 → motion_energy≈0 → 方差≈0 → 动态度被低估为"静态"，
        # 判定偏向 pose 构型（均值模板对运动过程不匹配）→ 动作无法触发。
        # 参考数据：动作识别以"参与动作的部位"的能量为幅度因子。
        traj_span = 0.0
        if self.trajectories:
            traj_span = max(
                (max(p[0] for p in t) - min(p[0] for p in t))
                + (max(p[1] for p in t) - min(p[1] for p in t))
                for t in self.trajectories
            )
        energy_vals = [max(s.get("motion_energy", 0.0),
                           s.get("arm_motion_energy", 0.0))
                       for s in self.samples]
        energy_std = _std(energy_vals) if energy_vals else 0.0
        dynamism = max(0.0, min(1.0,
            (traj_span / 0.15 + energy_std / 0.1) / 2.0))

        if pose:
            conditions["pose"] = pose
            conditions["pose_std"] = pose_std
        # 记录"明显参与动作的部位"（非空 = 动作类型；空 = 纯静态姿势）。
        # 触发端据此要求：动作类型必须当前偏离静止基线（真的在动）才触发，
        # 防止静止时手停在录制姿态附近被 pose 构型判定误命中（"静止误触"根因）。
        conditions["active_segments"] = list(active_segs)
        conditions["dynamism"] = round(dynamism, 3)
        # duration = 0.15 + 0.85×(1-D)：静态 1.0s / 动态 0.15s 线性过渡。
        # 动态下限 0.15s：拍手/比划等瞬时动作约 0.1~0.3s（参考：手势识别
        # 最短 3 帧≈0.1s、正常识别延迟 0.21±0.05s），旧下限 0.3s 需动作
        # 连续 9 帧命中，瞬时动作任一帧抖动即清零 → 录制动作无法触发。
        conditions["duration"] = round(0.15 + 0.85 * (1.0 - dynamism), 2)

        # DTW 轨迹模板：取录制期间肘部跨度最大的快照（代表动作主体形态），
        # 下采样到 20 点并归一化，供触发端 _match_template 用 DTW 匹配。
        # 跨度按 x+y 综合计算——上下摇摆等纵向动作肘部横向跨度小，
        # 若只按 x 会选不出动作主体（自定义动作无法触发的根因之一）。
        # 轨迹形状（而非数值区间）是更本质的判别：同一种动作快慢/幅度不同
        # 都能匹配，手指动作的轨迹形态与录制动作完全不同 → DTW 距离大。
        if self.trajectories:
            best_traj = max(
                self.trajectories,
                key=lambda t: ((max(p[0] for p in t) - min(p[0] for p in t))
                               + (max(p[1] for p in t) - min(p[1] for p in t))),
            )
            conditions["trajectory"] = normalize_trajectory(
                resample_trajectory(best_traj, 20)).tolist()

        # 离散枚举只保留"可分类且稳定"的值（动作特征之外，辅助区分）
        # hand_gesture_L/R：精细手型（open/fist/victory/ok/thumbs_up/pointing）；
        # 过滤 None/unknown/half_fist——半握是兜底分类，用于判别不可靠
        for key in ["posture", "hand_position", "hand_gesture_L", "hand_gesture_R"]:
            values = [s[key] for s in self.samples
                      if key in s and s[key] not in (None, "unknown", "half_fist")]
            if values:
                from collections import Counter
                value, count = Counter(values).most_common(1)[0]
                if count / len(values) >= 0.6:
                    conditions[key] = [value]

        # ── 手部可见度检查（录制质量评估）──
        # 如果手部关键点在录制期可见度过低，会导致 pose_spread 缺失 →
        # 手型约束失效 → 容易与相似动作串播。
        # 这里统计可见度并给出警告，让用户知道录制质量不佳。
        if self.hand_visibility_samples:
            hand_vis_ratio = len(self.hand_visibility_samples) / max(len(self.samples), 1)
            if hand_vis_ratio < 0.5 and "手部" in active_segs:
                import sys
                print(
                    f"[WARN] 录制警告：手部关键点可见度仅 {hand_vis_ratio:.1%} "
                    f"({len(self.hand_visibility_samples)}/{len(self.samples)} 帧)，"
                    f"pose_spread 可能缺失 → 手型约束失效 → 易与相似动作串播",
                    file=sys.stderr
                )
                print(
                    f"[WARN] 建议：重新录制，确保手掌/手指清晰可见（正对镜头，光线充足）",
                    file=sys.stderr
                )
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
                except (OSError, json.JSONDecodeError, ValueError) as e:
                    print(f"[CustomPresetManager] 迁移预设失败: {e}")

            if migrated > 0:
                print(f"[CustomPresetManager] 已从旧版格式迁移 {migrated} 个预设到 {self.storage_dir}")

            # 迁移完成后重命名旧文件（不直接删除，保留备份）
            backup = legacy.with_suffix(".json.bak")
            legacy.rename(backup)
            print(f"[CustomPresetManager] 旧版文件已备份为 {backup.name}")

        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"[CustomPresetManager] 迁移旧版数据失败: {e}")

    @staticmethod
    def _sanitize_conditions(preset) -> bool:
        """清洗预设条件中的历史噪声。返回是否有改动。

        - posture/hand_position 为 ["unknown"]：识别失败的产物，删除
        - is_active 条件：与数值特征存在推导关系易自相矛盾，删除
        - duration > 1.0：降到 1.0（旧版 1.0+ 需动作保持过久；新录制动态
          动作已是 0.3、静态动作 1.0，均在合理范围，仅拦截历史超长值）
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

    @staticmethod
    def is_idle_matching(conditions: dict, strict: bool = True) -> bool:
        """
        检测"静止即触发"的退化预设 / 录制内容是否近似静止。

        录制时若几乎没动，generate_conditions 会生成 motion_energy 上限极小的
        条件（如 0~0.05），导致"站着不动"也满足 → 用户静止时每 1s 冷却后
        反复触发 custom_play（播放系统陷入频繁触发），且屏蔽内置动作。
        加载与录制结束时都应拒绝此类条件。

        注意：motion_energy 只统计躯干+腿部（不含手臂）——拍桌/指人等
        手臂动作躯干几乎不动，若只看 motion_energy 会把正常录制误判为静止。
        因此：
        - strict=False（录制时）：综合所有运动类指标，任一显著即有动作；
        - strict=True（加载时）：以 motion_energy 上限为主判据，
          拦截历史退化预设（如 motion_energy 上限极小但手臂指标为全范围）。

        阈值 0.12 需容纳 margin 扩展：generate_conditions 的 max 会
        +0.05~0.08 余量，静止录制（真实 max≈0.03）扩展后约 0.084，
        正常动作（真实 max≥0.15）扩展后 ≥0.2，0.12 居中安全。
        """
        if not isinstance(conditions, dict):
            return False
        numeric = ("motion_energy", "arm_motion_energy", "body_speed",
                   "arm_horizontal_motion")
        numeric_cond = {k: v for k, v in conditions.items()
                        if k in numeric and isinstance(v, dict)}
        if not numeric_cond:
            return False
        if strict:
            # 加载（strict=True）：motion_energy 上限极小 → 静止状态也满足，
            # 拦截历史退化预设（如 motion_energy 0~0.05 但手臂指标为全范围）
            me = conditions.get("motion_energy")
            if isinstance(me, dict) and me.get("max", 1.0) <= 0.12:
                return True
        # 主判据（录制 strict=False）：任一运动类指标上限显著 → 有动作，放行。
        # motion_energy 只统计躯干腿部，拍桌/指人等手臂动作须看 arm_motion_energy
        if any(v.get("max", 1.0) > 0.12 for v in numeric_cond.values()):
            return False
        # 全部运动类指标上限都很小 → 近似静止
        if all(v.get("max", 1.0) <= 0.12 for v in numeric_cond.values()):
            # 有离散区分（posture/hand_position）时不判定为静止
            for k in ("posture", "hand_position"):
                v = conditions.get(k)
                if isinstance(v, list) and v and all(x != "unknown" for x in v):
                    return False
            return True
        return False

    @staticmethod
    def count_discriminators(conditions: dict) -> int:
        """
        统计自定义动作条件的"有效判定点"数量。

        判定点 = 能排除"静止状态"（所有特征≈0）的约束：
        - 运动类数值特征 min > 0.08：要求该特征必须达到一定水平才匹配，
          静止（≈0）不满足 → 算 1 个判定点；
        - 离散特征（posture / hand_position）有明确值（非 unknown）→ 算 1 个。

        只有 1 个判定点的动作区分度不足：静止时也可能满足 → 反复误触卡死。
        录制校验要求至少 2 个判定点（用户建议），保证动作"真的在做动作"。
        """
        if not isinstance(conditions, dict):
            return 0
        count = 0
        numeric = ("motion_energy", "arm_motion_energy", "body_speed",
                   "arm_amplitude", "body_rhythm", "arm_horizontal_motion")
        for k in numeric:
            v = conditions.get(k)
            if isinstance(v, dict) and v.get("min", 0.0) > 0.08:
                count += 1
        for k in ("posture", "hand_position"):
            v = conditions.get(k)
            if isinstance(v, list) and v and all(x != "unknown" for x in v):
                count += 1
        return count

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
                    # 拒绝"静止即触发"的退化预设：避免用户静止时频繁误触发
                    if self.is_idle_matching(preset.conditions):
                        print(f"[CustomPresetManager] 跳过退化预设（静止即触发，避免频繁误触发）: {preset.name}")
                        continue
                    self.presets.append(preset)
                elif isinstance(data, dict) and "custom_presets" in data:
                    # 旧版集合格式（理论上已被迁移，但防御性兼容）
                    for preset_data in data["custom_presets"]:
                        preset = CustomPreset.from_dict(preset_data)
                        if self.is_idle_matching(preset.conditions):
                            print(f"[CustomPresetManager] 跳过退化预设（静止即触发）: {preset.name}")
                            continue
                        self.presets.append(preset)
                else:
                    print(f"[CustomPresetManager] 无法识别的格式: {json_file.name}")

            except json.JSONDecodeError as e:
                print(f"[CustomPresetManager] JSON 解析失败（已跳过）: {json_file.name} — {e}")
            except (OSError, ValueError, TypeError) as e:
                print(f"[CustomPresetManager] 加载失败（已跳过）: {json_file.name} — {e}")
            except Exception as e:  # noqa: BLE001
                # 【P0-⑥】兜底：单个预设文件损坏不应导致整个应用启动崩溃。
                # GesturePreset.from_dict 会对缺失字段主动 raise KeyError，
                # 手写/编辑过的预设、跨版本遗留文件都可能命中——这里一律跳过并告警。
                print(f"[CustomPresetManager] 预设文件异常（已跳过）: "
                      f"{json_file.name} — {type(e).__name__}: {e}")

        # 按持久化的 order 恢复手动排序
        self.presets.sort(key=lambda p: getattr(p, 'order', 9999))
        # 清洗历史预设中的噪声条件（unknown 枚举是识别失败的产物，
        # 写入条件会导致永远匹配不上；duration 过长易被打断）
        for p in self.presets:
            if self._sanitize_conditions(p):
                try:
                    self._save_single_preset(p)
                    print(f"[CustomPresetManager] 已清洗预设条件: {p.name}")
                except (OSError, TypeError, ValueError) as e:
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

    def add_preset(self, name: str, conditions: dict, tracks: List[str],
                   start_pct: float = 0.0) -> CustomPreset:
        """
        添加新的自定义预设。

        Args:
            name: 预设名称
            conditions: 触发条件
            tracks: 关联歌曲列表
            start_pct: 播放起始段落百分比 0.0~1.0（0.0 从头播放）

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
            tracks=tracks,
            start_pct=max(0.0, min(1.0, float(start_pct or 0.0))),
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
