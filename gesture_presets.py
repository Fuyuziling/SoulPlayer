"""
gesture_presets.py - 动作预设系统

负责管理动作预设的定义、加载/保存，以及基于动作特征的实时匹配引擎。
是 SoulPlayer 的核心决策模块：将 motion_capture 输出的特征向量映射为具体的播放控制动作。
"""

import json
import os
import time
from typing import Dict, List, Optional, Any

import numpy as np

# DTW 轨迹匹配阈值（平均每步距离）：同形状轨迹 ≈0.05~0.08，
# 手指/无关动作轨迹 >0.2，0.15 居中——形态相同放行、形态不同排除
DTW_MATCH_THRESHOLD = 0.15
# DTW 轨迹最小原始幅度（归一化前，肘部相对肩部跨度）：
# 静止/手指微动时肘部相对肩部几乎不动（跨度常 <0.04），若归一化会把
# 微小漂移放大成"完整幅度动作"，形态可能贴近模板导致误触。
# 真实自定义动作肘部相对肩部跨度通常 ≥0.08，0.04 门槛安全分隔。
DTW_MIN_AMPLITUDE = 0.04


# ---------------------------------------------------------------------------
#  自定义动作判定参数（独立于通用动作，单独调参）
#  依据（人体运动学调研）：
#  - MediaPipe 静止 landmark 抖动 ≈ 0.001 归一化（One-Euro 滤波后）；
#  - 明显参与动作的部位位移通常 ≥ 2~3% 屏宽；
#  - 人体动作识别通用做法：按身体部位拆分为最小判断条件，
#    全部子条件同时满足才判定一次动作（多重最小拆分原则）。
# ---------------------------------------------------------------------------
# 五部分判断法：头部/胸腔/大臂/小臂/手部。录制端（custom_preset.py）与
# 识别端共用此定义，保证"录制捕获哪些部位"与"识别约束哪些部位"严格一致。
POSE_SEGMENTS = [
    ("头部", ("pose_head_dx", "pose_head_dy")),
    ("胸腔", ("pose_torso_dx", "pose_torso_dy")),
    ("大臂", ("pose_upper_dx_L", "pose_upper_dy_L",
              "pose_upper_dx_R", "pose_upper_dy_R")),
    ("小臂", ("pose_fore_dx_L", "pose_fore_dy_L",
              "pose_fore_dx_R", "pose_fore_dy_R")),
    ("手部", ("pose_hand_dx_L", "pose_hand_dy_L",
              "pose_hand_dx_R", "pose_hand_dy_R",
              "pose_spread_L", "pose_spread_R")),
]
# 单段 σ 上限：录制捕获的部位必须全部命中（五部分判断法核心）。
# 2.0 → 1.6：配合 std 下限 0.03，单段容差 ≈ 0.048（约 5% 屏宽）——
# 远高于 0.1% 静止抖动（防误触），又低于"相似但错误"姿势的偏移量（保专一）。
CUSTOM_POSE_SEG_SIGMA = 1.6
# 单键最大 σ 上限：段内任一键超限即拒绝——防止"段内多键平均稀释"：
# 手部等 6 键段中仅 1 键大幅偏离（3σ）时段均值仍 <1.6σ，
# 若不设单键门槛，"全部满足"会被平均值稀释成"大致满足"。
CUSTOM_POSE_KEY_SIGMA = 2.2
# 单段 std 下限：录制期近似恒定的部位 std≈0.01，若直接用会把容差收紧到
# 0.016，实机估计噪声即超限 → 永远不触发；下限保证对噪声有容差。
CUSTOM_POSE_STD_FLOOR = 0.03
# 运动均值距离 σ 上限：当前特征须距动作模板 < 2σ（整体运动量一致）
CUSTOM_MOTION_SIGMA = 2.0
# 偏离静止基线 σ 下限：动态动作必须明显偏离基线（真的在动），
# 静止时特征≈基线 → 天然不满足（防"静止即触发"）
CUSTOM_BASELINE_SIGMA = 1.0
# 运动特征 std 下限（动作模板距离归一化用）
CUSTOM_MOTION_STD_FLOOR = 0.05


# ---------------------------------------------------------------------------
#  轨迹模板工具（DTW 匹配用，纯 numpy 实现）
# ---------------------------------------------------------------------------
def normalize_trajectory(points) -> np.ndarray:
    """
    轨迹归一化：去均值（消除绝对位置差异）+ 按最大幅度缩放（消除幅度差异）。

    使"同一种动作在不同位置、不同幅度"下轨迹形状一致，
    DTW 只比较运动形态，天然区分挥手（平滑弧线）与手指动作（高频抖动）。

    注意：原始幅度过小时不缩放——手指动作时肘部几乎不动，轨迹仅含
    估计噪声（幅度可能 <0.02），若强行放大到 1 会把噪声当信号，
    与真实挥手的距离被错误压低。
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 2:
        return pts
    pts = pts - pts.mean(axis=0)
    scale = float(np.abs(pts).max())
    if scale > 0.02:          # 真实运动幅度足够时才归一化
        pts = pts / scale
    return pts


def resample_trajectory(points, n: int = 20) -> np.ndarray:
    """轨迹下采样到固定点数 n（等间隔取点），便于模板存储与距离比较。"""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= n:
        return pts
    idx = np.linspace(0, len(pts) - 1, n).astype(int)
    return pts[idx]


def dtw_distance(a, b) -> float:
    """
    动态时间规整（DTW）距离：比较两条轨迹的形状相似度。

    与欧氏距离不同，DTW 允许时间轴伸缩对齐——同一种动作做得快/慢
    都能对齐匹配；而手指动作的轨迹形态与挥手完全不同，DTW 距离大。

    :param a: (n, d) 轨迹数组
    :param b: (m, d) 轨迹数组
    :return: 规整路径的平均每步距离（0~1，越小越相似；长度无关）
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    # 逐点推进动态规划（n,m ≤ 20，纯 Python 循环足够快）
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = float(np.linalg.norm(ai - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # 除以路径步数 (n+m) 得平均每步距离：轨迹长短不影响可比性
    return float(cost[n, m]) / (n + m)


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


class CustomActionMatcher:
    """
    自定义动作独立判定器（与通用动作逻辑完全分离，单独调参）。

    五部分判断法（头部/胸腔/大臂/小臂/手部）：
    - 录制端（custom_preset.generate_conditions）按 POSE_SEGMENTS 统计各部位
      在录制窗口内的变化幅度，明显参与动作（≥2% 屏宽）的部位写入 pose/
      pose_std 并记录到 active_segments；未参与部位不写入 → 识别时不约束；
    - 识别端（本类 match）：凡录制时捕获的部位**全部**满足构型距离
      （< CUSTOM_POSE_SEG_SIGMAσ）才算姿势一致——任一部位偏离即拒绝，
      不再允许"整体平均距离小但某部位摆错"被稀释；
    - 动态动作（active_segments 非空）额外要求偏离静止基线（真的在动）；
    - DTW 轨迹仅作为运动确认的附加信号，不得绕过五段构型硬门槛。

    判定参数全部使用独立的 CUSTOM_* 常量，与通用动作的 _NUMERIC_KEYS 等
    互不干扰；PresetEngine 侧也使用独立的 _custom_hold_timers 计时器。
    """

    def match(self, conditions: dict, motion_features: dict) -> bool:
        """判定当前特征是否满足该自定义动作（静态/动态混合判定）。

        混合判定按该自定义动作**自身录制的动态度 D** 灵活切换：
        - 静态动作（D≈0，录制成"保持姿势"）：以五部分构型判定为主，
          录制捕获的部位必须全部命中（pose_ok）；
        - 动态动作（D≈1，录制成"运动过程"）：以运动判定为主（motion_ok），
          手部姿势（手指张开度）作为重点附加约束；
        - 中间态按 (1-D)·pose_ok + D·motion_ok ≥ 0.5 有机融合。
        """
        # 与通用动作的专属特征互斥（双保险，路由层已优先，此处再拒绝一次）：
        # 掌心向前悬停 = 举手暂停专属信号；挥手方向信号 = 切歌专属。
        if motion_features.get("palm_hover", 0.0) >= 0.5:
            return False
        if abs(motion_features.get("wave_direction", 0.0)) >= 0.5:
            return False

        # 动态度 D（录制时按该动作算出并存储；旧预设缺失默认 1.0，与旧版一致）
        _dyn = conditions.get("dynamism")
        dynamism = float(_dyn if _dyn is not None else 1.0)

        # 构型判定：录制捕获的部位全部命中（五部分判断法）
        pose_ok = self._pose_segments_match(conditions, motion_features)
        # 运动判定：距动作模板近且偏离静止基线（DTW 形态作为高置信附加）
        motion_ok = self._motion_matches(conditions, motion_features)

        # ── 静态/动态混合：按该动作自身动态度灵活切换 ──
        score = (1.0 - dynamism) * float(pose_ok) + dynamism * float(motion_ok)
        if score < 0.5:
            return False

        # 动作类型（录制时捕获了明显参与部位）必须真的在动（防静止误触）
        active_segs = conditions.get("active_segments") or []
        if active_segs and not motion_ok:
            return False

        # 手部重点：动态动作若录制时捕获了手部，手指张开度必须命中
        #（手型是区分"幅度相近但手型不同"动作的关键判别，见 _hand_pose_matches）
        if dynamism >= 0.5 and "手部" in active_segs:
            if not self._hand_pose_matches(conditions, motion_features):
                return False

        # 附加离散区分（录制时若稳定录入了 posture / hand_position / 精细手型）
        for key in ("posture", "hand_position", "hand_gesture_L", "hand_gesture_R"):
            if key in conditions:
                allowed = conditions[key]
                allowed = [allowed] if isinstance(allowed, str) else list(allowed)
                if motion_features.get(key) not in allowed:
                    return False
        return True

    def _pose_segments_match(self, conditions: dict, motion_features: dict) -> bool:
        """五段构型硬门槛：录制捕获的每个部位都必须命中。

        部位只取 conditions["pose"] 中实际存在的键（即录制时明显参与动作的
        部位）；每个部位内部各键距离取均值后与该段 σ 上限比较，全部
        < CUSTOM_POSE_SEG_SIGMA 才通过——任一捕获部位超限即整体拒绝。
        """
        pose = conditions.get("pose") or {}
        pose_std = conditions.get("pose_std") or {}
        if not pose:
            # 无构型模板（纯运动类旧预设）：无部位约束，放行到运动确认
            return True

        checked_segments = 0
        for _seg_name, seg_keys in POSE_SEGMENTS:
            pkeys = [k for k in seg_keys
                     if isinstance(pose.get(k), (int, float))
                     and k in motion_features
                     and isinstance(motion_features[k], (int, float))]
            if not pkeys:
                continue  # 录制时该部位未捕获 → 不约束
            checked_segments += 1
            d_sq = 0.0
            for k in pkeys:
                std = max(float(pose_std.get(k, 0.0) or 0.0), CUSTOM_POSE_STD_FLOOR)
                dev = (motion_features[k] - float(pose[k])) / std
                d_sq += dev ** 2
                if abs(dev) >= CUSTOM_POSE_KEY_SIGMA:
                    return False  # 段内任一键大幅偏离 → 该部位未命中
            d_seg = (d_sq / len(pkeys)) ** 0.5
            if not (d_seg < CUSTOM_POSE_SEG_SIGMA):
                return False  # 任一捕获部位未命中 → 整体判定失败
        return checked_segments > 0

    def _hand_pose_matches(self, conditions: dict, motion_features: dict) -> bool:
        """手部重点约束：手指张开度（pose_spread）必须与录制模板一致。

        手型（张掌/握拳/比划）在动作过程中保持稳定，是区分"运动幅度相近
        但手型不同"的动作（如挥手 vs 比划/握拳）的关键判别——运动量判定
        无法区分它们，故动态动作在运动判定之外单独校验手指张开度。
        """
        pose = conditions.get("pose") or {}
        pose_std = conditions.get("pose_std") or {}
        # 模板未捕获手指张开度（如手部可见度不足）→ 不设手型约束
        if not any(isinstance(pose.get(k), (int, float))
                   for k in ("pose_spread_L", "pose_spread_R")):
            return True
        checked = 0
        for k in ("pose_spread_L", "pose_spread_R"):
            if (isinstance(pose.get(k), (int, float))
                    and k in motion_features
                    and isinstance(motion_features[k], (int, float))):
                std = max(float(pose_std.get(k, 0.0) or 0.0), CUSTOM_POSE_STD_FLOOR)
                dev = (motion_features[k] - float(pose[k])) / std
                if abs(dev) >= CUSTOM_POSE_KEY_SIGMA:
                    return False  # 手型大幅偏离（如张掌变握拳）→ 拒绝
                checked += 1
        # 模板有张开度键但当前特征全部缺失（关节不可见）→ 保守拒绝
        return checked > 0

    def _motion_matches(self, conditions: dict, motion_features: dict) -> bool:
        """运动距离确认：距动作模板近（<2σ）且偏离静止基线（>1σ）。

        DTW 轨迹形态匹配作为高置信附加信号：命中即视为运动量一致；
        未命中（幅度不足/形态不同）时回退到数值均值距离，不直接拒绝。
        """
        # DTW 轨迹（附加）：形态一致 = 高置信度运动确认
        traj_tpl = conditions.get("trajectory")
        cur_traj = motion_features.get("arm_trajectory")
        if traj_tpl and cur_traj and len(cur_traj) >= 2:
            try:
                cur_arr = np.asarray(
                    [[float(p[0]), float(p[1])] for p in cur_traj], dtype=float)
                raw_span = float(np.abs(cur_arr - cur_arr.mean(axis=0)).max())
            except (ValueError, TypeError, IndexError):
                raw_span = 0.0
            if raw_span >= DTW_MIN_AMPLITUDE:
                cur_norm = normalize_trajectory(resample_trajectory(cur_traj, 20))
                tpl_norm = normalize_trajectory(
                    resample_trajectory(traj_tpl, 20))
                if dtw_distance(cur_norm, tpl_norm) < DTW_MATCH_THRESHOLD:
                    return True

        # 数值均值距离主判据
        action = conditions.get("action") or {}
        baseline = conditions.get("baseline") or {}
        action_std = conditions.get("action_std") or {}
        keys = [k for k in action
                if isinstance(action[k], (int, float))
                and k in motion_features
                and isinstance(motion_features[k], (int, float))]
        if not keys:
            return True  # 无运动模板键（纯静态动作）→ 由构型硬门槛把关
        d_action_sq = 0.0
        d_baseline_sq = 0.0
        for k in keys:
            std = max(float(action_std.get(k, 0.0) or 0.0), CUSTOM_MOTION_STD_FLOOR)
            v = motion_features[k]
            d_action_sq += ((v - float(action[k])) / std) ** 2
            b = float(baseline.get(k, 0.0))
            d_baseline_sq += ((v - b) / std) ** 2
        n = len(keys)
        d_action = (d_action_sq / n) ** 0.5
        d_baseline = (d_baseline_sq / n) ** 0.5
        return (d_action < CUSTOM_MOTION_SIGMA
                and d_baseline > CUSTOM_BASELINE_SIGMA)


class PresetEngine:
    """
    预设匹配引擎
    
    核心职责：根据实时动作特征，匹配并返回最高优先级的触发动作。
    匹配采用"持续时间阈值"机制，避免动作抖动导致的误触发。
    """
    
    # 支持数值范围检查的特征键（对应 motion_features 中的 float 值）
    _NUMERIC_KEYS = {"motion_energy", "body_speed", "arm_amplitude", "body_rhythm", "arm_motion_energy", "arm_horizontal_motion", "wave_direction", "palm_hover"}
    # 支持列表/枚举匹配的特征键（hand_gesture_L/R = 精细手型，可选约束）
    _LIST_KEYS = {"posture", "hand_position", "hand_gesture_L", "hand_gesture_R"}
    # 支持布尔匹配的特征键
    _BOOL_KEYS = {"is_active"}
    
    def __init__(self):
        """初始化预设匹配引擎"""
        self.presets: List[GesturePreset] = []
        self.active_preset: Optional[GesturePreset] = None
        # 记录每个预设的持续满足时间（秒），键为预设名称
        self.hold_timers: Dict[str, float] = {}
        # 自定义动作独立计时器：与通用动作的 hold_timers 完全分离
        self._custom_hold_timers: Dict[str, float] = {}
        # 自定义动作独立判定器（五部分判断法，参数/逻辑均独立）
        self._custom_matcher = CustomActionMatcher()
        # 记录上次 match 调用的时间戳，用于计算时间差
        self._last_match_time: Optional[float] = None
        # 举手（palm_hover）路由保持截止时间：palm_hover 一旦激活，
        # 在 HOVER_HOLD_SEC 内即使瞬时失败（抖动超限导致某帧 <0.5）也
        # 保持 hover 路由——防止"交警掌心向前"手势的抖动帧落到 custom，
        # 被构型相似的"向前指"自定义模板抢走（举手暂停串自定义根因）。
        self._hover_until: float = 0.0
        # hover 保持时长（秒）：覆盖 palm_hover 悬停抖动超限的瞬时帧
        self._hover_hold_sec: float = 0.5
    
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
        self._custom_hold_timers = {}
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
        self._custom_hold_timers[preset.name] = 0.0  # 自定义独立计时器
    
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
                self._custom_hold_timers.pop(name, None)
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

        自定义预设（方案A）使用模板距离匹配：conditions 含 "action"/"baseline"
        结构时走 _match_template——当前特征须距动作模板近（<2σ）且明显偏离
        中性基线（>1σ）才匹配，静止时特征≈基线天然不满足。

        Args:
            preset: 要检查的预设
            motion_features: 当前动作特征向量

        Returns:
            bool: 所有条件均满足返回 True，任一不满足返回 False
        """
        conditions = preset.conditions

        # 注意：自定义动作（conditions 含 action/baseline 模板）不经过此方法，
        # 由 PresetEngine.match 路由到独立的 CustomActionMatcher（五部分判断法）
        # 判定，本方法仅处理通用动作的数值/枚举/布尔范围条件。

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

    def _preset_category(self, preset) -> str:
        """返回预设的动作类别（借鉴虚拟主播"关节→骨骼一对一映射"思路）：
        - "hover": 举手暂停（palm_hover 专属）
        - "wave":  挥手切歌/上一首（wave_direction 专属）
        - "custom": 自定义动作（轨迹/均值模板专属）
        - "other": 其他（mood 切换等，不参与互斥路由）
        """
        if getattr(preset, '_is_custom', False):
            return "custom"
        action = preset.action
        if action == "player_pause":
            return "hover"
        if action in ("player_next", "player_prev"):
            return "wave"
        return "other"

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
        dtw_custom_holding = False  # 有 DTW 轨迹模板的自定义正在匹配（高置信度）

        for preset in self.presets:
            preset_name = preset.name
            # 获取持续时间阈值，默认为 0.0（即立即触发）
            duration = preset.conditions.get("duration", 0.0)

            # 路由：模板型自定义动作（录制产生 action/baseline/pose 模板）走
            # 独立判定器（五部分判断法 + 静态/动态混合）；旧格式范围条件预设
            #（min/max 结构、无模板）与通用动作走 _check_conditions 范围判定。
            # 关键：旧格式预设若进新判定器会被当成"无模板约束"而任意命中
            #（此前误触变频繁的根因），必须按条件结构分流。
            if (getattr(preset, '_is_custom', False)
                    and isinstance(preset.conditions.get("action"), dict)):
                is_matching = self._custom_matcher.match(
                    preset.conditions, motion_features)
            else:
                is_matching = self._check_conditions(preset, motion_features)

            if is_matching:
                if getattr(preset, '_is_custom', False):
                    custom_holding = True
                    # DTW 轨迹模板 = 高置信度自定义：匹配形态而非区间，
                    # 精确匹配时优先级应高于通用动作
                    if isinstance(preset.conditions.get("trajectory"), list):
                        dtw_custom_holding = True
                # 条件满足：累加持续时间计时器（自定义动作用独立计时器）
                timers = (self._custom_hold_timers
                          if getattr(preset, '_is_custom', False)
                          else self.hold_timers)
                timers[preset_name] = timers.get(preset_name, 0.0) + delta_time

                # 检查是否达到持续时间阈值
                if timers[preset_name] >= duration:
                    matched_presets.append(preset)
            else:
                # 条件不满足：立即清零计时器，避免条件恢复后直接继承旧时间
                timers = (self._custom_hold_timers
                          if getattr(preset, '_is_custom', False)
                          else self.hold_timers)
                timers[preset_name] = 0.0

        # 动作类别路由（借鉴虚拟主播"关节→骨骼一对一映射"思路）：
        # 每帧先确定唯一动作类别，只在该类别内匹配，其他类别计时器清零——
        # 从累积阶段就杜绝"同一特征被多类动作共享"导致的串扰。
        # 路由按"专属信号优先"：palm_hover / wave_direction 是通用动作的
        # 专属判别特征，一旦激活即明确是举手/挥手，自定义动作让位——
        # 否则做挥手时若肘部轨迹形态碰巧接近自定义模板，会被自定义抢占。
        # 1. palm_hover ≥ 0.5 → hover（举手暂停专属特征，最高优先）
        # 2. |wave_direction| ≥ 0.5 → wave（挥手专属特征）
        # 3. DTW 自定义匹配（高置信度）→ custom
        # 4. 普通自定义匹配 → custom
        # 5. 都不满足 → None（无明确动作，全部清零）
        # hover 路由保持：palm_hover 一旦激活（≥0.5），在 _hover_hold_sec
        # 窗口内即使某帧瞬时失败（悬停抖动超限使 palm_hover <0.5）也保持
        # hover 路由——"交警掌心向前"手势的抖动帧不会落到 custom，被
        # 构型相似的"向前指"自定义模板抢走（举手暂停串自定义根因）。
        hover_raw = motion_features.get("palm_hover", 0.0)
        if hover_raw >= 0.5:
            self._hover_until = current_timestamp + self._hover_hold_sec
        hover_active = hover_raw >= 0.5 or current_timestamp < self._hover_until
        wave_active = abs(motion_features.get("wave_direction", 0.0)) >= 0.5
        if hover_active:
            route = "hover"
        elif wave_active:
            # 挥手专属信号优先于自定义：做挥手动作时 route=wave，
            # 自定义让位——否则挥手的肘部轨迹/特征碰巧接近自定义模板时
            # 会被自定义抢占（"自定义动作串台/误触切歌"根因）。
            # 自定义动作录制的是非挥手类手势（拍手/比划等），
            # 其触发时不激活 wave_direction，仍能正常走 custom 路由。
            route = "wave"
        elif dtw_custom_holding:
            # 高置信度自定义（含轨迹模板的形态匹配）
            route = "custom"
        elif custom_holding:
            route = "custom"
        else:
            route = None

        if route is None:
            for p in self.presets:
                self.hold_timers[p.name] = 0.0
                self._custom_hold_timers[p.name] = 0.0
            self.active_preset = None
            return None

        # 只保留路由类别内的可触发预设，非路由类别计时器清零
        matched_presets = [
            p for p in matched_presets
            if self._preset_category(p) == route
        ]
        for p in self.presets:
            if self._preset_category(p) != route:
                self.hold_timers[p.name] = 0.0
                self._custom_hold_timers[p.name] = 0.0

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
