"""
wave_learning_rules.py - 挥手触发规则白名单（增强版）

白名单验证机制：
1. 方向白名单：验证 方向→动作 是否匹配
2. 视觉特征白名单：验证动作形态是否符合目标动作的特征标准

增强策略：
- 不仅验证方向，还验证动作的视觉特征（幅度、速度、形态）
- 基于历史学习数据动态调整视觉特征阈值
- 过滤掉"方向对但动作形态不对"的误触
"""

from typing import Tuple, Optional, Dict, Any
from enum import Enum


class TriggerAction(Enum):
    """触发动作类型"""
    NEXT = "player_next"      # 下一首
    PREV = "player_prev"      # 上一首


# ═══════════════════════════════════════════════════════
#  白名单规则定义（简化版：只关注方向）
# ═══════════════════════════════════════════════════════

# 合法触发规则：direction → action
# direction: >0 表示向左, <0 表示向右
# 不区分左右手，任意手挥动都适用
LEGAL_TRIGGER_RULES = {
    # 手向左 → 下一首（左右手通用）
    1.0: TriggerAction.NEXT,
    # 手向右 → 上一首（左右手通用）
    -1.0: TriggerAction.PREV,
}


# ═══════════════════════════════════════════════════════
#  视觉特征白名单（增强版）
# ═══════════════════════════════════════════════════════

class VisualFeatureWhitelist:
    """视觉特征白名单

    基于动作的视觉特征（幅度、速度、形态）验证是否为有效触发
    防止"方向对但动作形态不对"的误触
    """

    def __init__(self):
        # 动作幅度阈值（基于历史学习数据动态调整）
        self.action_thresholds = {
            'wave': {
                # 基础阈值（冷启动期使用）
                'min_direction': 0.3,        # 最小方向值（wave_direction绝对值）
                'min_arm_motion': 0.1,       # 最小手臂横向运动幅度
                'min_x_span': 0.03,          # 最小x跨度（归一化）
                'min_ang_span': 0.5,         # 最小角度跨度（度）

                # 动态阈值（学习后调整，key: (hand, direction)）
                'dynamic': {}
            },
            # 可扩展其他动作类型（举手、自定义动作等）
        }

        # 动作形态统计（用于学习）
        self.action_stats = {
            'wave': {
                # 历史最大幅度（用于归一化）
                'max_direction': 0.0,
                'max_arm_motion': 0.0,
                'max_x_span': 0.0,
                'max_ang_span': 0.0,

                # 历史最小幅度（用于过滤过小动作）
                'min_direction': float('inf'),
                'min_arm_motion': float('inf'),
            }
        }

    def validate_wave(self, hand: str, direction: float,
                      wave_direction: float,
                      arm_motion: float,
                      x_span: float = 0.0,
                      ang_span: float = 0.0) -> Tuple[bool, str]:
        """
        验证挥手动作的视觉特征是否通过白名单

        Args:
            hand: 手别 'L' 或 'R'
            direction: 方向 +1.0(左) 或 -1.0(右)
            wave_direction: 实际检测到的方向值（绝对值越大越明显）
            arm_motion: 手臂横向运动幅度
            x_span: x方向跨度（可选）
            ang_span: 角度跨度（可选，单位：度）

        Returns:
            (is_valid, reason) - 是否通过验证及原因
        """
        thresholds = self.action_thresholds['wave']

        # 获取动态阈值（如果有）
        key = (hand, direction)
        if key in thresholds['dynamic']:
            dyn = thresholds['dynamic'][key]
            min_dir = dyn.get('min_direction', thresholds['min_direction'])
            min_arm = dyn.get('min_arm_motion', thresholds['min_arm_motion'])
        else:
            min_dir = thresholds['min_direction']
            min_arm = thresholds['min_arm_motion']

        # 1️⃣ 方向幅度检查
        wave_dir_abs = abs(wave_direction)
        if wave_dir_abs < min_dir:
            return False, f"方向幅度不足({wave_dir_abs:.2f}<{min_dir:.2f})"

        # 2️⃣ 手臂运动检查
        if arm_motion < min_arm:
            return False, f"手臂运动不足({arm_motion:.2f}<{min_arm:.2f})"

        # 3️⃣ 可选：x跨度检查（如果有数据）
        if x_span > 0 and x_span < thresholds['min_x_span']:
            return False, f"x跨度不足({x_span:.2f}<{thresholds['min_x_span']:.2f})"

        # 4️⃣ 可选：角度跨度检查（如果有数据）
        if ang_span > 0 and ang_span < thresholds['min_ang_span']:
            return False, f"角度跨度不足({ang_span:.1f}°<{thresholds['min_ang_span']:.1f}°)"

        # 5️⃣ 动作幅度合理性检查（防止异常大值）
        if wave_dir_abs > 2.0 or arm_motion > 1.0:
            return False, f"动作异常过大(wave_dir={wave_dir_abs:.2f}, arm_motion={arm_motion:.2f})"

        return True, "视觉特征验证通过"

    def update_thresholds(self, hand: str, direction: float,
                          wave_direction: float,
                          arm_motion: float,
                          triggered: bool):
        """
        基于历史触发结果动态调整阈值

        Args:
            hand: 手别
            direction: 方向
            wave_direction: 方向幅度
            arm_motion: 手臂运动幅度
            triggered: 是否成功触发
        """
        key = (hand, direction)
        thresholds = self.action_thresholds['wave']

        # 更新统计
        stats = self.action_stats['wave']
        stats['max_direction'] = max(stats['max_direction'], abs(wave_direction))
        stats['max_arm_motion'] = max(stats['max_arm_motion'], arm_motion)
        stats['min_direction'] = min(stats['min_direction'], abs(wave_direction))
        stats['min_arm_motion'] = min(stats['min_arm_motion'], arm_motion)

        # 动态调整阈值（仅在有足够样本后）
        if triggered:
            # 成功触发：适当放宽阈值（提升触发率）
            if key not in thresholds['dynamic']:
                thresholds['dynamic'][key] = {
                    'min_direction': max(0.2, abs(wave_direction) * 0.7),
                    'min_arm_motion': max(0.05, arm_motion * 0.7),
                }
            else:
                dyn = thresholds['dynamic'][key]
                # 缓慢降低门槛（指数移动平均）
                alpha = 0.1
                dyn['min_direction'] = dyn['min_direction'] * (1 - alpha) + max(0.2, abs(wave_direction) * 0.7) * alpha
                dyn['min_arm_motion'] = dyn['min_arm_motion'] * (1 - alpha) + max(0.05, arm_motion * 0.7) * alpha

    def get_visual_features_summary(self) -> Dict[str, Any]:
        """获取视觉特征白名单统计摘要"""
        return {
            'thresholds': self.action_thresholds,
            'stats': self.action_stats,
            'dynamic_thresholds_count': len(self.action_thresholds['wave']['dynamic'])
        }


def is_legal_trigger(hand: str, direction: float, expected_action: str) -> bool:
    """检查触发是否符合白名单规则（简化版：忽略 hand 参数）

    Args:
        hand: 手别 'L' 或 'R'（已忽略，仅为兼容旧接口）
        direction: 方向 >0(左) 或 <0(右)
        expected_action: 预期触发的动作

    Returns:
        是否为合法触发
    """
    # 只取方向符号，忽略手别
    dir_sign = 1.0 if direction > 0 else -1.0

    # 查找白名单
    if dir_sign not in LEGAL_TRIGGER_RULES:
        return False

    # 检查动作是否匹配
    legal_action = LEGAL_TRIGGER_RULES[dir_sign]
    return legal_action.value == expected_action


def get_expected_action(hand: str, direction: float) -> Optional[str]:
    """获取白名单中预期的动作

    Args:
        hand: 手别
        direction: 方向

    Returns:
        预期的动作字符串，如果不合法返回None
    """
    hand = hand.upper()
    if hand not in ('L', 'R'):
        return None

    dir_sign = 1.0 if direction > 0 else -1.0
    key = (hand, dir_sign)

    if key not in LEGAL_TRIGGER_RULES:
        return None

    return LEGAL_TRIGGER_RULES[key].value


def validate_and_filter_trigger(hand: str, direction: float,
                                 triggered_action: str,
                                 confidence: float = 1.0) -> Tuple[bool, str]:
    """验证触发是否符合白名单规则，并决定是否记录到学习系统

    过滤逻辑：
    1. 首先检查是否在白名单中
    2. 如果不在白名单，直接返回False（不记录）
    3. 如果在白名单但置信度<0.5，也返回False
    4. 只有在白名单且置信度≥0.5才返回True

    Args:
        hand: 手别
        direction: 方向
        triggered_action: 实际触发的动作
        confidence: 触发置信度

    Returns:
        (should_record, reason) - 是否应该记录到学习系统及原因
    """
    # 第1步：检查是否在白名单中
    if not is_legal_trigger(hand, direction, triggered_action):
        return False, f"不在白名单({hand}手向{'左' if direction > 0 else '右'}→{triggered_action})"

    # 第2步：检查置信度
    if confidence < 0.5:
        return False, f"白名单内但置信度不足({confidence:.2f}<0.5)"

    # 通过所有检查
    return True, "白名单验证通过"


def validate_trigger_with_visual_features(
    hand: str,
    direction: float,
    triggered_action: str,
    confidence: float,
    wave_direction: float,
    arm_motion: float,
    x_span: float = 0.0,
    ang_span: float = 0.0,
    visual_whitelist: Optional[VisualFeatureWhitelist] = None
) -> Tuple[bool, str]:
    """
    增强版白名单验证：方向 + 视觉特征双重验证

    验证流程：
    1. 方向白名单验证（方向→动作是否匹配）
    2. 置信度验证（是否≥0.5）
    3. 视觉特征白名单验证（动作形态是否符合标准）

    Args:
        hand: 手别 'L' 或 'R'
        direction: 方向 +1.0(左) 或 -1.0(右)
        triggered_action: 实际触发的动作
        confidence: 触发置信度
        wave_direction: 实际检测到的方向幅度（归一化）
        arm_motion: 手臂横向运动幅度（归一化）
        x_span: x方向跨度（可选）
        ang_span: 角度跨度（可选，单位：度）
        visual_whitelist: 视觉特征白名单实例（可选，None则跳过视觉验证）

    Returns:
        (should_record, reason) - 是否应该记录到学习系统及原因
    """
    # 第1步：方向白名单验证
    if not is_legal_trigger(hand, direction, triggered_action):
        return False, f"方向白名单拦截({hand}手向{'左' if direction > 0 else '右'}→{triggered_action})"

    # 第2步：置信度验证
    if confidence < 0.5:
        return False, f"置信度不足({confidence:.2f}<0.5)"

    # 第3步：视觉特征验证（可选）
    if visual_whitelist is not None:
        is_valid, reason = visual_whitelist.validate_wave(
            hand=hand,
            direction=direction,
            wave_direction=wave_direction,
            arm_motion=arm_motion,
            x_span=x_span,
            ang_span=ang_span
        )
        if not is_valid:
            return False, f"视觉特征白名单拦截({reason})"

    # 通过所有检查
    return True, "白名单验证通过（方向+视觉特征）"


def print_rules():
    """打印白名单规则（用于调试）"""
    print("=" * 60)
    print("挥手触发白名单规则")
    print("=" * 60)
    print("  简化版：只关注方向，不区分左右手")
    for direction, action in LEGAL_TRIGGER_RULES.items():
        dir_str = "向左" if direction > 0 else "向右"
        action_str = "下一首" if action == TriggerAction.NEXT else "上一首"
        print(f"  手{dir_str} -> {action_str}")
    print("=" * 60)


# 全局视觉特征白名单实例（单例模式）
_visual_whitelist_instance = None

def get_visual_whitelist() -> VisualFeatureWhitelist:
    """获取全局视觉特征白名单实例（单例）

    Returns:
        VisualFeatureWhitelist 实例
    """
    global _visual_whitelist_instance
    if _visual_whitelist_instance is None:
        _visual_whitelist_instance = VisualFeatureWhitelist()
    return _visual_whitelist_instance


# 测试入口
if __name__ == "__main__":
    print_rules()

    # 基础白名单测试
    print("\n=== 基础方向白名单测试 ===")
    test_cases = [
        ("L", 1.0, "player_next", 0.85),   # ✅ 合法，高置信度
        ("L", -1.0, "player_prev", 0.85),  # ✅ 合法，高置信度
        ("R", 1.0, "player_next", 0.85),   # ✅ 合法，高置信度
        ("R", -1.0, "player_prev", 0.85),  # ✅ 合法，高置信度
        ("L", 1.0, "player_next", 0.35),   # ❌ 白名单内但置信度不足
        ("L", 1.0, "player_prev", 0.85),   # ❌ 动作不匹配
        ("L", 1.0, "player_next", 0.85),   # ✅ 合法
        ("L", -1.0, "player_next", 0.85),  # ❌ 不在白名单
    ]

    print("\n测试结果：")
    for hand, direction, action, confidence in test_cases:
        should_record, reason = validate_and_filter_trigger(hand, direction, action, confidence)
        status = "[OK] 记录" if should_record else "[SKIP] 过滤"
        print(f"  {status} | {hand}手向{'左' if direction > 0 else '右'} -> {action} (置信度={confidence:.2f}): {reason}")

    # 视觉特征白名单测试
    print("\n\n=== 视觉特征白名单测试 ===")
    vw = get_visual_whitelist()

    visual_test_cases = [
        # (hand, direction, wave_direction, arm_motion, expected, desc)
        ("R", 1.0, 0.85, 0.15, True, "正常挥手"),
        ("R", 1.0, 0.25, 0.15, False, "方向幅度不足"),
        ("R", 1.0, 0.85, 0.05, False, "手臂运动不足"),
        ("R", 1.0, 0.85, 0.15, 0.02, False, "x跨度不足"),
        ("L", -1.0, 0.90, 0.20, True, "正常挥手"),
    ]

    for case in visual_test_cases:
        if len(case) == 6:
            hand, direction, wave_dir, arm_motion, expected, desc = case
            is_valid, reason = vw.validate_wave(hand, direction, wave_dir, arm_motion)
        else:
            hand, direction, wave_dir, arm_motion, x_span, expected, desc = case
            is_valid, reason = vw.validate_wave(hand, direction, wave_dir, arm_motion, x_span=x_span)

        status = "[OK] 通过" if is_valid == expected else "[FAIL] 未通过"
        print(f"  {status} | {desc}: {reason}")

    print("\n")
