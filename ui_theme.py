# -*- coding: utf-8 -*-
"""
ui_theme.py — SoulPlayer 现代化主题与绘制工具

提供：
- 现代化配色方案（暗色主题 + 青绿强调）
- 圆角矩形、阴影、渐变、毛玻璃效果绘制
- 图标字体辅助绘制（播放/暂停/音量等）
- 排版工具

设计原则：
- 低饱和度暖暗色背景，高对比度文字
- 青绿色 (#00d4aa) 强调色，体现"灵性"
- 圆角 8-12px 统一面板风格
- 微妙阴影增加层次感
"""

from typing import Tuple, Optional, List
import pygame
import numpy as np


# ═══════════════════════════════════════════════════════
# 配色方案（Soul 主题）
# ═══════════════════════════════════════════════════════
class SoulTheme:
    """SoulPlayer 统一主题配置"""

    # 背景层级
    BG_DEEP     = (26, 26, 46)      # #1a1a2e 最深背景
    BG_PANEL    = (30, 30, 56)      # #1e1e38 面板背景
    BG_CARD     = (35, 35, 62)      # #23233e 卡片背景
    BG_HOVER    = (42, 42, 72)      # #2a2a48 悬停背景
    BG_INPUT    = (22, 22, 40)      # #161628 输入框背景

    # 文字层级
    TEXT_PRIMARY   = (238, 238, 246)  # #eeeef6 主文字
    TEXT_SECONDARY = (160, 160, 180)  # #a0a0b4 次要文字
    TEXT_MUTED     = (100, 100, 120)  # #646478 提示文字
    TEXT_ACCENT    = (0, 212, 170)    # #00d4aa 强调文字

    # 强调色（灵性青绿）
    ACCENT       = (0, 212, 170)     # #00d4aa
    ACCENT_SOFT  = (0, 180, 145)     # #00b491
    ACCENT_DIM   = (0, 140, 115)     # #008c73
    ACCENT_GLOW  = (0, 212, 170, 80)  # 带透明度的发光色

    # 状态色
    SUCCESS = (0, 212, 170)   # 青绿
    WARNING = (255, 180, 0)   # 琥珀
    DANGER  = (255, 90, 90)   # 柔和红
    INFO    = (100, 180, 255) # 天蓝

    # 进度条
    BAR_BG      = (45, 45, 70)   # 背景条
    BAR_LOW     = (80, 80, 110)  # 低值
    BAR_MID     = (0, 180, 145)  # 中值
    BAR_HIGH    = (0, 212, 170)  # 高值
    BAR_GLOW    = (0, 212, 170, 120)  # 发光

    # 边框与分隔
    BORDER      = (50, 50, 78)   # 常规边框
    BORDER_LIGHT = (65, 65, 95)  # 高亮边框
    DIVIDER     = (40, 40, 65)   # 分隔线

    # 阴影
    SHADOW_COLOR = (10, 10, 20, 100)  # 阴影色（带 alpha）

    # 圆角半径
    RADIUS_PANEL = 12
    RADIUS_CARD  = 8
    RADIUS_BAR   = 4
    RADIUS_BTN   = 6

    # 间距
    PAD = 12
    GAP = 8


# ═══════════════════════════════════════════════════════
# 绘制辅助函数
# ═══════════════════════════════════════════════════════

class UIRenderer:
    """
    UI 渲染器：封装所有高级绘制效果。
    所有方法均接收 pygame.Surface 作为第一个参数。
    """

    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self._shadow_cache: dict = {}  # 缓存阴影 Surface

    # ── 圆角矩形 ──
    def rounded_rect(self, rect: pygame.Rect, color: Tuple[int, int, int],
                     radius: int = SoulTheme.RADIUS_CARD, border: int = 0,
                     border_color: Optional[Tuple[int, int, int]] = None) -> None:
        """绘制实心圆角矩形，可带边框"""
        if border > 0 and border_color:
            # 外框（边框色）
            pygame.draw.rect(self.screen, border_color, rect, border_radius=radius)
            # 内框（填充色）
            inner = rect.inflate(-border * 2, -border * 2)
            pygame.draw.rect(self.screen, color, inner, border_radius=max(0, radius - border))
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)

    # ── 阴影 ──
    def shadow(self, rect: pygame.Rect, radius: int = SoulTheme.RADIUS_PANEL,
               blur: int = 10, offset: Tuple[int, int] = (0, 4)) -> None:
        """在 rect 下方绘制柔和阴影"""
        cache_key = (rect.width, rect.height, radius, blur, offset)
        if cache_key not in self._shadow_cache:
            # 创建阴影 surface
            w, h = rect.width + blur * 2, rect.height + blur * 2
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(surf, SoulTheme.SHADOW_COLOR,
                             (blur, blur, rect.width, rect.height), border_radius=radius)
            # 简单模糊：缩放缩小再放大
            if blur > 0:
                small = pygame.transform.smoothscale(surf, (w // 2, h // 2))
                blurred = pygame.transform.smoothscale(small, (w, h))
                self._shadow_cache[cache_key] = blurred
            else:
                self._shadow_cache[cache_key] = surf

        shadow_surf = self._shadow_cache[cache_key]
        self.screen.blit(shadow_surf, (rect.x - blur + offset[0], rect.y - blur + offset[1]))

    # ── 毛玻璃面板 ──
    def glass_panel(self, rect: pygame.Rect, color: Tuple[int, int, int] = SoulTheme.BG_PANEL,
                    radius: int = SoulTheme.RADIUS_PANEL, alpha: int = 220) -> None:
        """绘制半透明毛玻璃风格面板"""
        # 创建带 alpha 的 surface
        surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*color, alpha), (0, 0, rect.width, rect.height), border_radius=radius)
        # 高光边框
        pygame.draw.rect(surf, (*SoulTheme.BORDER_LIGHT, 60),
                         (0, 0, rect.width, rect.height), border_radius=radius, width=1)
        self.screen.blit(surf, (rect.x, rect.y))

    # ── 渐变进度条 ──
    def gradient_bar(self, x: int, y: int, width: int, height: int, progress: float,
                     color_low: Tuple[int, int, int] = SoulTheme.BAR_LOW,
                     color_high: Tuple[int, int, int] = SoulTheme.BAR_HIGH,
                     bg_color: Tuple[int, int, int] = SoulTheme.BAR_BG,
                     radius: int = SoulTheme.RADIUS_BAR) -> None:
        """绘制带动画感的渐变进度条"""
        progress = float(np.clip(progress, 0.0, 1.0))
        # 背景条
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=radius)

        if progress > 0:
            # 前景条（带发光）
            fill_width = int(width * progress)
            fill_rect = pygame.Rect(x, y, fill_width, height)

            # 根据进度值选择颜色
            if progress < 0.3:
                fill_color = SoulTheme.BAR_LOW
            elif progress < 0.7:
                fill_color = SoulTheme.BAR_MID
            else:
                fill_color = SoulTheme.BAR_HIGH

            pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=radius)

            # 高光端点（营造发光效果）
            if progress > 0.1 and fill_width > 4:
                glow_w = min(8, fill_width)
                glow_rect = pygame.Rect(x + fill_width - glow_w, y, glow_w, height)
                glow_surf = pygame.Surface((glow_w, height), pygame.SRCALPHA)
                for i in range(glow_w):
                    alpha = int(180 * (1 - i / glow_w))
                    pygame.draw.line(glow_surf, (*SoulTheme.ACCENT, alpha), (i, 0), (i, height))
                self.screen.blit(glow_surf, (glow_rect.x, glow_rect.y))

    # ── 脉冲圆点 ──
    def pulse_dot(self, x: int, y: int, radius: int, color: Tuple[int, int, int],
                  pulse_phase: float = 0.0) -> None:
        """绘制脉冲动画圆点（用于录制状态等）"""
        # 外圈脉冲
        pulse_r = radius + int(6 * (0.5 + 0.5 * np.sin(pulse_phase * 2 * np.pi)))
        pulse_surf = pygame.Surface((pulse_r * 2, pulse_r * 2), pygame.SRCALPHA)
        pulse_alpha = int(80 * (0.5 + 0.5 * np.cos(pulse_phase * 2 * np.pi)))
        pygame.draw.circle(pulse_surf, (*color, pulse_alpha), (pulse_r, pulse_r), pulse_r)
        self.screen.blit(pulse_surf, (x - pulse_r, y - pulse_r))
        # 实心核心
        pygame.draw.circle(self.screen, color, (x, y), radius)
        # 高光
        pygame.draw.circle(self.screen, (255, 255, 255, 180), (x - 1, y - 1), max(1, radius // 3))

    # ── 图标按钮 ──
    def icon_button(self, rect: pygame.Rect, icon_text: str, font: pygame.font.Font,
                    color: Tuple[int, int, int] = SoulTheme.BG_CARD,
                    hover: bool = False, active: bool = False) -> bool:
        """绘制图标按钮，返回鼠标是否悬停"""
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)

        # 背景
        bg = SoulTheme.BG_HOVER if is_hovered else color
        if active:
            bg = SoulTheme.ACCENT_DIM
        self.rounded_rect(rect, bg, radius=SoulTheme.RADIUS_BTN)

        # 图标文字
        text_surf = font.render(icon_text, True, SoulTheme.TEXT_PRIMARY)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

        return is_hovered

    # ── 文字标签 ──
    def label(self, x: int, y: int, text: str, font: pygame.font.Font,
              color: Tuple[int, int, int] = SoulTheme.TEXT_PRIMARY,
              align: str = "left") -> pygame.Rect:
        """绘制文字标签，返回文字区域"""
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if align == "center":
            rect.center = (x, y)
        elif align == "right":
            rect.right = x
            rect.y = y
        else:
            rect.x = x
            rect.y = y
        self.screen.blit(surf, rect)
        return rect

    # ── 分隔线 ──
    def divider(self, x1: int, y1: int, x2: int, y2: int,
                color: Tuple[int, int, int] = SoulTheme.DIVIDER, width: int = 1) -> None:
        """绘制渐变分隔线"""
        pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

    # ── 卡片 ──
    def card(self, rect: pygame.Rect, title: Optional[str] = None,
             font_title: Optional[pygame.font.Font] = None) -> None:
        """绘制标准卡片容器（阴影 + 圆角 + 可选标题）"""
        self.shadow(rect, radius=SoulTheme.RADIUS_CARD)
        self.glass_panel(rect, radius=SoulTheme.RADIUS_CARD)
        if title and font_title:
            self.label(rect.x + 12, rect.y + 10, title, font_title,
                       color=SoulTheme.TEXT_SECONDARY)


# ═══════════════════════════════════════════════════════
# 图标字符映射（使用 Unicode 符号作为简单图标）
# ═══════════════════════════════════════════════════════
ICONS = {
    "play": "▶",
    "pause": "⏸",
    "stop": "⏹",
    "next": "⏭",
    "prev": "⏮",
    "volume_high": "🔊",
    "volume_low": "🔉",
    "volume_mute": "🔇",
    "record": "⏺",
    "plus": "＋",
    "trash": "🗑",
    "music": "🎵",
    "camera": "📷",
    "fire": "🔥",
    "sparkles": "✨",
    "wave": "〰",
    "dot": "●",
    "check": "✓",
    "cross": "✗",
    "arrow_up": "▲",
    "arrow_down": "▼",
    "arrow_left": "◀",
    "arrow_right": "▶",
}


# ═══════════════════════════════════════════════════════
# 骨架绘制（MediaPipe 33 点连接）
# ═══════════════════════════════════════════════════════

# MediaPipe Pose 骨架连线定义（点对索引）
POSE_CONNECTIONS = [
    # 面部
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # 眼睛
    (3, 7), (6, 8), (9, 10),  # 耳朵 + 嘴
    # 躯干
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # 右臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# 连线颜色映射（不同部位不同颜色）
CONNECTION_COLORS = {
    "face": (200, 200, 220),
    "torso": (0, 212, 170),
    "left_arm": (100, 180, 255),
    "right_arm": (255, 140, 100),
    "left_leg": (180, 120, 255),
    "right_leg": (255, 200, 80),
}


def draw_pose_skeleton(surface: pygame.Surface, landmarks: List[dict],
                       frame_w: int, frame_h: int,
                       line_color: Tuple[int, int, int] = SoulTheme.ACCENT,
                       line_width: int = 2, dot_radius: int = 3) -> None:
    """
    在 surface 上绘制 MediaPipe 33 点骨架。

    Args:
        landmarks: 33 个关键点字典列表，每个含 x, y, visibility
        frame_w, frame_h: 摄像头帧的实际尺寸（用于坐标映射）
    """
    if not landmarks or len(landmarks) < 33:
        return

    # 映射到屏幕坐标
    points = []
    for lm in landmarks:
        px = int(lm.get("x", 0) * frame_w)
        py = int(lm.get("y", 0) * frame_h)
        vis = lm.get("visibility", 0)
        points.append((px, py, vis))

    # 绘制连线
    for conn in POSE_CONNECTIONS:
        a, b = conn
        if a < len(points) and b < len(points):
            x1, y1, v1 = points[a]
            x2, y2, v2 = points[b]
            if v1 > 0.5 and v2 > 0.5:
                pygame.draw.line(surface, line_color, (x1, y1), (x2, y2), line_width)

    # 绘制关键点
    for px, py, vis in points:
        if vis > 0.5:
            pygame.draw.circle(surface, line_color, (px, py), dot_radius)
            pygame.draw.circle(surface, (255, 255, 255), (px, py), max(1, dot_radius - 1))


# ═══════════════════════════════════════════════════════
# 运动可视化
# ═══════════════════════════════════════════════════════

def draw_motion_trail(surface: pygame.Surface, landmarks_history: List[List[dict]],
                      frame_w: int, frame_h: int, color: Tuple[int, int, int] = SoulTheme.ACCENT) -> None:
    """
    绘制质心运动轨迹。

    Args:
        landmarks_history: 最近 N 帧的 landmarks 列表
    """
    if len(landmarks_history) < 2:
        return

    centers = []
    for lms in landmarks_history:
        if len(lms) < 33:
            continue
        # 使用髋部中点作为质心
        left_hip = lms[23]
        right_hip = lms[24]
        if left_hip.get("visibility", 0) > 0.5 and right_hip.get("visibility", 0) > 0.5:
            cx = int((left_hip["x"] + right_hip["x"]) / 2 * frame_w)
            cy = int((left_hip["y"] + right_hip["y"]) / 2 * frame_h)
            centers.append((cx, cy))

    # 绘制渐隐轨迹
    for i in range(1, len(centers)):
        alpha = int(255 * (i / len(centers)))
        trail_color = (*color, alpha)
        trail_surf = pygame.Surface((frame_w, frame_h), pygame.SRCALPHA)
        pygame.draw.line(trail_surf, trail_color, centers[i - 1], centers[i], 2)
        surface.blit(trail_surf, (0, 0))


# ═══════════════════════════════════════════════════════
# 辅助：颜色插值
# ═══════════════════════════════════════════════════════

def lerp_color(c1: Tuple[int, ...], c2: Tuple[int, ...], t: float) -> Tuple[int, int, int]:
    """线性插值两个 RGB 颜色，t 在 [0, 1]"""
    t = float(np.clip(t, 0.0, 1.0))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return (r, g, b)


def value_to_color(value: float, low_color: Tuple[int, int, int] = SoulTheme.BAR_LOW,
                   mid_color: Tuple[int, int, int] = SoulTheme.BAR_MID,
                   high_color: Tuple[int, int, int] = SoulTheme.BAR_HIGH) -> Tuple[int, int, int]:
    """将 0~1 的值映射到三段颜色"""
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.5:
        return lerp_color(low_color, mid_color, value * 2)
    else:
        return lerp_color(mid_color, high_color, (value - 0.5) * 2)
