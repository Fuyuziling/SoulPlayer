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

    # 背景层级（近黑蓝绿冷底——未来科技 HUD）
    BG_DEEP     = (6, 11, 15)       # 最深背景
    BG_PANEL    = (10, 19, 25)      # 面板背景
    BG_CARD     = (13, 24, 31)      # 卡片背景
    BG_HOVER    = (18, 33, 42)      # 悬停背景
    BG_INPUT    = (8, 15, 20)       # 输入框背景

    # 文字层级（冷白 → 灰蓝）
    TEXT_PRIMARY   = (216, 238, 240)
    TEXT_SECONDARY = (132, 162, 172)
    TEXT_MUTED     = (72, 102, 112)
    TEXT_ACCENT    = (0, 226, 196)

    # 强调色（灵魂青，HUD 化）
    ACCENT       = (0, 226, 196)
    ACCENT_SOFT  = (0, 180, 160)
    ACCENT_DIM   = (0, 132, 118)
    ACCENT_GLOW  = (0, 226, 196, 70)

    # 状态色
    SUCCESS = (0, 226, 196)
    WARNING = (255, 196, 70)
    DANGER  = (255, 76, 70)
    INFO    = (84, 190, 255)

    # 进度条
    BAR_BG      = (20, 38, 46)   # 背景条
    BAR_LOW     = (60, 100, 110) # 低值
    BAR_MID     = (0, 180, 160)  # 中值
    BAR_HIGH    = (0, 226, 196)  # 高值
    BAR_GLOW    = (0, 226, 196, 120)  # 发光

    # 边框与分隔
    BORDER      = (28, 56, 64)   # 常规边框
    BORDER_LIGHT = (0, 168, 156) # 高亮边框（青）
    DIVIDER     = (22, 46, 54)   # 分隔线

    # HUD 网格 / 扫描线（背景氛围）
    GRID        = (0, 78, 90)
    GRID_DIM    = (0, 52, 62)
    SCANLINE    = (0, 0, 0, 26)

    # 阴影（柔和环境漫射）
    SHADOW_COLOR = (0, 0, 8, 70)

    # 发丝级边框（白 8~10% alpha）
    HAIRLINE = (255, 255, 255, 26)
    HAIRLINE_SOFT = (255, 255, 255, 14)
    # 内层芯顶部受光高光
    INSET_HIGHLIGHT = (255, 255, 255, 30)
    # 双层嵌套（Double-Bezel）的外层壳底色（白 3%）
    SHELL_BG = (255, 255, 255, 8)

    # 环境辉光（背景氛围：青/蓝/紫三团微光，叠加网格）
    GLOW_ACCENT = (0, 226, 196, 22)
    GLOW_BLUE = (84, 190, 255, 16)
    GLOW_PURPLE = (120, 90, 220, 14)

    # 圆角半径
    RADIUS_PANEL = 10
    RADIUS_CARD  = 6
    RADIUS_BAR   = 6
    RADIUS_BTN   = 4

    # 间距
    PAD = 15
    GAP = 10


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
        # 环境辉光背景（Ethereal Glass）：预渲染一次，每帧直接 blit
        self.ambient = self._build_ambient(screen.get_width(), screen.get_height())

    def _build_ambient(self, w: int, h: int) -> pygame.Surface:
        """预渲染 HUD 背景氛围：三团微光 + 网格线 + CRT 扫描线（一次生成，每帧 blit）。"""
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # 三团径向微光（青/蓝/紫，保留原氛围基调）
        orbs = [
            (0.06, 0.05, 0.34, SoulTheme.GLOW_ACCENT),
            (0.96, 0.10, 0.30, SoulTheme.GLOW_BLUE),
            (0.12, 0.94, 0.30, SoulTheme.GLOW_PURPLE),
        ]
        for fx, fy, fr, color in orbs:
            cx, cy = int(w * fx), int(h * fy)
            r = int(max(w, h) * fr)
            for i in range(r, 0, -3):
                a = int(color[3] * (1.0 - i / r))
                pygame.draw.circle(surf, (*color[:3], a), (cx, cy), i)
        # 网格线（HUD 语言）
        spacing = 48
        for gx in range(0, w, spacing):
            pygame.draw.line(surf, (*SoulTheme.GRID_DIM, 26), (gx, 0), (gx, h), 1)
        for gy in range(0, h, spacing):
            pygame.draw.line(surf, (*SoulTheme.GRID_DIM, 26), (0, gy), (w, gy), 1)
        # CRT 扫描线（低 alpha，避免伤可读性）
        for sy in range(0, h, 3):
            pygame.draw.line(surf, (0, 0, 0, 22), (0, sy), (w, sy), 1)
        return surf

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

    # ── 阴影（柔和环境漫射） ──
    def shadow(self, rect: pygame.Rect, radius: int = SoulTheme.RADIUS_PANEL,
               blur: int = 16, offset: Tuple[int, int] = (0, 6)) -> None:
        """在 rect 下方绘制柔和阴影（低 alpha + 大 blur，模拟浮起感）"""
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

    # ── 毛玻璃面板（双重嵌套 Double-Bezel） ──
    def glass_panel(self, rect: pygame.Rect, color: Tuple[int, int, int] = SoulTheme.BG_PANEL,
                    radius: int = SoulTheme.RADIUS_PANEL, alpha: int = 220) -> None:
        """绘制毛玻璃面板（外层壳 + 内层芯 + 发丝边框 + 顶部受光高光）。

        结构（Double-Bezel，参考高级感设计规范）：
        - 外层壳：白 3% 底色 + 白 10% 发丝外框（"光感"描边而非实色线框）；
        - 内层芯：内容底色（color/alpha）+ 内圆角 + 顶部 1px 受光高光，
          面板如"玻璃板嵌在托盘中"而非平铺在背景上。
        """
        w, h = rect.width, rect.height
        pad = 5
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        # 外层壳
        pygame.draw.rect(surf, SoulTheme.SHELL_BG, (0, 0, w, h), border_radius=radius)
        pygame.draw.rect(surf, SoulTheme.HAIRLINE, (0, 0, w, h),
                         width=1, border_radius=radius)
        # 内层芯
        inner = pygame.Rect(pad, pad, w - pad * 2, h - pad * 2)
        pygame.draw.rect(surf, (*color, alpha), inner,
                         border_radius=max(4, radius - pad))
        # 顶部受光高光（避开圆角端点）
        pygame.draw.line(surf, SoulTheme.INSET_HIGHLIGHT,
                         (pad + 6, pad + 1), (w - pad - 6, pad + 1))
        self.screen.blit(surf, (rect.x, rect.y))

    # ── HUD 半透明圆角矩形（SRCALPHA 安全） ──
    def alpha_rect(self, rect: pygame.Rect, color: Tuple[int, ...], radius: int = 0) -> None:
        """半透明圆角矩形：必须经 SRCALPHA 临时表面绘制（主屏无逐像素 alpha）。"""
        if len(color) == 3:
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)
            return
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, color, s.get_rect(), border_radius=radius)
        self.screen.blit(s, rect.topleft)

    # ── HUD 辉光描边（霓虹泛光） ──
    def glow_rect(self, rect: pygame.Rect, color: Tuple[int, ...],
                  radius: int = SoulTheme.RADIUS_BTN, glow: int = 2) -> None:
        """多层半透明叠加描边，产生霓虹泛光感（HUD 核心语言）。"""
        base = color[:3] if len(color) >= 3 else SoulTheme.ACCENT
        for i in range(glow, 0, -1):
            a = int(26 * (glow - i + 1) / glow)
            r = rect.inflate(i * 4, i * 4)
            self.alpha_rect(r, (*base, a), radius=radius + i * 2)

    # ── HUD 四角角标（瞄准镜） ──
    def corner_brackets(self, rect: pygame.Rect,
                        color: Tuple[int, int, int] = SoulTheme.BORDER_LIGHT,
                        length: int = 12, width: int = 2) -> None:
        """四角框标（瞄准镜角标）：HUD 语言标志性元素。"""
        x, y, w, h = rect
        pts = [
            [(x, y + length), (x, y), (x + length, y)],
            [(x + w - length, y), (x + w, y), (x + w, y + length)],
            [(x, y + h - length), (x, y + h), (x + length, y + h)],
            [(x + w - length, y + h), (x + w, y + h), (x + w, y + h - length)],
        ]
        for seg in pts:
            pygame.draw.lines(self.screen, color, False, seg, width)

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
              align: str = "left", anchor: Optional[str] = None) -> pygame.Rect:
        """绘制文字标签，返回文字区域。anchor 支持 Rect 锚点（topright/center/…），优先于 align。"""
        surf = font.render(text, True, color)
        rect = surf.get_rect()
        if anchor:
            setattr(rect, anchor, (x, y))
        elif align == "center":
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
                       line_width: int = 2, dot_radius: int = 3,
                       offset=(0, 0)) -> None:
    """
    在 surface 上绘制 MediaPipe 33 点骨架。

    Args:
        landmarks: 33 个关键点字典列表，每个含 x, y, visibility
        frame_w, frame_h: 摄像头帧的实际尺寸（用于坐标映射）
        offset: (ox, oy) 画面在 surface 上的位置（预览框偏移）
    """
    if not landmarks or len(landmarks) < 33:
        return

    ox, oy = offset
    # 映射到屏幕坐标：直接按归一化坐标做 x 镜像（与摄像头画面/手部骨架
    # 同方向），骨架显示在真人的实际位置，随身体移动而移动。
    # 注意：不再做"肩膀线平移到画面底端"的 dx/dy 重定位——那会把整个
    # 骨架（含脸 0-10 / 肩 11-12）定死在预览框底部，不跟随真人。
    points = []
    for lm in landmarks:
        px = int(ox + (1.0 - lm.get("x", 0)) * frame_w)
        py = int(oy + lm.get("y", 0) * frame_h)
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

    # ── 手部增强（轻量版：不依赖新模型，用 Pose 手部 5 点表达手型） ──
    # 手部关键点：腕 15/16、小指 17/18、食指 19/20、拇指 21/22
    # 三根手指线异色加粗 + 指尖圆点 + 腕部张开度状态圆环（绿=张开 / 黄=半开 / 红=握拳）
    for wrist, pinky, index, thumb in ((15, 17, 19, 21), (16, 18, 20, 22)):
        wp = points[wrist]
        if wp[2] <= 0.5:
            continue
        for tip, col in ((thumb, (255, 200, 80)),       # 拇指：琥珀
                         (index, (84, 190, 255)),       # 食指：亮蓝
                         (pinky, (255, 140, 160))):     # 小指：粉红
            tp = points[tip]
            if tp[2] > 0.5:
                pygame.draw.line(surface, col, (wp[0], wp[1]), (tp[0], tp[1]),
                                 line_width + 2)
                pygame.draw.circle(surface, col, (tp[0], tp[1]), dot_radius + 2)
                pygame.draw.circle(surface, (255, 255, 255), (tp[0], tp[1]),
                                   dot_radius + 1, 1)
        # 张开度：食指-小指间距 / 框宽 → 0~1
        ip, pp = points[index], points[pinky]
        if ip[2] > 0.5 and pp[2] > 0.5:
            spread = ((ip[0] - pp[0]) ** 2 + (ip[1] - pp[1]) ** 2) ** 0.5
            openness = max(0.0, min(1.0, spread / (0.28 * frame_w)))
            ring_col = (0, 226, 196) if openness > 0.5 else \
                       (255, 196, 70) if openness > 0.22 else (255, 76, 70)
            rr = 7 + int(9 * openness)
            ring = pygame.Surface((rr * 2, rr * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*ring_col, 160), (rr, rr), rr, 2)
            surface.blit(ring, (wp[0] - rr, wp[1] - rr))


# ═══════════════════════════════════════════════════════
# 运动可视化
# ═══════════════════════════════════════════════════════

# MediaPipe Hands 21 点骨架连线（每只手，虚拟博主式细骨架）
HAND_CONNECTIONS = [
    # 拇指：腕0 → 根1 → 掌2 → 指节3 → 尖4
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 食指：腕0 → 根5 → 中6 → 末7 → 尖8
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指：腕0 → 根9 → 中10 → 末11 → 尖12
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 无名指：腕0 → 根13 → 中14 → 末15 → 尖16
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指：腕0 → 根17 → 中18 → 末19 → 尖20
    (0, 17), (17, 18), (18, 19), (19, 20),
    # 指根横连（手掌）
    (5, 9), (9, 13), (13, 17),
]


def draw_hand_skeleton(surface: pygame.Surface, hand_pts: List[dict],
                       frame_w: int, frame_h: int,
                       line_color: Tuple[int, int, int] = (255, 200, 80),
                       line_width: int = 2, dot_radius: int = 2,
                       offset=(0, 0)) -> None:
    """
    在 surface 上绘制一只手（MediaPipe Hands 21 点细骨架）。

    手部坐标与 Pose 同一坐标系（归一化 0~1 相对整帧），
    绘制时做 x 镜像（与画面/骨架同向）并加上预览框偏移。

    Args:
        hand_pts: 21 个关键点字典列表，每个含 x, y
        frame_w, frame_h: 摄像头帧尺寸
        offset: (ox, oy) 画面在 surface 上的位置
    """
    if not hand_pts or len(hand_pts) < 21:
        return
    ox, oy = offset
    pts = []
    for lm in hand_pts:
        px = int(ox + (1.0 - lm.get("x", 0)) * frame_w)
        py = int(oy + lm.get("y", 0) * frame_h)
        pts.append((px, py))
    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            pygame.draw.line(surface, line_color, pts[a], pts[b], line_width)
    for px, py in pts:
        pygame.draw.circle(surface, line_color, (px, py), dot_radius)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), max(1, dot_radius - 1))


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
            # 统一做 x 镜像（与 draw_pose_skeleton / draw_hand_skeleton 一致）
            cx = int((1.0 - (left_hip["x"] + right_hip["x"]) / 2) * frame_w)
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
