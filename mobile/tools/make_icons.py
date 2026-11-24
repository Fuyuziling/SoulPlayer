#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Android 图标与启动图。

品牌标记复用 PC 端的五根波形柱（tools/make_icon.py 里的 BARS），
只把配色换成手机端的暖琥珀体系——图标要和打开后的界面是同一个产品，
而不是"另一套皮肤"。

用法：
    py mobile/tools/make_icons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pygame

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]            # mobile/tools -> mobile -> 项目根
sys.path.insert(0, str(ROOT / "tools"))

import make_icon as brand          # noqa: E402  只借品牌资产，不借配色

RES = ROOT / "mobile" / "android" / "app" / "src" / "main" / "res"

# 与 mobile/src/ui/styles.css 的 --bg / --accent 保持一致
BG = (23, 21, 19)                 # --bg  #171513
ACCENT = (196, 149, 106)          # --accent #c4956a
ACCENT_DIM = (140, 104, 72)

BARS = brand.BARS

# 密度 -> 倍率
DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}

# 启动图基准尺寸（mdpi 下的竖屏 / 横屏）
SPLASH_PORT = (320, 480)
SPLASH_LAND = (480, 320)


def draw_bars(surf: pygame.Surface, size: int, color, margin_ratio: float) -> None:
    """在 size×size 的画布中央画五根柱。几何比例与 PC 图标一致。"""
    n = len(BARS)
    margin = size * margin_ratio
    usable = size - margin * 2
    slot = usable / n
    bar_w = max(1, int(slot * 0.52))
    gap = slot - bar_w
    for i, ratio in enumerate(BARS):
        h = max(2, int(usable * 0.78 * ratio))
        x = margin + i * slot + gap / 2
        y = (size - h) / 2
        pygame.draw.rect(
            surf, color, (round(x), round(y), bar_w, h),
            border_radius=max(1, bar_w // 2))


def icon_surface(size: int, round_icon: bool = False) -> pygame.Surface:
    """完整图标：底色 + 描边 + 波形柱（与 PC 端 assets/soulplayer.png 同构）。"""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    if round_icon:
        pygame.draw.circle(surf, BG, (size // 2, size // 2), size // 2)
        if size >= 48:
            pygame.draw.circle(surf, ACCENT_DIM, (size // 2, size // 2), size // 2 - 1,
                               width=max(1, size // 64))
    else:
        radius = max(2, int(size * 0.22))
        pygame.draw.rect(surf, BG, (0, 0, size, size), border_radius=radius)
        if size >= 48:
            pygame.draw.rect(surf, ACCENT_DIM, (1, 1, size - 2, size - 2),
                             width=max(1, size // 64), border_radius=radius - 1)

    draw_bars(surf, size, ACCENT, 0.20 if not round_icon else 0.22)
    return surf


def foreground_surface(size: int) -> pygame.Surface:
    """自适应图标前景：透明底，内容收在 72/108 安全区内。

    安全区是 66%，内容取 52%——太满会在圆形/方圆遮罩下被切掉柱子的两端，
    这是自适应图标最常见的翻车点（做出来在模拟器方角预览里好看，
    到真实圆形遮罩启动器上缺一块）。
    """
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    draw_bars(surf, size, ACCENT, 0.24)
    return surf


def splash_surface(w: int, h: int) -> pygame.Surface:
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill(BG)
    logo = min(w, h) * 0.34
    icon = icon_surface(int(logo))
    surf.blit(icon, ((w - icon.get_width()) // 2, (h - icon.get_height()) // 2))
    return surf


def save(surf: pygame.Surface, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, str(path))


def main() -> int:
    pygame.init()
    written = 0

    for bucket, scale in DENSITIES.items():
        base = int(48 * scale)
        d = RES / f"mipmap-{bucket}"
        save(icon_surface(base), d / "ic_launcher.png")
        save(icon_surface(base, round_icon=True), d / "ic_launcher_round.png")
        save(foreground_surface(int(108 * scale)), d / "ic_launcher_foreground.png")
        written += 3

        pw, ph = int(SPLASH_PORT[0] * scale), int(SPLASH_PORT[1] * scale)
        save(splash_surface(pw, ph), RES / f"drawable-port-{bucket}" / "splash.png")
        lw, lh = int(SPLASH_LAND[0] * scale), int(SPLASH_LAND[1] * scale)
        save(splash_surface(lw, lh), RES / f"drawable-land-{bucket}" / "splash.png")
        written += 2

    # 无密度后缀的兜底图（styles.xml 里 @drawable/splash 会命中它）
    save(splash_surface(480, 800), RES / "drawable" / "splash.png")
    written += 1

    # 自适应图标的背景色（部分启动器不读 adaptive 的 background 时仍然要有底色）
    (RES / "values" / "ic_launcher_background.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<resources>\n"
        f'    <color name="ic_launcher_background">#{BG[0]:02X}{BG[1]:02X}{BG[2]:02X}</color>\n'
        "</resources>\n",
        encoding="utf-8")
    written += 1

    print(f"[OK] 已写入 {written} 个资源到 {RES}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
