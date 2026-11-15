#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 SoulPlayer 图标资源（assets/soulplayer.ico + .png）

不引入 Pillow：用 pygame 渲染各尺寸位图，再按 ICO 容器格式封装
（Vista 起 ICO 允许直接内嵌 PNG 数据，单文件即可多尺寸）。

用法：
    py tools/make_icon.py
"""

from __future__ import annotations

import struct
import sys
from io import BytesIO
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "assets"

BG = (6, 11, 15)          # SoulTheme.BG_DEEP
ACCENT = (0, 226, 196)    # SoulTheme.ACCENT
ACCENT_DIM = (0, 132, 118)
SIZES = (16, 24, 32, 48, 64, 128, 256)

# 五根波形柱的相对高度（0~1），中间高两端低，一眼能看出是音频
BARS = (0.42, 0.72, 1.00, 0.66, 0.38)


def render(size: int) -> pygame.Surface:
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))

    radius = max(2, int(size * 0.22))
    pygame.draw.rect(surf, BG, (0, 0, size, size), border_radius=radius)

    # 内描边：小尺寸下不要，否则糊成一团
    if size >= 48:
        pygame.draw.rect(
            surf, ACCENT_DIM, (1, 1, size - 2, size - 2),
            width=max(1, size // 64), border_radius=radius - 1)

    n = len(BARS)
    margin = size * 0.20
    usable = size - margin * 2
    slot = usable / n
    bar_w = max(1, int(slot * 0.52))
    gap = slot - bar_w

    for i, ratio in enumerate(BARS):
        h = max(2, int(usable * 0.78 * ratio))
        x = margin + i * slot + gap / 2
        y = (size - h) / 2
        pygame.draw.rect(
            surf, ACCENT, (round(x), round(y), bar_w, h),
            border_radius=max(1, bar_w // 2))
    return surf


def png_bytes(surf: pygame.Surface) -> bytes:
    buf = BytesIO()
    pygame.image.save(surf, buf, "icon.png")
    return buf.getvalue()


def write_ico(images: dict[int, bytes], path: Path) -> None:
    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = b""
    payload = b""
    for size, data in images.items():
        entries += struct.pack(
            "<BBBBHHII",
            size % 256, size % 256,  # 0 表示 256
            0, 0, 1, 32, len(data), offset)
        offset += len(data)
        payload += data
    path.write_bytes(header + entries + payload)


def main() -> int:
    pygame.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = {}
    for size in SIZES:
        images[size] = png_bytes(render(size))

    ico_path = OUT_DIR / "soulplayer.ico"
    png_path = OUT_DIR / "soulplayer.png"
    write_ico(images, ico_path)
    png_path.write_bytes(images[256])

    print(f"[OK] {ico_path}  ({ico_path.stat().st_size} 字节, {len(SIZES)} 个尺寸)")
    print(f"[OK] {png_path}  ({png_path.stat().st_size} 字节)")
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
