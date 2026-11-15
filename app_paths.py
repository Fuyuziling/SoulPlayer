#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer 应用路径层

把"程序放在哪"和"数据写到哪"彻底解耦。正式版安装到 Program Files 后
安装目录是只读的（普通用户无写权限），若仍把用户数据写在 exe 旁边，
表现为：录了自定义动作、重启就没了；日志也静默失败，出问题完全查不到。

三种运行态（按优先级）：

1. 环境变量覆盖  SOULPLAYER_DATA_DIR=<路径>
   测试与运维用，最高优先级，不再探测。
2. 开发态（未打包）  数据根 = 源码目录，行为与之前完全一致，
   保证 tests/ 与现有数据不受影响。
3. 打包态
   - 便携模式：exe 旁边存在 portable.flag → 数据根 = exe 所在目录
     （U 盘、免安装分发用，需目录可写）
   - 安装模式：数据根 = %LOCALAPPDATA%\\SoulPlayer

升级迁移：目标目录为空时，从旧位置（安装目录、项目目录、_internal、
旧版 ~/.soulplayer）一次性拷入，避免"版本一升级，录制全丢"。
"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

APP_NAME = "SoulPlayer"
ENV_DATA_DIR = "SOULPLAYER_DATA_DIR"
PORTABLE_FLAG = "portable.flag"
LOG_DIR_NAME = "logs"

# 旧版把学习数据写在用户目录，迁移时需识别
_LEGACY_HOME_DIR = Path.home() / ".soulplayer"

_MIGRATE_MARKER = ".migrated"


def is_frozen() -> bool:
    """是否运行在 PyInstaller 打包产物里。"""
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    """可执行文件（或源码）所在目录。"""
    if is_frozen():
        try:
            return Path(sys.executable).resolve().parent
        except OSError:
            return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def resource_dir() -> Path:
    """只读资源目录：打包后是 _internal，开发态是源码目录。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def data_root() -> Path:
    """
    用户数据根目录。所有需要写入的东西（预设、语音模板、学习数据、
    日志、诊断包）都挂在它下面。
    """
    env = os.environ.get(ENV_DATA_DIR, "").strip()
    if env:
        return Path(env).expanduser()

    if not is_frozen():
        # 开发态：沿用源码目录，零行为变化
        return Path(__file__).resolve().parent

    exe_dir = install_dir()
    if (exe_dir / PORTABLE_FLAG).is_file():
        return exe_dir

    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_NAME
    return Path.home() / f".{APP_NAME.lower()}"


def reset_cache() -> None:
    """清掉 data_root 缓存（切换便携/安装态或测试时用）。"""
    data_root.cache_clear()


def ensure_dir(path: Path) -> Path:
    """尽力创建目录；失败不抛异常，由调用方按"不可写"处理。"""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return path


def logs_dir() -> Path:
    return ensure_dir(data_root() / LOG_DIR_NAME)


def _has_content(d: Path) -> bool:
    try:
        return d.is_dir() and any(d.iterdir())
    except OSError:
        return False


def legacy_candidates(name: str) -> List[Path]:
    """
    旧数据可能残留的位置，按优先级排列。
    打包前 dist/SoulPlayer 位于项目内，所以 exe 上两级就是项目目录。
    """
    out: List[Path] = []
    if is_frozen():
        exe_dir = install_dir()
        out.append(exe_dir.parent.parent / name)
        out.append(exe_dir / name)
    out.append(resource_dir() / name)
    return out


def _copy_tree(src: Path, dst: Path) -> int:
    """把 src 内容合并进 dst（不覆盖已存在项），返回拷贝条目数。"""
    copied = 0
    for item in src.iterdir():
        dest = dst / item.name
        if dest.exists():
            continue
        try:
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
            copied += 1
        except OSError:
            continue
    return copied


def data_dir(name: str, migrate_from: Optional[Iterable[Path]] = None) -> Path:
    """
    返回 `<data_root>/<name>` 并确保存在。

    若目标为空，则从 migrate_from 中第一个有内容的位置迁移内容，
    连同同级配套文件 `<name>.json`（如 voice_templates.json）。
    迁移失败时仍返回目标目录（宁可空着，也不写回只读目录）。
    """
    target = ensure_dir(data_root() / name)
    if _has_content(target):
        return target
    if not migrate_from:
        return target

    for src in migrate_from:
        try:
            src = Path(src)
            if src.resolve() == target.resolve() or not _has_content(src):
                continue
        except OSError:
            continue

        copied = _copy_tree(src, target)
        # 配套映射文件与目录同级，例如 voice_templates.json
        try:
            sidecar_src = src.parent / f"{name}.json"
            sidecar_dst = target.parent / f"{name}.json"
            if sidecar_src.is_file() and not sidecar_dst.exists():
                shutil.copy2(sidecar_src, sidecar_dst)
                copied += 1
        except OSError:
            pass

        if copied:
            print(f"[数据迁移] {name}: {src} -> {target}（{copied} 项）")
        return target

    return target


def migrate_legacy_home_data(name: str) -> Path:
    """
    迁移旧版写在用户目录的数据（如 ~/.soulplayer/learning_data）。
    仅在目标为空且旧目录有内容时执行。
    """
    target = ensure_dir(data_root() / name)
    if _has_content(target):
        return target
    legacy = _LEGACY_HOME_DIR / name
    if not _has_content(legacy):
        return target
    copied = _copy_tree(legacy, target)
    if copied:
        print(f"[数据迁移] {name}: {legacy} -> {target}（{copied} 项）")
    return target


def describe() -> str:
    """一行式路径摘要，写进日志便于排查"数据到底存在哪"。"""
    mode = "开发态" if not is_frozen() else (
        "便携模式" if data_root() == install_dir() else "安装模式")
    return (f"运行态={mode} | 程序目录={install_dir()} | "
            f"数据根={data_root()} | 资源目录={resource_dir()}")
