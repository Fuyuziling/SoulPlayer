#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer .exe 打包脚本

用法：在项目根目录运行
    py build_exe.py

说明：
- 使用 PyInstaller --onedir 模式（单文件夹）
- 自动包含所有资源文件（模型、预设、语音模型、音乐）
- 处理 MediaPipe / pygame / Vosk 等依赖的隐藏导入
- 输出到 SoulPlayer/dist/SoulPlayer/
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ───────────────────────── 配置 ─────────────────────────

PROJECT_DIR = Path(__file__).parent.resolve()  # SoulPlayer 目录
ENTRY_SCRIPT = PROJECT_DIR / "run.py"
DIST_DIR = PROJECT_DIR / "dist"
BUILD_DIR = PROJECT_DIR / "build"
OUTPUT_NAME = "SoulPlayer"

# 需要打包进 exe 的资源文件/目录
# 格式：(源路径, 目标目录)
RESOURCES = [
    (PROJECT_DIR / "pose_landmarker_full.task", "."),
    (PROJECT_DIR / "presets" / "default_presets.json", "presets"),
    (PROJECT_DIR / "voice_triggers.json", "."),
    (PROJECT_DIR / "voice_templates.json", "."),
    (PROJECT_DIR / "voice_templates", "voice_templates"),
    (PROJECT_DIR / "vosk-model", "vosk-model"),
    (PROJECT_DIR / "custom_presets", "custom_presets"),
    (PROJECT_DIR / "music", "music"),
]

# 隐藏导入（PyInstaller 可能无法自动检测到的模块）
HIDDEN_IMPORTS = [
    # 项目子模块（main.py 中用 try/except 动态导入）
    "motion_capture",
    "gesture_presets",
    "music_player",
    "custom_preset",
    "voice_trigger",
    "ui_theme",
    "ui_animations",
    # OpenCV / NumPy
    "cv2",
    "numpy",
    "numpy.core._dtype_ctypes",
    # pygame
    "pygame",
    "pygame.mixer",
    "pygame.mixer.music",
    # MediaPipe（Tasks API 子模块）
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
    "mediapipe.python._framework_bindings",
    "google.protobuf",
    # Vosk / 语音
    "vosk",
    "sounddevice",
    # tkinter（文件对话框）
    "tkinter",
    "tkinter.filedialog",
    "tkinter.commondialog",
    "tkinter.dialog",
    # 其他可能被动态导入的模块
    "queue",
    "json",
    "threading",
    "collections",
    "traceback",
    "inspect",
]

# ───────────────────────── 辅助函数 ─────────────────────────

def clean_previous_builds():
    """清理旧的打包输出，避免冲突。"""
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            print(f"[INFO] 清理旧目录: {d}")
            shutil.rmtree(d, ignore_errors=True)

def verify_resources():
    """检查所有资源文件是否存在。"""
    missing = []
    for src, _dest in RESOURCES:
        if not src.exists():
            missing.append(str(src))
    if missing:
        print("[WARN] 以下资源文件/目录不存在，将跳过：")
        for m in missing:
            print(f"       - {m}")
    return missing

def build_command():
    """构建 PyInstaller 命令行参数。"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",           # 不确认覆盖
        "--onedir",              # 单文件夹模式（启动快、资源管理方便）
        "--console",             # 保留控制台窗口（便于调试和查看日志）
        "--name", OUTPUT_NAME,
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        # 降低警告级别，减少无关输出
        "--log-level", "WARN",
    ]

    # 添加资源文件
    for src, dest in RESOURCES:
        if src.exists():
            # Windows 上 --add-data 用分号分隔
            cmd.extend(["--add-data", f"{src};{dest}"])

    # 添加隐藏导入
    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    # 尝试自动收集 mediapipe 的所有子模块和数据
    cmd.extend(["--collect-all", "mediapipe"])
    cmd.extend(["--collect-all", "pygame"])
    cmd.extend(["--collect-all", "vosk"])
    cmd.extend(["--collect-all", "sounddevice"])

    # 入口脚本
    cmd.append(str(ENTRY_SCRIPT))

    return cmd

def copy_post_build_files():
    """
    打包后，将额外的文件复制到输出目录。
    如 run.bat、README 等。
    """
    output_dir = DIST_DIR / OUTPUT_NAME
    if not output_dir.exists():
        print(f"[WARN] 输出目录不存在: {output_dir}")
        return

    # 复制 run.bat（用户习惯双击 .bat）
    bat_src = PROJECT_DIR / "run.bat"
    if bat_src.exists():
        bat_dst = output_dir / "run.bat"
        shutil.copy2(bat_src, bat_dst)
        print(f"[INFO] 已复制: run.bat")

    # 复制 README
    readme_src = PROJECT_DIR / "README.md"
    if readme_src.exists():
        shutil.copy2(readme_src, output_dir / "README.md")
        print(f"[INFO] 已复制: README.md")

    # 复制 requirements.txt（供参考）
    req_src = PROJECT_DIR / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, output_dir / "requirements.txt")
        print(f"[INFO] 已复制: requirements.txt")

def print_summary():
    """打印打包结果摘要。"""
    output_dir = DIST_DIR / OUTPUT_NAME
    exe_path = output_dir / f"{OUTPUT_NAME}.exe"

    print()
    print("=" * 60)
    print("  SoulPlayer 打包完成")
    print("=" * 60)
    print()

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  可执行文件: {exe_path}")
        print(f"  文件大小:   {size_mb:.1f} MB")
    else:
        print(f"  [WARN] 未找到 exe 文件: {exe_path}")

    print(f"  输出目录:   {output_dir}")
    print()
    print("  使用方式:")
    print("  1. 将整个 SoulPlayer 文件夹复制到目标电脑")
    print("  2. 双击 SoulPlayer.exe 或 run.bat 运行")
    print("  3. 如有问题，查看 error.log 和控制台输出")
    print()
    print("=" * 60)

# ───────────────────────── 主流程 ─────────────────────────

def main():
    print("=" * 60)
    print("  SoulPlayer .exe 打包脚本")
    print("=" * 60)
    print()

    # 1. 验证入口脚本
    if not ENTRY_SCRIPT.exists():
        print(f"[ERROR] 入口脚本不存在: {ENTRY_SCRIPT}")
        sys.exit(1)

    # 2. 验证资源
    verify_resources()

    # 3. 清理旧构建
    clean_previous_builds()

    # 4. 构建命令
    cmd = build_command()
    print("[INFO] PyInstaller 命令:")
    print("  " + " ".join(cmd))
    print()

    # 5. 执行打包
    print("[INFO] 开始打包，这可能需要几分钟...")
    print()
    try:
        result = subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("[ERROR] PyInstaller 未找到。请确保已安装：")
        print("        py -m pip install pyinstaller")
        sys.exit(1)

    if result.returncode != 0:
        print(f"[ERROR] PyInstaller 退出码: {result.returncode}")
        print("[INFO] 尝试以 ERROR 级别日志重新运行以查看详情...")
        sys.exit(1)

    # 6. 复制额外文件
    copy_post_build_files()

    # 7. 打印摘要
    print_summary()

if __name__ == "__main__":
    main()
