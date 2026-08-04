#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoulPlayer 启动器
用于在 Windows 命令行中启动主程序，保留错误信息并暂停
"""

import sys
import os
import traceback
import time

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("  SoulPlayer 启动器")
print("  基于肢体动作捕捉的智能音乐播放器")
print("=" * 50)
print()

# 检查 Python 是否可用
print("[INFO] 检查 Python 环境...")
print(f"       Python 版本: {sys.version.split()[0]}")
print(f"       工作目录: {os.getcwd()}")
print()

# 检查依赖是否安装
print("[INFO] 检查依赖...")
missing = []
try:
    import pygame
    print("       pygame: OK")
except ImportError:
    missing.append("pygame")
    print("       pygame: NOT FOUND")

try:
    import numpy
    print("       numpy: OK")
except ImportError:
    missing.append("numpy")
    print("       numpy: NOT FOUND")

try:
    import cv2
    print("       opencv-python: OK")
except ImportError:
    missing.append("opencv-python")
    print("       opencv-python: NOT FOUND")

try:
    import mediapipe
    print("       mediapipe: OK")
except ImportError:
    missing.append("mediapipe")
    print("       mediapipe: NOT FOUND")

print()

if missing:
    print("[ERROR] 缺少依赖，请先运行 install.bat")
    print(f"       缺少: {', '.join(missing)}")
    print()
    input("按回车键退出...")
    sys.exit(1)

print("[INFO] 所有依赖已安装，正在启动主程序...")
print()
print("-" * 50)
print()

# 运行主程序
exit_code = 0
try:
    from main import main
    main()
except SystemExit as e:
    exit_code = e.code if isinstance(e.code, int) else 0
except Exception as e:
    exit_code = 1
    print()
    print("-" * 50)
    print("[ERROR] 程序发生异常：")
    print()
    traceback.print_exc()
    print()
    print("-" * 50)
    print()
    # 同时将错误写入日志文件
    try:
        with open("error.log", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 启动器捕获异常:\n")
            traceback.print_exc(file=f)
            f.write(f"{'='*50}\n")
    except Exception:
        pass

print()
print("-" * 50)
if exit_code != 0:
    print(f"[INFO] 程序异常退出，错误代码: {exit_code}")
else:
    print("[INFO] 程序已退出")
print("-" * 50)
print()
input("按回车键关闭窗口...")
