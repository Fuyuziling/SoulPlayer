#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 dist/ 里的 exe 是否与当前源码同步。

用法（项目根目录）：
    py check_build_sync.py          # 检查并报告
    py check_build_sync.py -v       # 额外列出落后的 commit 与构建信息全文

退出码：
    0  已同步（exe 由当前 HEAD 打出，且构建后源码无改动）
    1  已过期（需要重新打包）
    2  尚未打包（dist/SoulPlayer 或 BUILD_INFO.txt 不存在）

为什么需要这个脚本：
    PyInstaller 打的是源码的**快照**。改完代码不重打包，用户双击 exe
    体验到的仍是旧逻辑——本项目已发生过一次：10-31 的抖动容忍与 11-03 的
    灵敏度调整都未进包，实机验证时误以为"改了没效果"。
"""

import datetime
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_DIR / "dist" / "SoulPlayer"
EXE_PATH = DIST_DIR / "SoulPlayer.exe"
INFO_PATH = DIST_DIR / "BUILD_INFO.txt"

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def _load_dirty_filter():
    """
    复用 build_exe 的"影响 exe 的改动"过滤器，保证打包与检查口径一致。
    （两份逻辑各写一遍迟早漂移，这是本项目已经吃过亏的模式。）
    """
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        from build_exe import filter_build_relevant, relevant_dirty_files
        return relevant_dirty_files, filter_build_relevant
    except Exception:
        return None, None


def git(*args):
    """执行 git 命令，失败返回空串。"""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def read_build_info():
    """解析 BUILD_INFO.txt 为 {键: 值}。解析失败返回 None。"""
    if not INFO_PATH.exists():
        return None
    data = {}
    try:
        for line in INFO_PATH.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                data[key.strip()] = val.strip()
    except Exception:
        return None
    return data


def main():
    if not EXE_PATH.exists():
        print("[未打包] dist/SoulPlayer/SoulPlayer.exe 不存在。")
        print("         运行：py build_exe.py")
        return 2

    if not INFO_PATH.exists():
        print("[已过期] exe 存在，但没有构建指纹 BUILD_INFO.txt。")
        print("         说明它由旧版 build_exe.py 打出，无法判断对应哪份源码。")
        print("         运行：py build_exe.py")
        return 1

    info = read_build_info()
    if not info or info.get("源码 commit", "unknown") == "unknown":
        print("[已过期] BUILD_INFO.txt 无法解析出 commit。请重新打包。")
        return 1

    exe_commit = info["源码 commit"]
    head_commit = git("rev-parse", "HEAD")
    dirty_filter, path_filter = _load_dirty_filter()
    if dirty_filter is not None:
        dirty_now = dirty_filter()
    else:
        # 兜底：不解析 porcelain 的列宽前缀（strip 会破坏首行偏移）
        dirty_now = []
        for cmd in (("diff", "--name-only", "HEAD"),
                    ("ls-files", "--others", "--exclude-standard")):
            out = git(*cmd)
            if out:
                dirty_now.extend(out.splitlines())

    print("=" * 62)
    print("  SoulPlayer exe 同步检查")
    print("=" * 62)
    print(f"  打包时间    : {info.get('打包时间', '?')}")
    print(f"  exe 的 commit: {info.get('commit 简称', exe_commit[:7])}  {info.get('commit 说明', '')}")
    print(f"  当前 HEAD    : {head_commit[:7] if head_commit else '?'}  {git('log', '-1', '--format=%s')}")
    print(f"  打包模式    : {info.get('打包模式', '?')}")
    print(f"  打包时工作区: {info.get('工作区脏', '?')}")
    print()

    need_rebuild = False
    reasons = []

    # 1. commit 不一致 → 先看落后的提交里有没有**影响 exe 的文件**。
    #    只提交文档/测试/工具不该报过期，否则每次记笔记都要重打包 3 分钟。
    if head_commit and exe_commit != head_commit:
        changed = git("log", "--name-only", "--format=",
                      f"{exe_commit}..HEAD").splitlines()
        relevant = path_filter(changed) if path_filter else changed
        behind = git("log", "--oneline", f"{exe_commit}..HEAD")
        n_behind = len(behind.splitlines()) if behind else 0
        if relevant:
            need_rebuild = True
            reasons.append("构建之后有影响 exe 内容的新提交")
            if behind:
                print("  exe 缺少以下 commit（涉及进包文件）：")
                for line in behind.splitlines():
                    print(f"    - {line}")
                print()
                if VERBOSE:
                    print("  涉及文件：")
                    for f in relevant:
                        print(f"    · {f}")
                    print()
            else:
                print("  （无法列出具体 commit）")
                print()
        else:
            print(f"  [INFO] HEAD 比 exe 新 {n_behind} 个提交，"
                  f"但仅涉及文档/测试/工具，exe 内容无差异。")
            print()
    elif not head_commit:
        reasons.append("无法读取当前 HEAD（不在 git 仓库内？）")

    # 2. 打包时工作区是脏的 → 那份 exe 不对应任何 commit，不可复现
    if info.get("工作区脏", "").startswith("是"):
        need_rebuild = True
        reasons.append("打包时工作区有未提交改动，exe 内容无法用 commit 复现")

    # 3. 现在有影响 exe 的未提交改动，且文件比 BUILD_INFO 新 → 改了没打包
    if dirty_now:
        try:
            info_mtime = INFO_PATH.stat().st_mtime
        except OSError:
            info_mtime = 0.0
        newer = []
        for rel in dirty_now:
            p = PROJECT_DIR / rel
            try:
                if p.is_file() and p.stat().st_mtime > info_mtime:
                    newer.append(rel)
            except OSError:
                pass
        if newer:
            need_rebuild = True
            reasons.append(
                f"打包后有 {len(newer)} 个进包文件被修改且未重新打包"
            )
            if VERBOSE:
                for f in newer:
                    print(f"    · {f}")
                print()

    # 4. 关键源文件 mtime 晚于打包时间（覆盖"改了又提交回去"的隐蔽情形）
    try:
        info_mtime = INFO_PATH.stat().st_mtime
        watched = [
            "main.py", "motion_capture.py", "gesture_presets.py",
            "custom_preset.py", "music_player.py", "voice_match.py",
            "wave_learning.py", "wave_learning_rules.py",
            "presets/default_presets.json",
        ]
        stale_files = []
        for rel in watched:
            p = PROJECT_DIR / rel
            if p.is_file() and p.stat().st_mtime > info_mtime + 1.0:
                stale_files.append(rel)
        if stale_files and not need_rebuild:
            need_rebuild = True
            reasons.append(f"{len(stale_files)} 个关键源文件比 exe 更新")
        if stale_files and VERBOSE:
            print("  比 exe 更新的源文件：")
            for f in stale_files:
                mt = datetime.datetime.fromtimestamp(
                    (PROJECT_DIR / f).stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M")
                print(f"    - {f}  ({mt})")
            print()
    except OSError:
        pass

    if need_rebuild:
        print("[已过期] exe 落后于源码，需要重新打包。")
        for r in reasons:
            print(f"         · {r}")
        print()
        print("  重新打包：py build_exe.py     （约 3 分钟）")
        print("  打包后验证：py smoke_test.py --exe")
        return 1

    print("[已同步] exe 由当前 HEAD 打出，且构建后源码无改动。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
